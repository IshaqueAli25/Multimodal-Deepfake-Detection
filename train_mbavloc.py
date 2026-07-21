"""
train_mbavloc.py
Training script for MB-AV-Loc.

Usage:
    python train_mbavloc.py --config mbavloc_train.yaml
"""

import os
import sys
import yaml
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
import random

# add repo root to path for libs/
sys.path.insert(0, os.path.dirname(__file__))

from dataset import LAVDFCachedDataset, collate_fn


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_model(cfg):
    """Import and build MB-AV-Loc model via registry."""
    # import to trigger registration
    from model import MBAVLocTransformer  # noqa
    from libs.modeling.models import make_meta_arch

    model_cfg = dict(cfg['model'])
    model_cfg['train_cfg'] = cfg['train_cfg']
    model_cfg['test_cfg'] = cfg['test_cfg']

    model = make_meta_arch(cfg['model_name'], **model_cfg)
    return model


def build_dataset(cfg, split, is_training):
    ds_cfg = cfg.get('dataset', {})
    dataset = LAVDFCachedDataset(
        cache_dir=cfg['cache_dir'],
        anno_file=cfg['anno_file'],
        split=split,
        feat_fps=ds_cfg.get('feat_fps', 25),
        max_seq_len=ds_cfg.get('max_seq_len', 1024),
        trunc_thresh=ds_cfg.get('trunc_thresh', 0.5),
        crop_ratio=ds_cfg.get('crop_ratio', (0.9, 1.0)),
        is_training=is_training,
    )
    return dataset


def build_optimizer(cfg, model):
    opt_cfg = cfg['opt']
    params = [p for p in model.parameters() if p.requires_grad]

    print(f"[Optimizer] {sum(p.numel() for p in params):,} trainable parameters")

    if opt_cfg['type'] == 'AdamW':
        optimizer = AdamW(
            params,
            lr=opt_cfg['learning_rate'],
            weight_decay=opt_cfg['weight_decay'],
        )
    else:
        optimizer = SGD(
            params,
            lr=opt_cfg['learning_rate'],
            momentum=opt_cfg.get('momentum', 0.9),
            weight_decay=opt_cfg['weight_decay'],
        )

    return optimizer


def build_scheduler(cfg, optimizer):
    opt_cfg = cfg['opt']
    total_epochs = opt_cfg['epochs']
    warmup_epochs = opt_cfg.get('warmup_epochs', 5)

    if opt_cfg.get('warmup', False) and warmup_epochs > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

    return scheduler


def train_one_epoch(model, dataloader, optimizer, cfg, epoch, device):
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    num_batches = 0

    clip_grad = cfg['train_cfg'].get('clip_grad_l2norm', 1.0)

    for batch_idx, video_list in enumerate(dataloader):
        # move tensors to device
        for item in video_list:
            item['visual_feats'] = item['visual_feats'].to(device)
            item['audio_feats'] = item['audio_feats'].to(device)
            item['segments'] = item['segments'].to(device)
            item['labels'] = item['labels'].to(device)

        losses = model(video_list)

        loss = losses['final_loss']

        optimizer.zero_grad()
        loss.backward()

        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        total_loss += loss.item()
        total_cls += losses['cls_loss'].item()
        total_reg += losses['reg_loss'].item()
        num_batches += 1

        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / num_batches
            print(f"  [Epoch {epoch}] batch {batch_idx+1}/{len(dataloader)} "
                  f"loss={avg_loss:.4f} cls={total_cls/num_batches:.4f} "
                  f"reg={total_reg/num_batches:.4f}")

    avg_loss = total_loss / max(num_batches, 1)
    avg_cls = total_cls / max(num_batches, 1)
    avg_reg = total_reg / max(num_batches, 1)

    return avg_loss, avg_cls, avg_reg


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    all_results = []

    for video_list in dataloader:
        for item in video_list:
            item['visual_feats'] = item['visual_feats'].to(device)
            item['audio_feats'] = item['audio_feats'].to(device)
            item['segments'] = item['segments'].to(device)
            item['labels'] = item['labels'].to(device)

        results = model(video_list)
        all_results.extend(results)

    # count detections
    total_dets = sum(len(r['segments']) for r in all_results)
    print(f"  [Val] {len(all_results)} videos, {total_dets} detections")

    return all_results


def save_checkpoint(model, optimizer, scheduler, epoch, cfg, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': cfg,
    }, filename)
    print(f"  Saved checkpoint: {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get('init_rand_seed', 42))

    device = torch.device(cfg['devices'][0])
    print(f"Using device: {device}")

    # build model
    print("Building model...")
    model = build_model(cfg)
    model = model.to(device)

    # build datasets
    print("Building datasets...")
    train_dataset = build_dataset(cfg, cfg['train_split'], is_training=True)
    val_dataset = build_dataset(cfg, cfg['val_split'], is_training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['loader']['batch_size'],
        shuffle=True,
        num_workers=cfg['loader']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['loader']['batch_size'],
        shuffle=False,
        num_workers=cfg['loader']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # optimizer and scheduler
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    # training loop
    total_epochs = cfg['opt']['epochs']
    print(f"\nStarting training for {total_epochs} epochs...")
    print(f"Train: {len(train_dataset)} videos, Val: {len(val_dataset)} videos")
    print(f"Batch size: {cfg['loader']['batch_size']}")

    best_loss = float('inf')

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()

        avg_loss, avg_cls, avg_reg = train_one_epoch(
            model, train_loader, optimizer, cfg, epoch, device
        )

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{total_epochs-1} ({elapsed:.0f}s) "
              f"loss={avg_loss:.4f} cls={avg_cls:.4f} reg={avg_reg:.4f} lr={lr:.6f}")

        # save every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(
                cfg['output_folder'], f'epoch_{epoch:03d}.pth.tar'
            )
            save_checkpoint(model, optimizer, scheduler, epoch, cfg, ckpt_path)

        # validate every 10 epochs
        if (epoch + 1) % 10 == 0:
            validate(model, val_loader, device)

        # save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(cfg['output_folder'], 'best.pth.tar')
            save_checkpoint(model, optimizer, scheduler, epoch, cfg, ckpt_path)

    print("\nTraining complete.")


if __name__ == '__main__':
    main()
