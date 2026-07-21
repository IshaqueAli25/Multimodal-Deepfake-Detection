"""
Step 3: CLIP Visual-Only Baseline
==================================
This is your FIRST result — the visual-only row in your ablation table.

Architecture:
  Frozen CLIP ViT-B/16 → unfreeze LayerNorm only → CLS token → Linear(768, 2)

This script:
  1. Defines the CLIPBaseline model
  2. Creates a dataset from preprocessed face crops
  3. Trains with cross-entropy loss
  4. Evaluates and reports AUC, accuracy

Usage: python step3_clip_baseline.py --data data/processed --epochs 10 --batch_size 32

IMPORTANT: Run step2_preprocess_data.py first to create the face crops.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import CLIPVisionModel, CLIPImageProcessor
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from tqdm import tqdm


# ============================================================
# MODEL
# ============================================================
class CLIPDeepfakeBaseline(nn.Module):
    """
    Visual-only deepfake detection baseline.
    
    Architecture:
        Frozen CLIP ViT-B/16 (only LayerNorm unfrozen)
        → CLS token (768-dim)
        → Dropout
        → Linear classifier (768 → 2)
    
    Total trainable params: ~90K (0.03% of CLIP) + 1538 (classifier)
    """
    
    def __init__(self, dropout=0.1):
        super().__init__()
        
        # Load CLIP vision encoder
        print("Loading CLIP ViT-B/16...")
        self.clip = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch16",
            use_safetensors=True,
        )
        
        # ---- FREEZE everything ----
        for param in self.clip.parameters():
            param.requires_grad = False
        
        # ---- UNFREEZE only LayerNorm ----
        ln_count = 0
        for name, param in self.clip.named_parameters():
            if 'layer_norm' in name.lower() or 'layernorm' in name.lower():
                param.requires_grad = True
                ln_count += 1
        
        # Classifier head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, 2)  # 768 = CLIP ViT-B hidden dim
        
        # Print parameter summary
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        print(f"\n  Model Summary:")
        print(f"    Total parameters:     {total:>10,}")
        print(f"    Frozen parameters:    {frozen:>10,}")
        print(f"    Trainable parameters: {trainable:>10,} ({trainable/total*100:.3f}%)")
        print(f"    LayerNorm layers unfrozen: {ln_count}")
        print(f"    Classifier parameters:     {sum(p.numel() for p in self.classifier.parameters()):,}")
    
    def forward(self, pixel_values):
        """
        Args:
            pixel_values: (batch, 3, 224, 224) - preprocessed face images
        
        Returns:
            logits: (batch, 2) - real/fake logits
        """
        # Get CLIP features
        outputs = self.clip(pixel_values=pixel_values)
        
        # Use CLS token (first token)
        cls_token = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        
        # Classify
        cls_token = self.dropout(cls_token)
        logits = self.classifier(cls_token)
        
        return logits


# ============================================================
# DATASET
# ============================================================
class DeepfakeFaceDataset(Dataset):
    """
    Dataset of preprocessed face crops.
    
    Expects directory structure from step2:
        data/processed/
            real/video001/faces/frame_0000.jpg
            fake/video003/faces/frame_0000.jpg
    
    Each face image is one sample.
    Label: 0 = real, 1 = fake
    """
    
    def __init__(self, data_dir, split='train', transform=None, train_ratio=0.8):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Collect all face images with labels
        self.samples = []  # list of (image_path, label)
        
        label_map = {'real': 0, 'fake': 1}
        
        for label_name, label_idx in label_map.items():
            label_dir = self.data_dir / label_name
            if not label_dir.exists():
                print(f"  ⚠️  Directory not found: {label_dir}")
                continue
            
            for video_dir in sorted(label_dir.iterdir()):
                faces_dir = video_dir / "faces"
                if not faces_dir.exists():
                    continue
                
                for face_img in sorted(faces_dir.glob("*.jpg")):
                    self.samples.append((str(face_img), label_idx))
        
        # Split into train/val
        np.random.seed(42)  # Reproducible split
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(self.samples) * train_ratio)
        
        if split == 'train':
            selected = indices[:split_idx]
        else:
            selected = indices[split_idx:]
        
        self.samples = [self.samples[i] for i in selected]
        
        # Count labels
        labels = [s[1] for s in self.samples]
        n_real = labels.count(0)
        n_fake = labels.count(1)
        print(f"  {split} set: {len(self.samples)} images ({n_real} real, {n_fake} fake)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ============================================================
# TRAINING
# ============================================================
def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f"{total_loss / (batch_idx + 1):.4f}",
            'acc': f"{correct / total:.4f}"
        })
    
    return total_loss / len(dataloader), correct / total


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []
    total_loss = 0
    
    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)
        
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item()
        
        probs = F.softmax(logits, dim=1)[:, 1]  # probability of "fake"
        preds = logits.argmax(dim=1)
        
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    
    # Metrics
    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)
    
    return {
        'auc': auc,
        'accuracy': acc,
        'loss': avg_loss,
        'labels': all_labels,
        'probs': all_probs,
        'preds': all_preds
    }


# ============================================================
# MAIN
# ============================================================
def main(args):
    print("=" * 60)
    print("STEP 3: CLIP VISUAL-ONLY BASELINE")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # ---- Transforms ----
    # CLIP expects specific normalisation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],   # CLIP means
            std=[0.26862954, 0.26130258, 0.27577711]     # CLIP stds
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    # ---- Dataset ----
    print("\nLoading dataset...")
    train_dataset = DeepfakeFaceDataset(args.data, split='train', transform=train_transform)
    val_dataset = DeepfakeFaceDataset(args.data, split='val', transform=val_transform)
    
    if len(train_dataset) == 0:
        print("❌ No training data found. Run step2_preprocess_data.py first.")
        return
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # ---- Model ----
    print("\nBuilding model...")
    model = CLIPDeepfakeBaseline(dropout=args.dropout)
    model = model.to(device)
    
    # Use mixed precision for memory efficiency
    scaler = torch.amp.GradScaler('cuda')
    
    # ---- Optimizer ----
    # Only optimise trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # ---- Training Loop ----
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    best_auc = 0
    results_log = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, epoch)
        
        # Evaluate
        val_results = evaluate(model, val_loader, device)
        
        scheduler.step()
        
        # Log
        lr = optimizer.param_groups[0]['lr']
        print(f"\n  Epoch {epoch}/{args.epochs}:")
        print(f"    Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
        print(f"    Val Loss:   {val_results['loss']:.4f}  Val Acc:   {val_results['accuracy']:.4f}")
        print(f"    Val AUC:    {val_results['auc']:.4f}  LR: {lr:.6f}")
        
        results_log.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_results['loss'],
            'val_acc': val_results['accuracy'],
            'val_auc': val_results['auc'],
            'lr': lr
        })
        
        # Save best model
        if val_results['auc'] > best_auc:
            best_auc = val_results['auc']
            save_path = os.path.join(args.save_dir, "clip_baseline_best.pt")
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'val_results': {k: v for k, v in val_results.items() if k not in ['labels', 'probs', 'preds']}
            }, save_path)
            print(f"    ✅ New best AUC! Model saved to {save_path}")
    
    # ---- Final Results ----
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best Val AUC: {best_auc:.4f}")
    
    # Classification report on best model
    print(f"\n  Classification Report (final epoch):")
    print(classification_report(
        val_results['labels'], val_results['preds'],
        target_names=['Real', 'Fake']
    ))
    
    # Save results log
    log_path = os.path.join(args.save_dir, "clip_baseline_results.json")
    with open(log_path, 'w') as f:
        json.dump({
            'config': vars(args),
            'best_auc': best_auc,
            'training_log': results_log,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"  Results saved to {log_path}")
    
    print(f"\n🎉 You now have your first ablation table entry!")
    print(f"   Visual-only baseline AUC: {best_auc:.4f}")
    print(f"   Next step: run step4_wavlm_baseline.py for the audio-only row")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLIP visual-only deepfake baseline")
    parser.add_argument('--data', type=str, default='data/processed',
                        help='Path to preprocessed data (default: data/processed)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32, reduce if OOM)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    main(args)
