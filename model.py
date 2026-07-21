"""
model.py
MB-AV-Loc: plugs cached CLIP + WavLM features through
cross-modal bottleneck fusion into the PtTransformer backbone.

Registers as a new meta_arch "MBAVLocTransformer" so it can be
selected in a YAML config alongside the existing LocPointTransformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# MM-DDL imports (when placed in repo root or mb_av_loc/)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from libs.modeling.models import (
    register_meta_arch, make_backbone, make_neck, make_generator
)
from libs.modeling.meta_archs import PtTransformerClsHead, PtTransformerRegHead
from libs.modeling.losses import ctr_diou_loss_1d, sigmoid_focal_loss
from libs.utils import batched_nms

from fusion import MBAVFusion


@register_meta_arch("MBAVLocTransformer")
class MBAVLocTransformer(nn.Module):
    """
    MB-AV-Loc meta architecture.

    Input: video_list where each item has:
        'visual_feats': [T, 512]   (cached CLIP features)
        'audio_feats':  [4, T_a, 768]  (cached WavLM 4-layer features)
        'segments':     [N, 2]     (GT segments)
        'labels':       [N]        (GT labels)

    Pipeline:
        cached features -> learnable layer-weighted agg (audio)
                        -> MBAVFusion [B, T, 256]
                        -> PtTransformer backbone [B, 512, T//s] per level
                        -> FPN neck
                        -> cls + reg heads
    """

    def __init__(
        self,
        # fusion params
        visual_dim=512,
        audio_dim=768,
        fusion_dim=256,
        num_latents=8,
        num_fusion_layers=2,
        p_drop_visual=0.1,
        p_drop_audio=0.1,
        # audio layer-weighted aggregation
        num_audio_layers=4,
        # backbone params (same as PtTransformer)
        backbone_type='convTransformer',
        fpn_type='fpn',
        backbone_arch=(2, 1, 5),
        scale_factor=2,
        max_seq_len=1024,
        max_buffer_len_factor=16.0,
        n_head=4,
        n_mha_win_size=-1,
        embd_kernel_size=3,
        embd_dim=512,
        embd_with_ln=True,
        fpn_dim=512,
        fpn_with_ln=True,
        fpn_start_level=0,
        head_dim=512,
        regression_range=((0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)),
        head_num_layers=3,
        head_kernel_size=3,
        head_with_ln=True,
        use_abs_pe=False,
        use_rel_pe=False,
        num_classes=1,
        train_cfg=None,
        test_cfg=None,
        use_lstm=True,
        with_Difference=False,
    ):
        super().__init__()

        # ── Audio layer-weighted aggregation (Novelty 1) ──────────────
        self.num_audio_layers = num_audio_layers
        self.audio_layer_weights = nn.Parameter(torch.ones(num_audio_layers))

        # ── Cross-modal bottleneck fusion (main contribution) ─────────
        self.fusion = MBAVFusion(
            visual_dim=visual_dim,
            audio_dim=audio_dim,
            d_model=fusion_dim,
            num_latents=num_latents,
            num_layers=num_fusion_layers,
            p_drop_visual=p_drop_visual,
            p_drop_audio=p_drop_audio,
        )

        # ── FPN strides and regression ranges ─────────────────────────
        self.fpn_strides = [scale_factor ** i for i in range(
            fpn_start_level, backbone_arch[-1] + 1
        )]
        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len

        # window sizes
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size] * (1 + backbone_arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + backbone_arch[-1])
            self.mha_win_size = n_mha_win_size

        # ── Training config ───────────────────────────────────────────
        if train_cfg is None:
            train_cfg = {
                'init_loss_norm': 100, 'cls_prior_prob': 0.01,
                'center_sample': 'radius', 'center_sample_radius': 1.5,
                'clip_grad_l2norm': 1.0, 'loss_weight': 1.0,
                'head_empty_cls': [], 'dropout': 0.0, 'droppath': 0.1,
                'label_smoothing': 0.0,
            }
        self.train_center_sample = train_cfg['center_sample']
        self.train_center_sample_radius = train_cfg['center_sample_radius']
        self.train_loss_weight = train_cfg['loss_weight']
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']
        self.train_label_smoothing = train_cfg['label_smoothing']

        # ── Test config ───────────────────────────────────────────────
        if test_cfg is None:
            test_cfg = {
                'voting_thresh': 0.7, 'pre_nms_topk': 2000,
                'max_seg_num': 10, 'min_score': 0.9,
                'iou_threshold': 0.01, 'multiclass_nms': True,
                'nms_method': 'hard', 'pre_nms_thresh': 0.001,
                'nms_sigma': 0.5, 'duration_thresh': 0.05,
                'ext_score_file': None,
            }
        self.test_pre_nms_thresh = test_cfg['pre_nms_thresh']
        self.test_pre_nms_topk = test_cfg['pre_nms_topk']
        self.test_iou_threshold = test_cfg['iou_threshold']
        self.test_min_score = test_cfg['min_score']
        self.test_max_seg_num = test_cfg['max_seg_num']
        self.test_nms_method = test_cfg['nms_method']
        self.test_duration_thresh = test_cfg['duration_thresh']
        self.test_multiclass_nms = test_cfg['multiclass_nms']
        self.test_nms_sigma = test_cfg['nms_sigma']
        self.test_voting_thresh = test_cfg['voting_thresh']

        # ── Backbone (convTransformer) ────────────────────────────────
        # input_dim = fusion_dim (256), backbone projects to embd_dim (512)
        self.backbone = make_backbone(
            'convTransformer',
            **{
                'n_in': fusion_dim,
                'n_embd': embd_dim,
                'n_head': n_head,
                'n_embd_ks': embd_kernel_size,
                'max_len': max_seq_len,
                'arch': backbone_arch,
                'mha_win_size': self.mha_win_size,
                'scale_factor': scale_factor,
                'with_ln': embd_with_ln,
                'attn_pdrop': 0.0,
                'proj_pdrop': self.train_dropout,
                'path_pdrop': self.train_droppath,
                'use_abs_pe': use_abs_pe,
                'use_rel_pe': use_rel_pe,
                'use_lstm': use_lstm,
                'with_Difference': with_Difference,
            }
        )

        if isinstance(embd_dim, (list, tuple)):
            embd_dim = sum(embd_dim)

        # ── FPN neck ─────────────────────────────────────────────────
        self.neck = make_neck(
            fpn_type,
            **{
                'in_channels': [embd_dim] * (backbone_arch[-1] + 1),
                'out_channel': fpn_dim,
                'scale_factor': scale_factor,
                'start_level': fpn_start_level,
                'with_ln': fpn_with_ln,
            }
        )

        # ── Point generator ──────────────────────────────────────────
        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len': max_seq_len * max_buffer_len_factor,
                'fpn_strides': self.fpn_strides,
                'regression_range': self.reg_range,
            }
        )

        # ── Classification and regression heads ──────────────────────
        self.cls_head = PtTransformerClsHead(
            fpn_dim, head_dim, self.num_classes,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=head_with_ln,
            num_layers=head_num_layers,
            empty_cls=train_cfg['head_empty_cls'],
        )
        self.reg_head = PtTransformerRegHead(
            fpn_dim, head_dim, len(self.fpn_strides),
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=head_with_ln,
        )

        # loss normalizer
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9

    @property
    def device(self):
        return list(set(p.device for p in self.parameters()))[0]

    # ── Audio layer-weighted aggregation ──────────────────────────────

    def aggregate_audio_layers(self, audio_feats):
        """
        audio_feats: [4, T_a, 768] (cached 4-layer WavLM features)
        returns:     [T_a, 768]
        """
        weights = torch.softmax(self.audio_layer_weights, dim=0)  # [4]
        # [4, T_a, 768] * [4, 1, 1] -> sum -> [T_a, 768]
        aggregated = (audio_feats * weights[:, None, None]).sum(dim=0)
        return aggregated

    # ── Preprocessing ─────────────────────────────────────────────────

    def preprocessing(self, video_list):
        """
        Pad visual and audio features, run fusion, return [B, d, T] + masks.
        """
        vis_list = []
        aud_list = []

        for item in video_list:
            vis = item['visual_feats']   # [T, 512]
            aud = item['audio_feats']    # [4, T_a, 768]

            # aggregate audio layers
            aud = self.aggregate_audio_layers(aud)  # [T_a, 768]

            vis_list.append(vis)
            aud_list.append(aud)

        # pad visual to same T
        max_T = max(v.shape[0] for v in vis_list)
        # round up to max_div_factor
        max_div_factor = 1
        for s, w in zip(self.fpn_strides, self.mha_win_size):
            stride = s * (w // 2) * 2 if w > 1 else s
            if max_div_factor < stride:
                max_div_factor = stride
        max_T = (max_T + max_div_factor - 1) // max_div_factor * max_div_factor

        B = len(vis_list)
        padded_vis = torch.zeros(B, max_T, 512, device=self.device)
        padded_aud = torch.zeros(B, max_T, 768, device=self.device)
        masks = torch.zeros(B, 1, max_T, device=self.device)

        for i, (v, a) in enumerate(zip(vis_list, aud_list)):
            T_v = v.shape[0]
            padded_vis[i, :T_v] = v.to(self.device)

            # resample audio to T_v before padding
            a = a.to(self.device)
            a = a.unsqueeze(0).permute(0, 2, 1)  # [1, 768, T_a]
            a = F.interpolate(a, size=T_v, mode='linear', align_corners=False)
            a = a.permute(0, 2, 1).squeeze(0)    # [T_v, 768]
            padded_aud[i, :T_v] = a
            masks[i, 0, :T_v] = 1.0

        # run fusion: [B, T, 512] + [B, T, 768] -> [B, T, fusion_dim]
        fused = self.fusion(padded_vis, padded_aud)  # [B, T, 256]

        # transpose to [B, C, T] for backbone
        fused = fused.permute(0, 2, 1)  # [B, 256, T]

        return fused, masks

    # ── Forward ───────────────────────────────────────────────────────

    def forward(self, video_list):
        batched_inputs, batched_masks = self.preprocessing(video_list)

        # backbone -> neck -> heads
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)

        # points for decoding
        points = self.point_generator(fpn_feats)

        # classification and regression
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        out_offsets = self.reg_head(fpn_feats, fpn_masks)

        # permute outputs: [B, C, T] -> [B, T, C]
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]

        if self.training:
            # assemble GT
            gt_segments = [item['segments'].to(self.device) for item in video_list]
            gt_labels = [item['labels'].to(self.device) for item in video_list]

            gt_cls_labels, gt_offsets = self.label_points(
                points, gt_segments, gt_labels
            )

            losses = self.losses(
                fpn_masks,
                out_cls_logits, out_offsets,
                gt_cls_labels, gt_offsets,
            )
            return losses
        else:
            results = self.inference(
                video_list, points, fpn_masks,
                out_cls_logits, out_offsets,
            )
            return results

    # ── Label assignment (same as PtTransformer) ──────────────────────

    def label_points(self, points, gt_segments, gt_labels):
        concat_points = torch.cat(points, dim=0)  # [sum(T_l), 4]
        cls_targets_list, reg_targets_list = [], []

        for gt_seg, gt_lbl in zip(gt_segments, gt_labels):
            cls_targets, reg_targets = self.label_points_single_video(
                concat_points, gt_seg, gt_lbl
            )
            cls_targets_list.append(cls_targets)
            reg_targets_list.append(reg_targets)

        return cls_targets_list, reg_targets_list

    def label_points_single_video(self, concat_points, gt_segments, gt_labels):
        num_pts = concat_points.shape[0]
        num_gts = gt_segments.shape[0]

        if num_gts == 0:
            cls_targets = gt_segments.new_full((num_pts,), self.num_classes, dtype=torch.long)
            reg_targets = gt_segments.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        lens = gt_segments[:, 1] - gt_segments[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        gt_segs = gt_segments[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets_per_gt = torch.stack((left, right), dim=-1)

        if self.train_center_sample == 'radius':
            center = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            t_mins = center - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = center + concat_points[:, 3, None] * self.train_center_sample_radius
            inside_gt = (concat_points[:, 0, None] >= t_mins) & (
                concat_points[:, 0, None] <= t_maxs
            )
        else:
            inside_gt = reg_targets_per_gt.min(-1)[0] >= 0

        inside_gt_mask = inside_gt & (
            reg_targets_per_gt.min(-1)[0] >= 0
        )

        reg_lower = concat_points[:, 1, None]
        reg_upper = concat_points[:, 2, None]
        inside_range = (lens >= reg_lower) & (lens <= reg_upper)

        valid_mask = inside_gt_mask & inside_range

        lens[~valid_mask] = float('inf')
        min_len, min_len_inds = lens.min(dim=1)

        cls_targets = gt_labels[min_len_inds].clone()
        cls_targets[min_len == float('inf')] = self.num_classes

        reg_targets = reg_targets_per_gt[range(num_pts), min_len_inds]
        reg_targets[min_len == float('inf')] = 0

        return cls_targets, reg_targets

    # ── Losses ────────────────────────────────────────────────────────

    def losses(self, fpn_masks, out_cls_logits, out_offsets,
               gt_cls_labels, gt_offsets):

        valid_mask = torch.cat(fpn_masks, dim=1)  # [B, sum(T_l)]

        gt_cls = torch.stack(gt_cls_labels)       # [B, sum(T_l)]
        gt_off = torch.stack(gt_offsets)           # [B, sum(T_l), 2]

        pred_cls = torch.cat(out_cls_logits, dim=1)  # [B, sum(T_l), C]
        pred_off = torch.cat(out_offsets, dim=1)     # [B, sum(T_l), 2]

        pos_mask = (gt_cls >= 0) & (gt_cls < self.num_classes)
        pos_mask = pos_mask & (valid_mask > 0)

        num_pos = pos_mask.sum().clamp(min=1).float()

        # update loss normalizer
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos.item(), 1)

        # classification: focal loss
        gt_target = torch.zeros_like(pred_cls)
        gt_target[pos_mask] = 1.0

        cls_loss = sigmoid_focal_loss(pred_cls, gt_target, reduction='sum')
        cls_loss = cls_loss / self.loss_normalizer

        # regression: DIoU loss
        if pos_mask.any():
            pred_off_pos = pred_off[pos_mask]  # [num_pos, 2]
            gt_off_pos = gt_off[pos_mask]      # [num_pos, 2]

            pred_off_pos = torch.relu(pred_off_pos)

            reg_loss = ctr_diou_loss_1d(pred_off_pos, gt_off_pos, reduction='sum')
            reg_loss = reg_loss / self.loss_normalizer
        else:
            reg_loss = pred_off.sum() * 0.0

        final_loss = cls_loss + self.train_loss_weight * reg_loss

        return {
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'final_loss': final_loss,
        }

    # ── Inference ─────────────────────────────────────────────────────

    def inference(self, video_list, points, fpn_masks,
                  out_cls_logits, out_offsets):
        results = []
        B = len(video_list)

        concat_points = torch.cat(points, dim=0)
        fpn_masks_cat = torch.cat(fpn_masks, dim=1)

        for b in range(B):
            pred_cls = torch.cat(out_cls_logits, dim=1)[b]  # [sum(T_l), C]
            pred_off = torch.cat(out_offsets, dim=1)[b]     # [sum(T_l), 2]
            mask = fpn_masks_cat[b].bool()                  # [sum(T_l)]

            pred_cls = pred_cls[mask]
            pred_off = pred_off[mask]
            pts = concat_points[mask]

            pred_prob = pred_cls.sigmoid()

            # decode segments
            pred_off = torch.relu(pred_off)
            seg_left = pts[:, 0] - pred_off[:, 0]
            seg_right = pts[:, 0] + pred_off[:, 1]
            pred_segs = torch.stack((seg_left, seg_right), dim=-1)

            # filter by score
            max_scores, max_labels = pred_prob.max(dim=-1)
            keep = max_scores > self.test_pre_nms_thresh
            pred_segs = pred_segs[keep]
            max_scores = max_scores[keep]
            max_labels = max_labels[keep]

            # topk
            if max_scores.shape[0] > self.test_pre_nms_topk:
                _, topk_idx = max_scores.topk(self.test_pre_nms_topk)
                pred_segs = pred_segs[topk_idx]
                max_scores = max_scores[topk_idx]
                max_labels = max_labels[topk_idx]

            # NMS
            if self.test_nms_method != 'none' and pred_segs.shape[0] > 0:
                keep = batched_nms(
                    pred_segs, max_scores, max_labels,
                    self.test_iou_threshold,
                    self.test_min_score,
                    self.test_max_seg_num,
                    use_soft_nms=(self.test_nms_method == 'soft'),
                    multiclass=self.test_multiclass_nms,
                    sigma=self.test_nms_sigma,
                    voting_thresh=self.test_voting_thresh,
                )
                pred_segs = pred_segs[keep]
                max_scores = max_scores[keep]
                max_labels = max_labels[keep]

            # filter by duration
            seg_lens = pred_segs[:, 1] - pred_segs[:, 0]
            keep = seg_lens > self.test_duration_thresh
            pred_segs = pred_segs[keep]
            max_scores = max_scores[keep]
            max_labels = max_labels[keep]

            results.append({
                'video_id': video_list[b].get('video_id', f'video_{b}'),
                'segments': pred_segs.cpu(),
                'scores': max_scores.cpu(),
                'labels': max_labels.cpu(),
            })

        return results
