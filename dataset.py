"""
dataset.py
Dataset for MB-AV-Loc: loads cached CLIP visual + WavLM audio features
with LAV-DF temporal forgery annotations.

Each sample returns a dict:
    'video_id':      str
    'visual_feats':  [T, 512]    float32
    'audio_feats':   [4, T_a, 768]  float32
    'segments':      [N, 2]      float32 (start/end in feature-frame indices)
    'labels':        [N]         int64   (0 = fake)
    'duration':      float       (num visual frames)
    'fps':           float       (visual feature fps, default 25)
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class LAVDFCachedDataset(Dataset):
    """
    Loads pre-extracted features from .npy files and pairs them
    with LAV-DF temporal annotations.

    Directory layout expected:
        cache_dir/
            visual/  {video_id}.npy   -> [T, 512] fp16
            audio/   {video_id}.npy   -> [4, T_a, 768] fp16

        anno_file: LAV-DF metadata JSON with fields per entry:
            - file:         filename (e.g. "000001.mp4")
            - modify_type:  "real" or fake type string
            - fake_periods: [[start_sec, end_sec], ...]  (empty for real)
    """

    def __init__(
        self,
        cache_dir,
        anno_file,
        split='training',
        feat_fps=25,
        max_seq_len=1024,
        trunc_thresh=0.5,
        crop_ratio=(0.9, 1.0),
        is_training=True,
    ):
        super().__init__()

        self.cache_dir = cache_dir
        self.vis_dir = os.path.join(cache_dir, 'visual')
        self.aud_dir = os.path.join(cache_dir, 'audio')
        self.feat_fps = feat_fps
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        self.is_training = is_training

        # load annotations
        with open(anno_file, 'r') as f:
            all_annos = json.load(f)

        # filter by split and available cached features
        self.data_list = []
        for entry in all_annos:
            # LAV-DF metadata format
            vid = entry.get('file', entry.get('video_id', ''))
            if isinstance(vid, str) and vid.endswith('.mp4'):
                vid = vid[:-4]

            # check split
            entry_split = entry.get('split', split)
            if entry_split != split:
                continue

            # check cached features exist
            vis_path = os.path.join(self.vis_dir, f'{vid}.npy')
            aud_path = os.path.join(self.aud_dir, f'{vid}.npy')
            if not os.path.exists(vis_path) or not os.path.exists(aud_path):
                continue

            # parse fake periods -> segments in seconds
            fake_periods = entry.get('fake_periods', [])
            is_fake = entry.get('modify_video', False) or entry.get('modify_audio', False)

            segments = []
            labels = []
            if is_fake and len(fake_periods) > 0:
                for period in fake_periods:
                    if len(period) >= 2:
                        segments.append([float(period[0]), float(period[1])])
                        labels.append(0)  # 0 = fake class

            self.data_list.append({
                'video_id': vid,
                'vis_path': vis_path,
                'aud_path': aud_path,
                'segments': segments,  # in seconds
                'labels': labels,
            })

        print(f"[LAVDFCachedDataset] split={split}, "
              f"loaded {len(self.data_list)} videos with cached features")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        # load cached features (fp16 -> fp32)
        vis_feats = np.load(item['vis_path']).astype(np.float32)  # [T, 512]
        aud_feats = np.load(item['aud_path']).astype(np.float32)  # [4, T_a, 768]

        vis_feats = torch.from_numpy(vis_feats)
        aud_feats = torch.from_numpy(aud_feats)

        T = vis_feats.shape[0]  # number of visual frames

        # convert segments from seconds to frame indices
        segments = torch.tensor(item['segments'], dtype=torch.float32)
        labels = torch.tensor(item['labels'], dtype=torch.long)

        if len(segments) > 0:
            segments = segments * self.feat_fps  # seconds -> frame indices
        else:
            segments = torch.zeros((0, 2), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        # truncate if longer than max_seq_len
        if self.is_training and T > self.max_seq_len:
            vis_feats, aud_feats, segments, labels = self._truncate(
                vis_feats, aud_feats, segments, labels, T
            )
            T = vis_feats.shape[0]

        # random crop augmentation during training
        if self.is_training and len(self.crop_ratio) == 2:
            vis_feats, aud_feats, segments, labels = self._random_crop(
                vis_feats, aud_feats, segments, labels, T
            )
            T = vis_feats.shape[0]

        data_dict = {
            'video_id': item['video_id'],
            'visual_feats': vis_feats,     # [T, 512]
            'audio_feats': aud_feats,      # [4, T_a, 768]
            'segments': segments,           # [N, 2] in frame indices
            'labels': labels,               # [N]
            'duration': float(T),
            'fps': float(self.feat_fps),
        }

        return data_dict

    def _truncate(self, vis, aud, segs, labels, T):
        """Truncate to max_seq_len with random offset, keeping valid segments."""
        max_len = self.max_seq_len

        # random start
        start = np.random.randint(0, T - max_len + 1)
        end = start + max_len

        vis = vis[start:end]

        # adjust segments
        if len(segs) > 0:
            segs = segs - start
            # keep segments that overlap with the window
            valid = (segs[:, 1] > 0) & (segs[:, 0] < max_len)
            segs = segs[valid].clamp(min=0, max=max_len)
            labels = labels[valid]

            # remove segments shorter than threshold
            if len(segs) > 0:
                seg_lens = segs[:, 1] - segs[:, 0]
                keep = seg_lens > (self.trunc_thresh * (segs[:, 1] - segs[:, 0]).clamp(min=1))
                # simpler: keep segments with reasonable length
                keep = seg_lens > 0.5
                segs = segs[keep]
                labels = labels[keep]

        return vis, aud, segs, labels

    def _random_crop(self, vis, aud, segs, labels, T):
        """Randomly crop the temporal extent."""
        if T <= 1:
            return vis, aud, segs, labels

        ratio = np.random.uniform(self.crop_ratio[0], self.crop_ratio[1])
        new_T = max(1, int(T * ratio))

        if new_T >= T:
            return vis, aud, segs, labels

        start = np.random.randint(0, T - new_T + 1)
        end = start + new_T

        vis = vis[start:end]

        if len(segs) > 0:
            segs = segs - start
            valid = (segs[:, 1] > 0) & (segs[:, 0] < new_T)
            segs = segs[valid].clamp(min=0, max=new_T)
            labels = labels[valid]

            if len(segs) > 0:
                seg_lens = segs[:, 1] - segs[:, 0]
                keep = seg_lens > 0.5
                segs = segs[keep]
                labels = labels[keep]

        return vis, aud, segs, labels


def collate_fn(batch):
    """Custom collate: return list of dicts (variable-length features)."""
    return batch
