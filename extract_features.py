"""
extract_features.py
Cache CLIP ViT-B/16 visual features and WavLM-Base 4-layer audio features
from LAV-DF videos as fp16 numpy arrays.

Output per video:
  visual/  {video_id}.npy  -> [T, 512]  fp16
  audio/   {video_id}.npy  -> [4, T_a, 768]  fp16

Usage (on cluster):
  python extract_features.py \
      --data_dir /scratch/Deepfake/datasets/LAV-DF \
      --output_dir /scratch/Deepfake/new/MM-DDL/.cached_features \
      --device cuda:0 \
      --batch_size 16 \
      --subset 25000
"""

import os
import argparse
import json
import glob
import numpy as np
import torch
import cv2
import torchaudio
from tqdm import tqdm


# ── CLIP visual encoder (frozen) ──────────────────────────────────────────

def build_clip_encoder(device):
    import clip
    model, preprocess = clip.load("ViT-B/16", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.visual.float(), preprocess


def extract_visual_features(video_path, clip_visual, preprocess, device,
                            target_fps=25, max_frames=1024):
    """
    Read video frames, run through CLIP visual encoder.
    Returns: numpy array [T, 512] fp16
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open {video_path}")
        return None

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, round(src_fps / target_fps))

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            # BGR -> RGB -> PIL -> CLIP preprocess
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image
            pil_img = Image.fromarray(frame_rgb)
            tensor = preprocess(pil_img)
            frames.append(tensor)
            if len(frames) >= max_frames:
                break
        idx += 1
    cap.release()

    if len(frames) == 0:
        return None

    # batch through CLIP
    frames_tensor = torch.stack(frames).to(device)  # [T, 3, 224, 224]
    all_feats = []

    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(frames_tensor), batch_size):
            batch = frames_tensor[i:i + batch_size]
            feats = clip_visual(batch)  # [bs, 512]
            all_feats.append(feats.cpu())

    all_feats = torch.cat(all_feats, dim=0)  # [T, 512]
    return all_feats.half().numpy()


# ── WavLM audio encoder (frozen, 4-layer cache) ──────────────────────────

LAYER_IDS = [0, 4, 8, 12]  # embedding, low, mid, high


def build_wavlm_encoder(device):
    from transformers import WavLMModel
    model = WavLMModel.from_pretrained(
        "microsoft/wavlm-base",
        output_hidden_states=True,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def extract_audio_features(video_path, wavlm, device, target_sr=16000):
    """
    Extract audio from video, run through WavLM, cache 4 layer outputs.
    Returns: numpy array [4, T_a, 768] fp16
    """
    try:
        waveform, sr = torchaudio.load(video_path)
    except Exception as e:
        print(f"[WARN] Cannot load audio from {video_path}: {e}")
        return None

    # mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    waveform = waveform.to(device)  # [1, samples]

    with torch.no_grad():
        out = wavlm(input_values=waveform, output_hidden_states=True)

    # out.hidden_states: tuple of 13 tensors, each [1, T_a, 768]
    selected = torch.stack([out.hidden_states[i] for i in LAYER_IDS], dim=0)
    # [4, 1, T_a, 768] -> [4, T_a, 768]
    selected = selected.squeeze(1).cpu().half().numpy()

    return selected


# ── Main ──────────────────────────────────────────────────────────────────

def find_videos(data_dir):
    """Find all mp4 files in LAV-DF directory structure."""
    patterns = [
        os.path.join(data_dir, "**", "*.mp4"),
        os.path.join(data_dir, "*.mp4"),
    ]
    videos = []
    for pattern in patterns:
        videos.extend(glob.glob(pattern, recursive=True))
    # deduplicate
    videos = sorted(set(videos))
    return videos


def video_id_from_path(path):
    """Extract video ID from path (filename without extension)."""
    return os.path.splitext(os.path.basename(path))[0]


def main():
    parser = argparse.ArgumentParser(description="Cache CLIP + WavLM features for LAV-DF")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to LAV-DF dataset root")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save cached features")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--subset", type=int, default=0,
                        help="Only process first N videos (0 = all)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip videos that already have cached features")
    args = parser.parse_args()

    vis_dir = os.path.join(args.output_dir, "visual")
    aud_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(aud_dir, exist_ok=True)

    # find videos
    videos = find_videos(args.data_dir)
    print(f"Found {len(videos)} videos in {args.data_dir}")

    if args.subset > 0:
        videos = videos[:args.subset]
        print(f"Processing subset of {len(videos)} videos")

    # build encoders
    print("Loading CLIP ViT-B/16...")
    clip_visual, clip_preprocess = build_clip_encoder(args.device)
    print("Loading WavLM-Base...")
    wavlm = build_wavlm_encoder(args.device)

    # extract
    success, fail = 0, 0
    for video_path in tqdm(videos, desc="Extracting features"):
        vid = video_id_from_path(video_path)
        vis_path = os.path.join(vis_dir, f"{vid}.npy")
        aud_path = os.path.join(aud_dir, f"{vid}.npy")

        if args.skip_existing and os.path.exists(vis_path) and os.path.exists(aud_path):
            success += 1
            continue

        # visual
        vis_feats = extract_visual_features(
            video_path, clip_visual, clip_preprocess, args.device
        )

        # audio
        aud_feats = extract_audio_features(video_path, wavlm, args.device)

        if vis_feats is not None and aud_feats is not None:
            np.save(vis_path, vis_feats)
            np.save(aud_path, aud_feats)
            success += 1
        else:
            fail += 1

    print(f"\nDone. Success: {success}, Failed: {fail}")
    print(f"Visual features: {vis_dir}")
    print(f"Audio features:  {aud_dir}")


if __name__ == "__main__":
    main()
