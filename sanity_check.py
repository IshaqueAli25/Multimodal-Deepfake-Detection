"""
sanity_check.py
Load one real LAV-DF video, run it end-to-end through MB-AV-Loc,
print every intermediate shape. Kills the pipeline if anything breaks.
"""
import os, sys, json
import torch
import numpy as np
import cv2
import torchaudio
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- pick one fake video from LAV-DF (fake has GT segments to test loss path) ----
LAVDF_ROOT = "/scratch/zm21856/LAV-DF/LAV-DF"
META = os.path.join(LAVDF_ROOT, "metadata.min.json")

with open(META) as f:
    all_meta = json.load(f)

entry = next(e for e in all_meta
             if (e['modify_video'] or e['modify_audio'])
             and e['split'] == 'train'
             and len(e['fake_periods']) > 0)

video_path = os.path.join(LAVDF_ROOT, entry['file'])
print(f"Test video: {entry['file']}")
print(f"  duration: {entry['duration']:.2f}s, video_frames: {entry['video_frames']}")
print(f"  fake_periods: {entry['fake_periods']}")

DEV = "cuda:0"

# ---- CLIP visual features ----
print("\n[1/5] Loading CLIP and extracting visual features...")
from transformers import CLIPModel, CLIPProcessor
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(DEV).eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

cap = cv2.VideoCapture(video_path)
src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
interval = max(1, round(src_fps / 25))
frames, idx = [], 0
while True:
    ret, f = cap.read()
    if not ret: break
    if idx % interval == 0:
        frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    idx += 1
cap.release()
print(f"  extracted {len(frames)} frames at 25fps")

with torch.no_grad():
    inputs = processor(images=frames, return_tensors="pt").to(DEV)
    vis_feats = clip.get_image_features(**inputs)  # [T, 512]
print(f"  visual_feats shape: {tuple(vis_feats.shape)}   (expect [T, 512])")

# ---- WavLM audio features ----
print("\n[2/5] Loading WavLM and extracting 4-layer audio features...")
from transformers import WavLMModel
wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base",
                                    output_hidden_states=True).to(DEV).eval()

wav, sr = torchaudio.load(video_path)
if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
if sr != 16000:
    wav = torchaudio.transforms.Resample(sr, 16000)(wav)
wav = wav.to(DEV)

with torch.no_grad():
    out = wavlm(input_values=wav, output_hidden_states=True)
LAYER_IDS = [0, 4, 8, 12]
aud_feats = torch.stack([out.hidden_states[i] for i in LAYER_IDS], 0).squeeze(1)  # [4, T_a, 768]
print(f"  audio_feats shape: {tuple(aud_feats.shape)}   (expect [4, T_a, 768])")

# ---- Build MB-AV-Loc model ----
print("\n[3/5] Building MB-AV-Loc model...")
import model as _model_module  # registers meta_arch
from libs.modeling.models import make_meta_arch

mbav = make_meta_arch(
    "MBAVLocTransformer",
    visual_dim=512, audio_dim=768, fusion_dim=256,
    num_latents=8, num_fusion_layers=2,
    p_drop_visual=0.1, p_drop_audio=0.1,
    num_audio_layers=4,
    backbone_type='convTransformer', fpn_type='fpn',
    backbone_arch=(2, 1, 5), scale_factor=2,
    max_seq_len=1024, max_buffer_len_factor=16.0,
    n_head=4, n_mha_win_size=-1,
    embd_kernel_size=3, embd_dim=512, embd_with_ln=True,
    fpn_dim=512, fpn_with_ln=True, fpn_start_level=0,
    head_dim=512,
    regression_range=((0,4),(4,8),(8,16),(16,32),(32,64),(64,10000)),
    head_num_layers=3, head_kernel_size=3, head_with_ln=True,
    use_abs_pe=False, use_rel_pe=False,
    num_classes=1,
    use_lstm=True, with_Difference=False,
).to(DEV)
n_params = sum(p.numel() for p in mbav.parameters())
n_train = sum(p.numel() for p in mbav.parameters() if p.requires_grad)
print(f"  total params: {n_params/1e6:.2f}M | trainable: {n_train/1e6:.2f}M")
print(f"  fpn_strides: {mbav.fpn_strides}   (expect 6 levels)")

# ---- Build one video_list item ----
print("\n[4/5] Assembling video_list and running forward (train mode)...")
segs_sec = torch.tensor(entry['fake_periods'], dtype=torch.float32)
segments = segs_sec * 25  # sec -> frame indices at 25fps
labels = torch.zeros(len(segs_sec), dtype=torch.long)

video_list = [{
    'video_id': entry['file'],
    'visual_feats': vis_feats.float().to(DEV),
    'audio_feats': aud_feats.float().to(DEV),
    'segments': segments,
    'labels': labels,
    'duration': float(vis_feats.shape[0]),
    'fps': 25.0,
}]
print(f"  segments (frames): {segments.tolist()}")

mbav.train()
losses = mbav(video_list)
print(f"  losses: cls={losses['cls_loss'].item():.4f}  "
      f"reg={losses['reg_loss'].item():.4f}  "
      f"final={losses['final_loss'].item():.4f}")

# ---- Backward pass ----
print("\n[5/5] Backward pass...")
losses['final_loss'].backward()
print("  backward OK")

print("\n✓ SANITY CHECK PASSED — model is wired correctly end-to-end.")
