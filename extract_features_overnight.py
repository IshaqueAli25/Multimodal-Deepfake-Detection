"""
extract_features_overnight.py
Cache CLIP + WavLM features for a LAV-DF subset. Resumable.

Uses transformers (matching the sanity-check environment).
Safe to re-run: skips videos whose features are already cached.

Usage:
  python3 extract_features_overnight.py --subset 400 --include_dev 100     # quick test
  python3 extract_features_overnight.py --subset 0 --include_dev 0         # everything
"""
import os, json, argparse, time
import numpy as np
import torch
import cv2
import torchaudio
from PIL import Image
from tqdm import tqdm


def build_clip(device):
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device).eval()
    p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    for pr in m.parameters():
        pr.requires_grad = False
    return m, p


def build_wavlm(device):
    from transformers import WavLMModel
    m = WavLMModel.from_pretrained(
        "microsoft/wavlm-base", output_hidden_states=True
    ).to(device).eval()
    for pr in m.parameters():
        pr.requires_grad = False
    return m


@torch.no_grad()
def extract_visual(video_path, clip_model, processor, device,
                   target_fps=25, max_frames=1024):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, round(src_fps / target_fps))
    frames, idx = [], 0
    while True:
        ret, f = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            if len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    if not frames:
        return None
    all_feats = []
    for i in range(0, len(frames), 64):
        batch = frames[i:i + 64]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        feats = clip_model.get_image_features(**inputs)  # [b, 512]
        all_feats.append(feats.cpu())
    return torch.cat(all_feats, 0).half().numpy()  # [T, 512] fp16


@torch.no_grad()
def extract_audio(video_path, wavlm, device, target_sr=16000):
    LAYER_IDS = [0, 4, 8, 12]
    try:
        wav, sr = torchaudio.load(video_path)
    except Exception:
        return None
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    wav = wav.to(device)
    out = wavlm(input_values=wav, output_hidden_states=True)
    sel = torch.stack(
        [out.hidden_states[i] for i in LAYER_IDS], 0
    ).squeeze(1)  # [4, T_a, 768]
    return sel.cpu().half().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/scratch/zm21856/LAV-DF/LAV-DF")
    ap.add_argument("--meta",
                    default="/scratch/zm21856/LAV-DF/LAV-DF/metadata.min.json")
    ap.add_argument("--out_dir", default="/scratch/zm21856/cached_features")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--subset", type=int, default=15000,
                    help="How many training videos to process (0 = all)")
    ap.add_argument("--include_dev", type=int, default=2000,
                    help="Extra dev-split videos to process (0 = all dev)")
    args = ap.parse_args()

    vis_dir = os.path.join(args.out_dir, "visual")
    aud_dir = os.path.join(args.out_dir, "audio")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(aud_dir, exist_ok=True)

    with open(args.meta) as f:
        all_meta = json.load(f)

    train_fake = [e for e in all_meta if e['split'] == 'train'
                  and (e['modify_video'] or e['modify_audio'])]
    train_real = [e for e in all_meta if e['split'] == 'train'
                  and not (e['modify_video'] or e['modify_audio'])]
    dev_all = [e for e in all_meta if e['split'] == 'dev']

    if args.subset > 0:
        n_fake = min(args.subset * 3 // 4, len(train_fake))
        n_real = min(args.subset - n_fake, len(train_real))
        train_sel = train_fake[:n_fake] + train_real[:n_real]
    else:
        train_sel = train_fake + train_real
    dev_sel = dev_all[:args.include_dev] if args.include_dev > 0 else dev_all

    to_process = train_sel + dev_sel
    n_f = sum(1 for e in train_sel if e['modify_video'] or e['modify_audio'])
    n_r = len(train_sel) - n_f
    print(f"Total to process: {len(to_process)}  "
          f"(train fake: {n_f}, train real: {n_r}, dev: {len(dev_sel)})")

    print("Loading CLIP...")
    clip_m, clip_p = build_clip(args.device)
    print("Loading WavLM...")
    wavlm = build_wavlm(args.device)

    ok, skip, fail = 0, 0, 0
    t0 = time.time()
    for entry in tqdm(to_process, desc="Extracting"):
        vid = entry['file'].replace('.mp4', '').replace('/', '_')
        vp = os.path.join(vis_dir, f"{vid}.npy")
        ap_ = os.path.join(aud_dir, f"{vid}.npy")
        if os.path.exists(vp) and os.path.exists(ap_):
            skip += 1
            continue
        video_path = os.path.join(args.data_root, entry['file'])
        try:
            v = extract_visual(video_path, clip_m, clip_p, args.device)
            a = extract_audio(video_path, wavlm, args.device)
            if v is not None and a is not None:
                np.save(vp, v)
                np.save(ap_, a)
                ok += 1
            else:
                fail += 1
        except Exception as e:
            fail += 1
            if fail < 10:
                print(f"[FAIL] {entry['file']}: {e}")
    dt = time.time() - t0
    print(f"\nDone in {dt/60:.1f} min. ok={ok} skip={skip} fail={fail}")


if __name__ == "__main__":
    main()
