"""
Step 1: Verify Setup
====================
Run this FIRST. It checks:
  1. Your GPU is detected and has enough VRAM
  2. CLIP ViT-B/16 loads correctly
  3. WavLM-Base loads correctly
  4. Both fit on your GPU simultaneously
  5. Forward passes work
  6. Counts trainable LN parameters (for your dissertation)

Usage: python step1_verify_setup.py
"""

import torch
import sys

print("=" * 60)
print("STEP 1: VERIFYING YOUR SETUP")
print("=" * 60)

# ---- Check GPU ----
if not torch.cuda.is_available():
    print("❌ CUDA not available. Check your PyTorch installation.")
    print("   Run: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"✅ GPU detected: {gpu_name}")
print(f"   Total VRAM: {gpu_vram:.1f} GB")

if gpu_vram < 14:
    print("⚠️  Less than 14GB VRAM. You may need to reduce batch sizes.")

# ---- Load CLIP ViT-B/16 ----
print("\n--- Loading CLIP ViT-B/16 ---")
try:
    from transformers import CLIPVisionModel, CLIPImageProcessor
    
    clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    clip_model = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-base-patch16",
        use_safetensors=True,
    )
    clip_model = clip_model.half().cuda().eval()
    
    clip_vram = torch.cuda.memory_allocated() / 1e9
    print(f"✅ CLIP loaded. VRAM used: {clip_vram:.2f} GB")
except Exception as e:
    print(f"❌ Failed to load CLIP: {e}")
    sys.exit(1)

# ---- Load WavLM-Base ----
print("\n--- Loading WavLM-Base ---")
try:
    from transformers import WavLMModel, Wav2Vec2FeatureExtractor
    
    wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
    wavlm_model = WavLMModel.from_pretrained(
        "microsoft/wavlm-base",
        use_safetensors=True,
    )
    wavlm_model = wavlm_model.half().cuda().eval()
    
    both_vram = torch.cuda.memory_allocated() / 1e9
    wavlm_vram = both_vram - clip_vram
    print(f"✅ WavLM loaded. VRAM used: {wavlm_vram:.2f} GB")
    print(f"   Combined VRAM (CLIP + WavLM): {both_vram:.2f} GB")
    print(f"   Remaining VRAM for training: {gpu_vram - both_vram:.1f} GB")
except Exception as e:
    print(f"❌ Failed to load WavLM: {e}")
    sys.exit(1)

# ---- Test CLIP forward pass ----
print("\n--- Testing CLIP forward pass ---")
try:
    dummy_img = torch.randn(4, 3, 224, 224).half().cuda()  # batch of 4 images
    with torch.no_grad():
        clip_out = clip_model(dummy_img)
    
    cls_token = clip_out.last_hidden_state[:, 0, :]  # CLS token
    print(f"✅ CLIP forward pass works")
    print(f"   Input shape:  {list(dummy_img.shape)}  (batch, channels, height, width)")
    print(f"   Full output:  {list(clip_out.last_hidden_state.shape)}  (batch, patches+1, hidden_dim)")
    print(f"   CLS token:    {list(cls_token.shape)}  (batch, hidden_dim)")
    print(f"   Hidden dim = {cls_token.shape[1]} → this is your visual feature size")
except Exception as e:
    print(f"❌ CLIP forward pass failed: {e}")

# ---- Test WavLM forward pass ----
print("\n--- Testing WavLM forward pass ---")
try:
    # 1 second of audio at 16kHz
    dummy_audio = torch.randn(4, 16000).half().cuda()  # batch of 4
    with torch.no_grad():
        wavlm_out = wavlm_model(dummy_audio)
    
    audio_features = wavlm_out.last_hidden_state
    print(f"✅ WavLM forward pass works")
    print(f"   Input shape:  {list(dummy_audio.shape)}  (batch, samples)")
    print(f"   Output shape: {list(audio_features.shape)}  (batch, time_steps, hidden_dim)")
    print(f"   Hidden dim = {audio_features.shape[2]} → this is your audio feature size")
    print(f"   Time steps = {audio_features.shape[1]} (for 1 second of audio)")
except Exception as e:
    print(f"❌ WavLM forward pass failed: {e}")

# ---- Count parameters ----
print("\n--- Parameter Count (for your dissertation) ---")

def count_params(model, model_name):
    total = sum(p.numel() for p in model.parameters())
    ln_params = sum(
        p.numel() for name, p in model.named_parameters()
        if 'layer_norm' in name.lower() or 'layernorm' in name.lower()
    )
    print(f"\n{model_name}:")
    print(f"   Total parameters:    {total:>12,}")
    print(f"   LayerNorm parameters:{ln_params:>12,}")
    print(f"   LN percentage:       {ln_params/total*100:>11.3f}%")
    print(f"   If you freeze all and unfreeze LN, you train {ln_params:,} params")
    return total, ln_params

clip_total, clip_ln = count_params(clip_model, "CLIP ViT-B/16")
wavlm_total, wavlm_ln = count_params(wavlm_model, "WavLM-Base")

print(f"\n  Combined total:     {clip_total + wavlm_total:,}")
print(f"  Combined trainable: {clip_ln + wavlm_ln:,}")
print(f"  Combined LN %:     {(clip_ln + wavlm_ln)/(clip_total + wavlm_total)*100:.3f}%")

# ---- Print LayerNorm layer names (so you know what you're unfreezing) ----
print("\n--- CLIP LayerNorm layer names ---")
for name, p in clip_model.named_parameters():
    if 'layer_norm' in name.lower() or 'layernorm' in name.lower():
        print(f"   {name}  shape={list(p.shape)}")

print("\n--- WavLM LayerNorm layer names ---")
for name, p in wavlm_model.named_parameters():
    if 'layer_norm' in name.lower() or 'layernorm' in name.lower():
        print(f"   {name}  shape={list(p.shape)}")

# ---- Final VRAM check ----
print("\n" + "=" * 60)
peak_vram = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak VRAM used during test: {peak_vram:.2f} GB")
print(f"VRAM remaining:            {gpu_vram - peak_vram:.1f} GB")
print(f"")
print(f"🎉 SETUP VERIFIED SUCCESSFULLY")
print(f"   You're ready to start building the baseline.")
print(f"   Next step: run step2_preprocess_data.py")
print("=" * 60)
