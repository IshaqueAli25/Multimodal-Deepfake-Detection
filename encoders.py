"""
mb_av_loc/encoders.py
Frozen foundation model encoders with LayerNorm-only PEFT.
  - CLIPVisualEncoder  : CLIP ViT-B/16, outputs [B, T, 512]
  - WavLMAudioEncoder  : WavLM-Base, learnable layer-weighted aggregation -> [B, T, 768]
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unfreeze_layernorm_only(module: nn.Module):
    """Freeze everything, then unfreeze all LayerNorm parameters."""
    for p in module.parameters():
        p.requires_grad = False
    for m in module.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad = True


# ---------------------------------------------------------------------------
# Visual encoder
# ---------------------------------------------------------------------------

class CLIPVisualEncoder(nn.Module):
    """
    Frozen CLIP ViT-B/16 visual encoder.
    Input : video_frames  [B, T, 3, 224, 224]
    Output: frame_embeds  [B, T, 512]
    """

    def __init__(self, clip_model_name: str = "ViT-B/16", device: str = "cuda"):
        super().__init__()
        try:
            import clip
        except ImportError:
            raise ImportError("pip install git+https://github.com/openai/CLIP.git")

        model, _ = clip.load(clip_model_name, device=device)
        self.visual = model.visual  # just the vision transformer
        self.visual = self.visual.float()

        _unfreeze_layernorm_only(self.visual)

        n_total = sum(p.numel() for p in self.visual.parameters())
        n_train = sum(p.numel() for p in self.visual.parameters() if p.requires_grad)
        print(f"[CLIPVisualEncoder] trainable: {n_train:,} / {n_total:,} "
              f"({100*n_train/n_total:.3f}%)")

    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        video_frames : [B, T, 3, 224, 224]
        returns      : [B, T, 512]
        """
        B, T, C, H, W = video_frames.shape
        # flatten time into batch
        x = video_frames.view(B * T, C, H, W)
        feats = self.visual(x)          # [B*T, 512]
        feats = feats.view(B, T, -1)    # [B, T, 512]
        return feats


# ---------------------------------------------------------------------------
# Audio encoder
# ---------------------------------------------------------------------------

class WavLMAudioEncoder(nn.Module):
    """
    Frozen WavLM-Base encoder with learnable layer-weighted aggregation (Novelty 1).

    WavLM has 13 hidden states (embedding layer + 12 transformer layers).
    We learn a scalar weight per layer and compute a weighted sum — this is
    the audio-side analogue of Pre-trained Information Bias (QTFP, CVPR 2026).

    Input : waveform      [B, raw_samples]   (16 kHz)
    Output: audio_embeds  [B, T_a, 768]      (T_a = waveform_len / 320)
    """

    def __init__(self, wavlm_model_name: str = "microsoft/wavlm-base"):
        super().__init__()
        try:
            from transformers import WavLMModel
        except ImportError:
            raise ImportError("pip install transformers")

        self.wavlm = WavLMModel.from_pretrained(
            wavlm_model_name,
            output_hidden_states=True,
        )

        _unfreeze_layernorm_only(self.wavlm)

        # Learnable per-layer weights — 13 layers (embedding + 12 transformer)
        self.num_layers = 13
        self.layer_weights = nn.Parameter(torch.ones(self.num_layers))

        n_total = sum(p.numel() for p in self.wavlm.parameters())
        n_train = sum(p.numel() for p in self.wavlm.parameters() if p.requires_grad)
        n_train += self.layer_weights.numel()
        print(f"[WavLMAudioEncoder] trainable: {n_train:,} / {n_total:,} "
              f"({100*n_train/n_total:.3f}%)")

    def forward(self, waveform: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        waveform       : [B, raw_samples]
        attention_mask : [B, raw_samples]  (optional, 1=valid 0=pad)
        returns        : [B, T_a, 768]
        """
        out = self.wavlm(
            input_values=waveform,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # hidden_states: tuple of (num_layers,) each [B, T_a, 768]
        hidden_states = torch.stack(out.hidden_states, dim=1)  # [B, 13, T_a, 768]

        # softmax-normalised weighted sum
        weights = torch.softmax(self.layer_weights, dim=0)     # [13]
        # einsum: b l t d, l -> b t d
        audio_embeds = torch.einsum('bltd,l->btd', hidden_states, weights)  # [B, T_a, 768]

        return audio_embeds
