"""
mb_av_loc/fusion.py
Temporal-preserving cross-modal bottleneck fusion for localisation.
Latents absorb both streams; streams read back from latents.
Output keeps the time axis: [B, T, d].
Includes modality dropout (Novelty 2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckFusionLayer(nn.Module):
    def __init__(self, d_model=256, num_latents=8, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn_lv = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_la = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_vl = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_al = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln_l1 = nn.LayerNorm(d_model)
        self.ln_l2 = nn.LayerNorm(d_model)
        self.ln_v  = nn.LayerNorm(d_model)
        self.ln_a  = nn.LayerNorm(d_model)
        self.ffn_v = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(d_model*4, d_model))
        self.ffn_a = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(d_model*4, d_model))
        self.ln_fv = nn.LayerNorm(d_model)
        self.ln_fa = nn.LayerNorm(d_model)

    def forward(self, vis, aud, lat):
        # latents gather from both streams
        lat = lat + self.attn_lv(self.ln_l1(lat), vis, vis)[0]
        lat = lat + self.attn_la(self.ln_l2(lat), aud, aud)[0]
        # streams read back from latents (time axis preserved)
        vis = vis + self.attn_vl(self.ln_v(vis), lat, lat)[0]
        aud = aud + self.attn_al(self.ln_a(aud), lat, lat)[0]
        vis = vis + self.ffn_v(self.ln_fv(vis))
        aud = aud + self.ffn_a(self.ln_fa(aud))
        return vis, aud, lat


class MBAVFusion(nn.Module):
    """
    visual [B,T,512] + audio [B,T_a,768] -> fused [B,T,d_model]
    """
    def __init__(self, visual_dim=512, audio_dim=768, d_model=256,
                 num_latents=8, num_layers=2, num_heads=8, dropout=0.1,
                 p_drop_visual=0.1, p_drop_audio=0.1):
        super().__init__()
        self.p_drop_visual = p_drop_visual
        self.p_drop_audio  = p_drop_audio
        self.vis_proj = nn.Linear(visual_dim, d_model)
        self.aud_proj = nn.Linear(audio_dim,  d_model)
        self.latents  = nn.Parameter(torch.randn(1, num_latents, d_model) * 0.02)
        self.layers   = nn.ModuleList([
            BottleneckFusionLayer(d_model, num_latents, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model * 2, d_model)
        self.ln_out   = nn.LayerNorm(d_model)

    def _modality_dropout(self, vis, aud):
        if not self.training:
            return vis, aud
        B = vis.size(0)
        if self.p_drop_visual > 0:
            m = (torch.rand(B, 1, 1, device=vis.device) > self.p_drop_visual).float()
            vis = vis * m
        if self.p_drop_audio > 0:
            m = (torch.rand(B, 1, 1, device=aud.device) > self.p_drop_audio).float()
            aud = aud * m
        return vis, aud

    def forward(self, visual_feats, audio_feats):
        B, T, _ = visual_feats.shape
        vis = self.vis_proj(visual_feats)                 # [B,T,d]
        aud = self.aud_proj(audio_feats)                  # [B,T_a,d]
        aud = aud.permute(0, 2, 1)
        aud = F.interpolate(aud, size=T, mode='linear', align_corners=False)
        aud = aud.permute(0, 2, 1)                        # [B,T,d]
        vis, aud = self._modality_dropout(vis, aud)
        lat = self.latents.expand(B, -1, -1)
        for layer in self.layers:
            vis, aud, lat = layer(vis, aud, lat)
        fused = self.out_proj(torch.cat([vis, aud], dim=-1))  # [B,T,d]
        return self.ln_out(fused)
