"""
TACFN — Transformer-based Adaptive Cross-modal Fusion Network
Reference: Liu et al. (2025), CAAI Artificial Intelligence Research, vol.2, p.9150019

Key difference from standard cross-modal attention (E12):
  Stage 1: Intra-modal self-attention filters noise within each modality FIRST
  Stage 2: Adaptive gating (learned sigmoid) controls cross-modal blending
  Residual connections throughout preserve original information

Input:  audio (B,768), text (B,768), visual (B,2048)
Output: OCEAN scores (B,5) in [0,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random, numpy as np


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class IntraModalBlock(nn.Module):
    """
    Stage 1: Self-attention within a single modality.
    Filters noise and highlights salient features BEFORE cross-modal interaction.
    """
    def __init__(self, dim, num_heads=8, dropout=0.3):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        xs = x.unsqueeze(1)
        out, _ = self.attn(xs, xs, xs)
        return self.norm(x + self.drop(out.squeeze(1)))


class AdaptiveCrossModalBlock(nn.Module):
    """
    Stage 2: Cross-modal attention with adaptive gating.

    Standard:   output = CrossAttn(query=target, key/value=source)
    TACFN:      cross  = CrossAttn(target, source)
                gate   = Sigmoid(Linear([target, cross]))
                output = gate * cross + (1-gate) * target

    The learned gate decides how much cross-modal info to blend in.
    If source adds no value, gate→0 and original is preserved.
    """
    def __init__(self, dim, num_heads=8, dropout=0.3):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, target, source):
        q, kv = target.unsqueeze(1), source.unsqueeze(1)
        cross, _ = self.attn(q, kv, kv)
        cross = cross.squeeze(1)
        gate = torch.sigmoid(self.gate(torch.cat([target, cross], dim=-1)))
        return self.norm(gate * cross + (1.0 - gate) * target)


class TACFN(nn.Module):
    """
    TACFN for Big Five personality prediction from audio+text+visual.

    Pipeline:
      1. L2 normalize inputs
      2. Project to D=256
      3. Intra-modal self-attention (Stage 1)
      4. Adaptive cross-modal attention (Stage 2, 6 directed pairs)
      5. Per-modality combination
      6. Transformer encoder over 3 tokens
      7. Regressor → 5 OCEAN scores
    """

    def __init__(self, audio_dim=768, text_dim=768, visual_dim=2048,
                 proj_dim=256, num_heads=8, ff_dim=512, num_layers=3,
                 dropout=0.3, num_traits=5):
        super().__init__()
        D = proj_dim

        def proj(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, D), nn.LayerNorm(D), nn.GELU(), nn.Dropout(dropout))

        # Step 2: Projections
        self.audio_proj  = proj(audio_dim)
        self.text_proj   = proj(text_dim)
        self.visual_proj = proj(visual_dim)

        # Step 3: Intra-modal
        self.audio_intra  = IntraModalBlock(D, num_heads, dropout)
        self.text_intra   = IntraModalBlock(D, num_heads, dropout)
        self.visual_intra = IntraModalBlock(D, num_heads, dropout)

        # Step 4: Adaptive cross-modal (6 directed pairs)
        self.a_from_t = AdaptiveCrossModalBlock(D, num_heads, dropout)
        self.a_from_v = AdaptiveCrossModalBlock(D, num_heads, dropout)
        self.t_from_a = AdaptiveCrossModalBlock(D, num_heads, dropout)
        self.t_from_v = AdaptiveCrossModalBlock(D, num_heads, dropout)
        self.v_from_a = AdaptiveCrossModalBlock(D, num_heads, dropout)
        self.v_from_t = AdaptiveCrossModalBlock(D, num_heads, dropout)

        # Step 5: Combine [refined | adapted1 | adapted2] → D
        def combine():
            return nn.Sequential(nn.Linear(D*3, D), nn.LayerNorm(D), nn.GELU())
        self.audio_combine  = combine()
        self.text_combine   = combine()
        self.visual_combine = combine()

        # Step 6: Transformer encoder
        enc = nn.TransformerEncoderLayer(
            d_model=D, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)

        # Step 7: Regressor
        self.regressor = nn.Sequential(
            nn.Linear(D*3, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, num_traits), nn.Sigmoid()
        )

    def forward(self, audio, text, visual):
        # Step 1: L2 normalize — resolves scale mismatch
        audio  = F.normalize(audio,  p=2, dim=-1)
        text   = F.normalize(text,   p=2, dim=-1)
        visual = F.normalize(visual, p=2, dim=-1)

        # Step 2: Project
        a, t, v = self.audio_proj(audio), self.text_proj(text), self.visual_proj(visual)

        # Step 3: Intra-modal
        ar, tr, vr = self.audio_intra(a), self.text_intra(t), self.visual_intra(v)

        # Step 4: Adaptive cross-modal
        a_final = self.audio_combine( torch.cat([ar, self.a_from_t(ar,tr), self.a_from_v(ar,vr)], -1))
        t_final = self.text_combine(  torch.cat([tr, self.t_from_a(tr,ar), self.t_from_v(tr,vr)], -1))
        v_final = self.visual_combine(torch.cat([vr, self.v_from_a(vr,ar), self.v_from_t(vr,tr)], -1))

        # Step 6: Transformer over tokens
        tokens = torch.stack([a_final, t_final, v_final], dim=1)
        tokens = self.transformer(tokens)
        fused  = tokens.reshape(tokens.size(0), -1)

        return self.regressor(fused)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    set_seed(42)
    m = TACFN()
    print(f"Parameters: {count_parameters(m):,}")
    B = 4
    out = m(torch.randn(B, 768), torch.randn(B, 768), torch.randn(B, 2048))
    assert out.shape == (B, 5) and (out >= 0).all() and (out <= 1).all()
    print(f"Output: {out.shape}, range [{out.min():.3f}, {out.max():.3f}]")
    print("✅ TACFN OK")
