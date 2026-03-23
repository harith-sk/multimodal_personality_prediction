"""
inference/predict.py
Single-video inference using the E12 TransformerFusion model.
Best model: test accuracy 0.9170 on ChaLearn First Impressions V2.

Run from project root:
    python -m inference.predict \
        --video      ./data/test/videos/some_video.mp4 \
        --model_path ./experiments/E12_full_transformer/best_model.pt \
        --config     ./experiments/E12_full_transformer/config.json
"""

import sys
import json
import logging
import argparse
import tempfile
import subprocess
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_extractors.audio_encoder  import get_audio_encoder
from feature_extractors.text_encoder   import get_text_encoder
from feature_extractors.visual_encoder import get_visual_encoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRAIT_KEYS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
NUM_FRAMES  = 16
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BAR_WIDTH   = 20


# ─────────────────────────────────────────────────────────────────────────────
# TransformerFusion model (E12) — embedded here so inference needs no extras
# Identical to training/train_experiments.py:TransformerFusion
# ─────────────────────────────────────────────────────────────────────────────

class TransformerFusion(nn.Module):
    """
    6 cross-attention pairs + TransformerEncoder over 3 modality tokens.
    Best performing model — test accuracy 0.9170.
    """
    def __init__(self, proj_dim: int = 256, num_heads: int = 8,
                 ff_dim: int = 512, num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        D = proj_dim

        self.audio_proj  = self._proj(768,  D, dropout)
        self.text_proj   = self._proj(768,  D, dropout)
        self.visual_proj = self._proj(2048, D, dropout)

        def ca():
            return nn.MultiheadAttention(D, num_heads, dropout=dropout, batch_first=True)

        self.a_from_t = ca(); self.a_from_v = ca()
        self.t_from_a = ca(); self.t_from_v = ca()
        self.v_from_a = ca(); self.v_from_t = ca()

        self.norm_a = nn.LayerNorm(D)
        self.norm_t = nn.LayerNorm(D)
        self.norm_v = nn.LayerNorm(D)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(D * 3, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 5), nn.Sigmoid(),
        )

    @staticmethod
    def _proj(in_d, out_d, drop):
        return nn.Sequential(
            nn.Linear(in_d, out_d), nn.LayerNorm(out_d), nn.GELU(), nn.Dropout(drop)
        )

    def _cross(self, module, query, key):
        q  = query.unsqueeze(1)
        kv = key.unsqueeze(1)
        out, _ = module(q, kv, kv)
        return out.squeeze(1)

    def forward(self, audio, text, visual):
        audio  = F.normalize(audio,  p=2, dim=-1)
        text   = F.normalize(text,   p=2, dim=-1)
        visual = F.normalize(visual, p=2, dim=-1)

        a = self.audio_proj(audio)
        t = self.text_proj(text)
        v = self.visual_proj(visual)

        a_out = self.norm_a(a + self._cross(self.a_from_t, a, t) + self._cross(self.a_from_v, a, v))
        t_out = self.norm_t(t + self._cross(self.t_from_a, t, a) + self._cross(self.t_from_v, t, v))
        v_out = self.norm_v(v + self._cross(self.v_from_a, v, a) + self._cross(self.v_from_t, v, t))

        tokens = torch.stack([a_out, t_out, v_out], dim=1)  # (B, 3, D)
        tokens = self.transformer(tokens)                     # (B, 3, D)
        fused  = tokens.reshape(tokens.size(0), -1)          # (B, 3D)
        return self.head(fused)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_audio(video_path: str) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1",
             "-vn", "-f", "wav", tmp_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
        )
        audio, _ = sf.read(tmp_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return get_audio_encoder().encode(audio).astype(np.float32)
    except Exception as e:
        logger.warning(f"Audio extraction failed: {e}")
        return np.zeros(768, dtype=np.float32)
    finally:
        try: os.unlink(tmp_path)
        except: pass


def extract_text(video_path: str):
    try:
        result = get_text_encoder().process(video_path)
        return np.array(result["text_features"], dtype=np.float32), result.get("transcript", "")
    except Exception as e:
        logger.warning(f"Text extraction failed: {e}")
        return np.zeros(768, dtype=np.float32), ""


def extract_visual(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros(2048, dtype=np.float32)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return np.zeros(2048, dtype=np.float32)
    indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return get_visual_encoder().encode(frames).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str, config_path: str) -> TransformerFusion:
    with open(config_path) as f:
        cfg = json.load(f)

    model = TransformerFusion(
        proj_dim=cfg.get("proj_dim",   256),
        num_heads=cfg.get("num_heads",   8),
        ff_dim=cfg.get("ff_dim",       512),
        num_layers=cfg.get("num_layers", 3),
        dropout=cfg.get("dropout",     0.3),
    ).to(DEVICE)

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)   # handles both save formats
    model.load_state_dict(state)
    model.eval()
    logger.info(f"TransformerFusion (E12) loaded from {model_path}  (epoch {ckpt.get('epoch', '?')})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Pretty Print
# ─────────────────────────────────────────────────────────────────────────────

def print_results(scores: dict, transcript: str):
    print("\n" + "="*58)
    print("  PERSONALITY PREDICTION — TransformerFusion (E12)")
    print("="*58)
    if transcript:
        preview = transcript[:80] + ("..." if len(transcript) > 80 else "")
        print(f"  Transcript : {preview}")
        print("-"*58)

    print(f"  {'Trait':<22} {'Score':>6}  {'Profile'}")
    print("-"*58)

    for trait in TRAIT_KEYS:
        score  = scores[trait]
        filled = int(round(score * BAR_WIDTH))
        bar    = "█" * filled + "░" * (BAR_WIDTH - filled)
        print(f"  {trait.capitalize():<22} {score:>6.3f}  {bar}")

    mean = np.mean(list(scores.values()))
    print("-"*58)
    print(f"  {'Mean':<22} {mean:>6.3f}")
    print("="*58 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Core inference function — call this from other code
# ─────────────────────────────────────────────────────────────────────────────

def predict(video_path: str, model_path: str, config_path: str) -> dict:
    """
    Run personality prediction on a single video.

    Args:
        video_path:  Path to any .mp4 video file
        model_path:  Path to E12 best_model.pt
        config_path: Path to E12 config.json

    Returns:
        dict with keys: openness, conscientiousness, extraversion,
                        agreeableness, neuroticism
        All values are floats in [0.0, 1.0]. Higher = stronger trait.
    """
    logger.info(f"Processing: {video_path}")

    audio_feat           = extract_audio(video_path)
    text_feat, transcript = extract_text(video_path)
    visual_feat          = extract_visual(video_path)

    model = load_model(model_path, config_path)

    audio_t  = torch.from_numpy(audio_feat).unsqueeze(0).to(DEVICE)
    text_t   = torch.from_numpy(text_feat).unsqueeze(0).to(DEVICE)
    visual_t = torch.from_numpy(visual_feat).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = model(audio_t, text_t, visual_t)   # (1, 5)

    scores = {trait: float(preds[0, i]) for i, trait in enumerate(TRAIT_KEYS)}
    print_results(scores, transcript)
    return scores


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="E12 TransformerFusion single-video inference")
    p.add_argument("--video",      required=True, help="Path to .mp4 video file")
    p.add_argument("--model_path", required=True, help="Path to E12 best_model.pt checkpoint")
    p.add_argument("--config",     required=True, help="Path to E12 config.json")
    args = p.parse_args()
    predict(args.video, args.model_path, args.config)