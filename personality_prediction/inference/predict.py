"""
inference/predict.py
Single-video inference — outputs OCEAN personality scores.

Used for the project demo. Takes one .mp4 file and the trained
TACFN checkpoint, runs all three encoders, prints a formatted
personality profile table.

Run from project root:
    python inference/predict.py ^
        --video      ./data/test/videos/some_video.mp4 ^
        --model_path ./experiments/E13_TACFN_final/best_model.pt ^
        --config     ./experiments/E13_TACFN_final/config.json
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
import cv2
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_extractors.audio_encoder  import get_audio_encoder
from feature_extractors.text_encoder   import get_text_encoder
from feature_extractors.visual_encoder import get_visual_encoder
from training.tacfn_model import TACFN

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRAIT_KEYS  = ["openness","conscientiousness","extraversion","agreeableness","neuroticism"]
NUM_FRAMES  = 16
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BAR_WIDTH   = 20


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction (same logic as build_cache.py)
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
        return np.array(result["text_features"], dtype=np.float32), result.get("transcript","")
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

def load_model(model_path: str, config_path: str) -> TACFN:
    with open(config_path) as f:
        cfg = json.load(f)

    model = TACFN(
        proj_dim=cfg.get("proj_dim",   256),
        num_heads=cfg.get("num_heads",   8),
        ff_dim=cfg.get("ff_dim",       512),
        num_layers=cfg.get("num_layers", 3),
        dropout=cfg.get("dropout",     0.3),
    ).to(DEVICE)

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Model loaded from {model_path}  (epoch {ckpt.get('epoch','?')})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Pretty Print
# ─────────────────────────────────────────────────────────────────────────────

def print_results(scores: dict, transcript: str):
    print("\n" + "="*58)
    print("  PERSONALITY PREDICTION — TACFN")
    print("="*58)
    if transcript:
        preview = transcript[:80] + ("..." if len(transcript) > 80 else "")
        print(f"  Transcript : {preview}")
        print("-"*58)

    header = f"  {'Trait':<22} {'Score':>6}  {'Profile'}"
    print(header)
    print("-"*58)

    for trait in TRAIT_KEYS:
        score    = scores[trait]
        filled   = int(round(score * BAR_WIDTH))
        bar      = "█" * filled + "░" * (BAR_WIDTH - filled)
        print(f"  {trait.capitalize():<22} {score:>6.3f}  {bar}")

    mean = np.mean(list(scores.values()))
    print("-"*58)
    print(f"  {'Mean':<22} {mean:>6.3f}")
    print("="*58 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def predict(video_path: str, model_path: str, config_path: str):
    logger.info(f"Processing: {video_path}")

    logger.info("Extracting audio features...")
    audio_feat = extract_audio(video_path)

    logger.info("Extracting text features (Whisper transcription)...")
    text_feat, transcript = extract_text(video_path)

    logger.info("Extracting visual features...")
    visual_feat = extract_visual(video_path)

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
    p = argparse.ArgumentParser(description="TACFN single-video inference")
    p.add_argument("--video",      required=True, help="Path to .mp4 video file")
    p.add_argument("--model_path", required=True, help="Path to best_model.pt checkpoint")
    p.add_argument("--config",     required=True, help="Path to config.json for this checkpoint")
    args = p.parse_args()
    predict(args.video, args.model_path, args.config)