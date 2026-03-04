"""
data_pipeline/build_cache.py
Builds the feature cache for all splits of First Impressions V2.

For each video, runs all three encoders and saves one .pt file:
    feature_cache/train/0021.pt  →  {'audio': (768,), 'text': (768,), 'visual': (2048,)}

This script is RESUMABLE — if it crashes, re-running will skip videos
that already have a .pt file. Do NOT delete the cache after it is built
(Rule 4 in the training guide). Rebuilding takes 3–4 hours.

Run from project root:
    python -m data_pipeline.build_cache --data_root ./data --cache_dir ./feature_cache

Windows note:
    ffmpeg must be on your system PATH.
    Download from https://ffmpeg.org/download.html and add to PATH.
    Verify with: ffmpeg -version
"""

import os
import sys
import pickle
import logging
import argparse
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import cv2
import torch
from tqdm import tqdm

# ── Add project root to path so imports work ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_extractors.audio_encoder  import get_audio_encoder
from feature_extractors.text_encoder   import get_text_encoder
from feature_extractors.visual_encoder import get_visual_encoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Fixed number of frames to sample per video (Rule from guide)
NUM_FRAMES = 16

ANNOTATION_FILES = {
    "train": "annotation_training.pkl",
    "val":   "annotation_validation.pkl",
    "test":  "annotation_test.pkl",
}


# ─────────────────────────────────────────────────────────────────────────────
def extract_audio_features(video_path: Path) -> np.ndarray:
    """
    Extract audio features from video using ffmpeg + WavLM.

    Steps:
    1. ffmpeg extracts 16kHz mono WAV to a temp file
    2. soundfile loads the WAV as float32 numpy
    3. WavLM encoder encodes to (768,)

    Returns np.ndarray (768,) or zeros if extraction fails.
    """
    encoder = get_audio_encoder()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # ffmpeg command — works on Windows with ffmpeg on PATH
        cmd = [
            "ffmpeg",
            "-y",                        # overwrite without asking
            "-i", str(video_path),       # input video
            "-ar", "16000",              # resample to 16kHz
            "-ac", "1",                  # mono
            "-vn",                       # no video
            "-f", "wav",                 # output format
            tmp_path,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,                  # 60s max per video
        )

        if result.returncode != 0:
            logger.warning(f"ffmpeg failed for {video_path.name}")
            return np.zeros(768, dtype=np.float32)

        audio, sr = sf.read(tmp_path, dtype="float32")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)   # stereo → mono (safety)

        if len(audio) == 0:
            logger.warning(f"Empty audio for {video_path.name}")
            return np.zeros(768, dtype=np.float32)

        return encoder.encode(audio).astype(np.float32)

    except Exception as e:
        logger.warning(f"Audio extraction failed for {video_path.name}: {e}")
        return np.zeros(768, dtype=np.float32)

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
def extract_text_features(video_path: Path):
    """
    Transcribe video audio with Whisper then encode with RoBERTa.
    Returns (features np.ndarray (768,), transcript str).
    """
    encoder = get_text_encoder()
    try:
        result = encoder.process(str(video_path))
        features = np.array(result["text_features"], dtype=np.float32)
        transcript = result.get("transcript", "")
        return features, transcript
    except Exception as e:
        logger.warning(f"Text extraction failed for {video_path.name}: {e}")
        return np.zeros(768, dtype=np.float32), ""


# ─────────────────────────────────────────────────────────────────────────────
def extract_visual_features(video_path: Path) -> np.ndarray:
    """
    Sample NUM_FRAMES frames uniformly from video, encode with ResNet50.
    Returns np.ndarray (2048,).
    """
    encoder = get_visual_encoder()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"OpenCV cannot open {video_path.name}")
        return np.zeros(2048, dtype=np.float32)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        logger.warning(f"No frames found in {video_path.name}")
        return np.zeros(2048, dtype=np.float32)

    # Uniformly spaced frame indices
    indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            # OpenCV reads BGR → convert to RGB for ResNet50
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()

    if not frames:
        logger.warning(f"No readable frames from {video_path.name}")
        return np.zeros(2048, dtype=np.float32)

    return encoder.encode(frames).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
def process_split(split: str, data_root: Path, cache_dir: Path):
    """Process all videos in one split and save .pt cache files."""

    video_dir  = data_root / split / "videos"
    ann_file   = data_root / split / ANNOTATION_FILES[split]
    split_cache = cache_dir / split
    split_cache.mkdir(parents=True, exist_ok=True)

    # Load annotation to get the official list of video filenames
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    with open(ann_file, "rb") as f:
        annotations = pickle.load(f, encoding="latin1")

    video_names = list(annotations.keys())
    total       = len(video_names)

    logger.info(f"\n{'='*60}")
    logger.info(f"Split: {split}  |  {total} videos  |  Cache: {split_cache}")
    logger.info(f"{'='*60}")

    done = skipped = failed = 0

    for video_name in tqdm(video_names, desc=f"{split}", unit="video"):
        stem       = Path(video_name).stem
        cache_file = split_cache / f"{stem}.pt"

        # ── Skip if already cached (RESUMABLE) ───────────────────────────────
        if cache_file.exists():
            skipped += 1
            continue

        video_path = video_dir / video_name
        if not video_path.exists():
            logger.warning(f"Video not found: {video_path}")
            failed += 1
            continue

        try:
            audio_feat   = extract_audio_features(video_path)
            text_feat, _ = extract_text_features(video_path)
            visual_feat  = extract_visual_features(video_path)

            cache_data = {
                "audio":  torch.from_numpy(audio_feat),   # (768,)
                "text":   torch.from_numpy(text_feat),    # (768,)
                "visual": torch.from_numpy(visual_feat),  # (2048,)
            }
            torch.save(cache_data, cache_file)
            done += 1

        except Exception as e:
            logger.error(f"Failed to process {video_name}: {e}")
            failed += 1

    logger.info(
        f"{split} complete — "
        f"processed:{done}  already_cached:{skipped}  failed:{failed}"
    )
    return done, skipped, failed


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Build feature cache for First Impressions V2"
    )
    parser.add_argument("--data_root",  default="./data",          help="Path to data/ folder")
    parser.add_argument("--cache_dir",  default="./feature_cache", help="Where to save .pt files")
    parser.add_argument("--splits",     nargs="+", default=["train", "val", "test"],
                        choices=["train", "val", "test"],
                        help="Which splits to process (default: all three)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Pre-loading all three encoders...")
    get_audio_encoder()
    get_text_encoder()
    get_visual_encoder()
    logger.info("All encoders loaded\n")

    total_done = total_skipped = total_failed = 0

    for split in args.splits:
        d, s, f = process_split(split, data_root, cache_dir)
        total_done    += d
        total_skipped += s
        total_failed  += f

    logger.info(f"\n{'='*60}")
    logger.info(f"CACHE BUILD COMPLETE")
    logger.info(f"  Processed : {total_done}")
    logger.info(f"  Skipped   : {total_skipped}  (already cached)")
    logger.info(f"  Failed    : {total_failed}")
    logger.info(f"{'='*60}")
    logger.info("Next step: python -m data_pipeline.diagnostics")


if __name__ == "__main__":
    main()