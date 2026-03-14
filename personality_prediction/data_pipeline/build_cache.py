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

TRANSCRIPTION_FILES = {
    "train": "transcription_training.pkl",
    "val":   "transcription_validation.pkl",
    "test":  "transcription_test.pkl",
}

# ChaLearn V2: splits may be named 'val' OR 'validate' on disk
# Maps canonical name → possible actual folder names to try
SPLIT_DIR_CANDIDATES = {
    "train": ["train"],
    "val":   ["val", "validate", "validation"],
    "test":  ["test"],
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


def _load_transcriptions(split_dir: Path, data_root: Path, split: str) -> dict:
    """
    Load pre-computed transcriptions from the dataset if available.
    Returns {video_stem: transcript_text} or {} if not found.

    ChaLearn V2 format: {filename.mp4: transcript_str}
    We convert keys to stems for fast lookup.
    """
    fname = TRANSCRIPTION_FILES.get(split)
    if not fname:
        return {}

    for candidate in [split_dir / fname, data_root / fname]:
        if candidate.exists():
            with open(candidate, "rb") as f:
                raw = pickle.load(f, encoding="latin1")
            # Convert {filename.mp4: text} → {stem: text}
            stemmed = {Path(k).stem: v for k, v in raw.items() if isinstance(v, str)}
            logger.info(
                f"  Loaded pre-computed transcriptions: {len(stemmed)} entries "
                f"from {candidate.name} — Whisper will be SKIPPED ✅"
            )
            return stemmed

    logger.info("  No transcription pkl found — will run Whisper for each video")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
def extract_text_features(video_path: Path, preloaded_text: str = None):
    """
    Encode text to RoBERTa features (768,).

    If preloaded_text is given, skips Whisper transcription entirely and
    encodes the pre-loaded transcript directly with RoBERTa.
    Returns (features np.ndarray (768,), transcript str).
    """
    encoder = get_text_encoder()
    try:
        if preloaded_text:
            # Fast path: skip Whisper, encode pre-loaded transcript directly
            features = encoder.encode_text(preloaded_text)
            return np.array(features, dtype=np.float32), preloaded_text

        # Slow path: run Whisper transcription
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
def _find_split_dir(data_root: Path, split: str) -> Path:
    """
    Resolve the actual folder name for a split on disk.
    ChaLearn V2 may use 'val', 'validate', or 'validation'.
    """
    for candidate in SPLIT_DIR_CANDIDATES.get(split, [split]):
        p = data_root / candidate
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot find split folder for '{split}' under {data_root}. "
        f"Tried: {SPLIT_DIR_CANDIDATES.get(split, [split])}"
    )


def _build_video_index(split_dir: Path, split: str) -> dict:
    """
    Build a dict {video_stem: full_path} by scanning ALL subfolders.

    ChaLearn V2 structure:
      train/ → train_1/ ... train_6/  (6 subfolders, ~1000 videos each)
      val/   → val_1/ val_2/          (2 subfolders)
      test/  → test_1/ test_2/        (2 subfolders)

    Also handles flat structure (all videos directly in split_dir/videos/
    or directly in split_dir) as a fallback.
    """
    index = {}

    # 1. Look for numbered subfolders: train_1, train_2, val_1, etc.
    numbered = sorted(split_dir.glob(f"{split}_*/"))

    # Also accept generic subfolder names (e.g. 'videos/', 'clips/')
    if not numbered:
        numbered = [p for p in split_dir.iterdir() if p.is_dir()]

    if numbered:
        for subfolder in numbered:
            for mp4 in subfolder.glob("*.mp4"):
                index[mp4.stem] = mp4
        if index:
            logger.info(f"  Found {len(index)} videos across {len(numbered)} subfolders")
            return index

    # 2. Fallback: videos directly in split_dir
    for mp4 in split_dir.glob("*.mp4"):
        index[mp4.stem] = mp4

    if index:
        logger.info(f"  Found {len(index)} videos directly in {split_dir.name}/")

    return index


TRAIT_KEYS_SET = {
    "openness", "conscientiousness", "extraversion",
    "agreeableness", "neuroticism", "interview",
}


def _normalise_annotations(raw: dict) -> dict:
    """
    ChaLearn V2 .pkl files come in two layouts:

    Layout A — file-keyed (what the code originally expected):
        { 'video001.mp4': {'openness': 0.5, 'extraversion': 0.6, ...}, ... }

    Layout B — trait-keyed (what the training PC dataset actually has):
        { 'openness': {'video001.mp4': 0.5, ...}, 'extraversion': {...}, ... }

    This function detects which layout is in use and normalises it to Layout A.
    """
    if not raw:
        return raw

    first_key = next(iter(raw))

    # Layout B: top-level keys are trait names
    if first_key.lower() in TRAIT_KEYS_SET:
        logger.info(
            "Detected trait-keyed annotation format (Layout B) — converting to file-keyed."
        )
        normalised = {}
        for trait, file_scores in raw.items():
            if not isinstance(file_scores, dict):
                continue
            for fname, score in file_scores.items():
                if fname not in normalised:
                    normalised[fname] = {}
                normalised[fname][trait] = float(score)
        logger.info(
            f"  Converted {len(normalised)} video entries from {len(raw)} trait keys."
        )
        return normalised

    # Layout A: already file-keyed — return as-is
    logger.info("Detected file-keyed annotation format (Layout A) — no conversion needed.")
    return raw


def process_split(split: str, data_root: Path, cache_dir: Path):
    """Process all videos in one split and save .pt cache files."""

    split_dir   = _find_split_dir(data_root, split)
    ann_file    = split_dir / ANNOTATION_FILES[split]
    split_cache = cache_dir / split
    split_cache.mkdir(parents=True, exist_ok=True)

    # Load annotation to get official list + labels
    if not ann_file.exists():
        # Some releases put annotation outside the split folder
        ann_file = data_root / ANNOTATION_FILES[split]
    if not ann_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found.\n"
            f"  Tried: {split_dir / ANNOTATION_FILES[split]}\n"
            f"  Tried: {data_root / ANNOTATION_FILES[split]}"
        )

    with open(ann_file, "rb") as f:
        raw_annotations = pickle.load(f, encoding="latin1")

    # Normalise to {filename: {trait: score}} regardless of source format
    annotations = _normalise_annotations(raw_annotations)

    # Load pre-computed transcriptions (skips Whisper if found)
    transcriptions = _load_transcriptions(split_dir, data_root, split)

    # Build a fast stem→path lookup across all video subfolders
    video_index = _build_video_index(split_dir, split)

    video_names = list(annotations.keys())
    total       = len(video_names)


    logger.info(f"\n{'='*60}")
    logger.info(f"Split : {split}  |  Folder: {split_dir.name}")
    logger.info(f"Labels: {total}  |  Videos on disk: {len(video_index)}")
    logger.info(f"Cache : {split_cache}")
    logger.info(f"{'='*60}")

    done = skipped = failed = 0

    for video_name in tqdm(video_names, desc=f"{split}", unit="video"):
        stem       = Path(video_name).stem
        cache_file = split_cache / f"{stem}.pt"

        # ── Skip if already cached (RESUMABLE) ───────────────────────────────
        if cache_file.exists():
            skipped += 1
            continue

        # Look up video path using index (handles multi-subfolder layout)
        video_path = video_index.get(stem)
        if video_path is None:
            logger.warning(f"Video not found in index: {video_name}")
            failed += 1
            continue

        try:
            # Look up pre-computed transcript for this video (skips Whisper if found)
            preloaded_text = transcriptions.get(stem)

            audio_feat   = extract_audio_features(video_path)
            text_feat, _ = extract_text_features(video_path, preloaded_text=preloaded_text)
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