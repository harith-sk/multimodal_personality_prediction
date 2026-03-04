"""
training/dataset.py
PyTorch Dataset and DataLoader factory for First Impressions V2.

Loads pre-built feature cache (.pt files) instead of re-processing
videos on every run. Cache is built once by data_pipeline/build_cache.py.

Usage:
    from training.dataset import get_dataloaders, TRAIT_KEYS
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root='./data',
        cache_dir='./feature_cache',
        batch_size=32,
        num_workers=0
    )
"""

import pickle
import logging
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ── Trait keys — fixed order, must match annotation .pkl keys ────────────────
TRAIT_KEYS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

# ChaLearn V2 may use 'val', 'validate', or 'validation' as the split folder name
SPLIT_DIR_CANDIDATES = {
    "train": ["train"],
    "val":   ["val", "validate", "validation"],
    "test":  ["test"],
}


def _resolve_ann_file(data_root: Path, split: str) -> Path:
    """
    Find the annotation pickle regardless of folder naming conventions.
    Search order:
      1. data_root/<split_folder>/annotation_<split>.pkl
      2. data_root/annotation_<split>.pkl  (flat layout)
    """
    ann_name = ANNOTATION_FILES[split]
    for candidate in SPLIT_DIR_CANDIDATES.get(split, [split]):
        p = data_root / candidate / ann_name
        if p.exists():
            return p
    # Flat fallback
    p = data_root / ann_name
    if p.exists():
        return p
    raise FileNotFoundError(
        f"Annotation file '{ann_name}' not found for split '{split}'.\n"
        f"Searched under: {data_root}\n"
        f"Tried folders : {SPLIT_DIR_CANDIDATES.get(split, [split])} + root"
    )


# ─────────────────────────────────────────────────────────────────────────────
class PersonalityDataset(Dataset):
    """
    Loads cached features (.pt files) and annotation labels for one split.

    Each .pt file contains:
        {
            'audio':  torch.Tensor shape (768,)
            'text':   torch.Tensor shape (768,)
            'visual': torch.Tensor shape (2048,)
        }

    __getitem__ returns:
        features: dict with keys 'audio', 'text', 'visual'
        labels:   torch.Tensor shape (5,) — OCEAN scores in [0, 1]
    """

    def __init__(self, split: str, data_root: str, cache_dir: str):
        """
        Args:
            split:      'train', 'val', or 'test'
            data_root:  path to data/ folder
            cache_dir:  path to feature_cache/ folder
        """
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"

        self.split      = split
        self.cache_path = Path(cache_dir) / split
        self.data_root  = Path(data_root)

        # ── Load annotations ──────────────────────────────────────────────────
        ann_file = _resolve_ann_file(self.data_root, split)

        with open(ann_file, "rb") as f:
            annotations = pickle.load(f, encoding="latin1")

        # ── Build sample list: (video_name_stem, label_tensor) ───────────────
        # annotation keys are video filenames like '0021.mp4'
        self.samples = []
        skipped = 0
        for video_name, trait_dict in annotations.items():
            try:
                label = torch.tensor(
                    [float(trait_dict[t]) for t in TRAIT_KEYS],
                    dtype=torch.float32
                )
                # Clamp to [0, 1] — handles rare annotation noise
                label = label.clamp(0.0, 1.0)
                # Store stem without extension for cache lookup
                stem = Path(video_name).stem
                self.samples.append((stem, label))
            except (KeyError, ValueError, TypeError) as e:
                skipped += 1
                logger.warning(f"Skipping {video_name}: {e}")

        if skipped > 0:
            logger.warning(f"{split}: skipped {skipped} samples with bad annotations")

        logger.info(f"PersonalityDataset [{split}]: {len(self.samples)} samples loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        stem, label = self.samples[idx]
        cache_file  = self.cache_path / f"{stem}.pt"

        if cache_file.exists():
            try:
                data = torch.load(cache_file, map_location="cpu", weights_only=True)
                features = {
                    "audio":  data["audio"].float(),   # (768,)
                    "text":   data["text"].float(),    # (768,)
                    "visual": data["visual"].float(),  # (2048,)
                }
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_file}: {e} — using zeros")
                features = _zero_features()
        else:
            logger.warning(f"Cache file missing: {cache_file} — using zeros")
            features = _zero_features()

        return features, label


def _zero_features() -> Dict[str, torch.Tensor]:
    """Return zero-filled feature tensors as a safe fallback."""
    return {
        "audio":  torch.zeros(768,  dtype=torch.float32),
        "text":   torch.zeros(768,  dtype=torch.float32),
        "visual": torch.zeros(2048, dtype=torch.float32),
    }


# ─────────────────────────────────────────────────────────────────────────────
def get_dataloaders(
    data_root:   str,
    cache_dir:   str,
    batch_size:  int = 32,
    num_workers: int = 0,       # 0 is safest on Windows
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build and return (train_loader, val_loader, test_loader).

    Args:
        data_root:   path to data/ folder
        cache_dir:   path to feature_cache/ folder
        batch_size:  samples per batch
        num_workers: DataLoader worker processes (use 0 on Windows)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    pin = torch.cuda.is_available()

    train_ds = PersonalityDataset("train", data_root, cache_dir)
    val_ds   = PersonalityDataset("val",   data_root, cache_dir)
    test_ds  = PersonalityDataset("test",  data_root, cache_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    logger.info(
        f"DataLoaders ready — "
        f"train:{len(train_ds)}  val:{len(val_ds)}  test:{len(test_ds)}  "
        f"batch:{batch_size}"
    )
    return train_loader, val_loader, test_loader


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data_root  = sys.argv[1] if len(sys.argv) > 1 else "./data"
    cache_dir  = sys.argv[2] if len(sys.argv) > 2 else "./feature_cache"

    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=data_root,
        cache_dir=cache_dir,
        batch_size=4,
        num_workers=0,
    )

    features, labels = next(iter(train_loader))
    print(f"audio  shape : {features['audio'].shape}")    # (4, 768)
    print(f"text   shape : {features['text'].shape}")     # (4, 768)
    print(f"visual shape : {features['visual'].shape}")   # (4, 2048)
    print(f"labels shape : {labels.shape}")               # (4, 5)
    print(f"labels range : [{labels.min():.3f}, {labels.max():.3f}]")
    print("✅ dataset.py OK")