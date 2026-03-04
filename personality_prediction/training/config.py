"""
training/config.py
Central configuration singleton for the personality prediction project.

All hyperparameters live here. No magic numbers anywhere else.
Import with: from training.config import get_config, TRAIT_KEYS
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List


# ── Trait keys — order is fixed and must match annotation .pkl keys ──────────
TRAIT_KEYS: List[str] = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    data_root:   str = "./data"
    cache_dir:   str = "./feature_cache"
    out_dir:     str = "./experiments"
    batch_size:  int = 32
    num_workers: int = 0          # 0 is safest on Windows (avoids multiprocessing issues)
    trait_keys:  List[str] = field(default_factory=lambda: list(TRAIT_KEYS))


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    audio_dim:  int   = 768
    text_dim:   int   = 768
    visual_dim: int   = 2048
    proj_dim:   int   = 256
    num_heads:  int   = 8
    ff_dim:     int   = 512
    num_layers: int   = 3
    dropout:    float = 0.3
    num_traits: int   = 5


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    lr:             float = 1e-4
    weight_decay:   float = 0.01   # Rule 6 — do not change
    epochs:         int   = 50
    patience:       int   = 7
    warmup_epochs:  int   = 5
    grad_accum:     int   = 2
    seed:           int   = 42


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    data:     DataConfig     = field(default_factory=DataConfig)
    model:    ModelConfig    = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device:   str            = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


# ── Singleton ─────────────────────────────────────────────────────────────────
_config_instance: Config = None


def get_config() -> Config:
    """Return the global config singleton. Creates it on first call."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def reset_config():
    """Reset singleton — useful for tests or re-initialisation."""
    global _config_instance
    _config_instance = None


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = get_config()
    print(f"Device     : {cfg.device}")
    print(f"Batch size : {cfg.data.batch_size}")
    print(f"proj_dim   : {cfg.model.proj_dim}")
    print(f"LR         : {cfg.training.lr}")
    print(f"Traits     : {cfg.data.trait_keys}")
    print("✅ config.py OK")