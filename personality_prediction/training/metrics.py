"""
training/metrics.py
Evaluation metrics for OCEAN personality trait regression.

Official First Impressions V2 metric:
    Mean Accuracy = 1.0 - Mean Absolute Error
    (higher is better, random baseline ≈ 0.50, target ≈ 0.911)
"""

import numpy as np
from typing import Dict

TRAIT_KEYS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Compute per-trait MAE and mean accuracy.

    Args:
        preds:   np.ndarray shape (N, 5) — predicted OCEAN scores
        targets: np.ndarray shape (N, 5) — ground truth OCEAN scores

    Returns:
        dict with keys:
            'openness', 'conscientiousness', 'extraversion',
            'agreeableness', 'neuroticism'  ← per-trait MAE
            'mean_mae'       ← mean MAE across all traits
            'mean_accuracy'  ← 1.0 - mean_mae  (the official metric)
    """
    assert preds.shape == targets.shape, (
        f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}"
    )
    assert preds.shape[1] == 5, f"Expected 5 traits, got {preds.shape[1]}"

    # Per-trait MAE — shape (5,)
    per_trait_mae = np.abs(preds - targets).mean(axis=0)

    metrics = {}
    for i, trait in enumerate(TRAIT_KEYS):
        metrics[trait] = float(per_trait_mae[i])

    metrics["mean_mae"]      = float(per_trait_mae.mean())
    metrics["mean_accuracy"] = float(1.0 - per_trait_mae.mean())

    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """Return a single-line string summary for logging."""
    trait_str = "  ".join(
        f"{k[0].upper()}:{metrics[k]:.4f}" for k in TRAIT_KEYS
    )
    return (
        f"MeanAcc:{metrics['mean_accuracy']:.4f}  "
        f"MeanMAE:{metrics['mean_mae']:.4f}  |  {trait_str}"
    )


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    preds   = np.random.rand(100, 5).astype(np.float32)
    targets = np.random.rand(100, 5).astype(np.float32)
    m = compute_metrics(preds, targets)

    print("Per-trait MAE:")
    for trait in TRAIT_KEYS:
        print(f"  {trait:20s}: {m[trait]:.4f}")
    print(f"Mean MAE      : {m['mean_mae']:.4f}")
    print(f"Mean Accuracy : {m['mean_accuracy']:.4f}")
    print(f"\nFormatted: {format_metrics(m)}")
    print("✅ metrics.py OK")