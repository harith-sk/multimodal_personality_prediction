"""
data_pipeline/diagnostics.py
Pre-training health checks for the feature cache.

Run this after build_cache.py completes and BEFORE any training.
Do not start training until this reports zero issues.

Run from project root:
    python -m data_pipeline.diagnostics --cache_dir ./feature_cache --data_root ./data
"""

import sys
import pickle
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works on Windows without display
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TRAIT_KEYS = [
    "openness", "conscientiousness", "extraversion",
    "agreeableness", "neuroticism",
]
ANNOTATION_FILES = {
    "train": "annotation_training.pkl",
    "val":   "annotation_validation.pkl",
    "test":  "annotation_test.pkl",
}
EXPECTED_DIMS = {"audio": 768, "text": 768, "visual": 2048}

_TRAIT_KEYS_SET = {
    "openness", "conscientiousness", "extraversion",
    "agreeableness", "neuroticism", "interview",
}


def _normalise_annotations(raw: dict) -> dict:
    """Converts trait-keyed {trait:{file:score}} to file-keyed {file:{trait:score}}."""
    if not raw:
        return raw
    first_key = next(iter(raw))
    if first_key.lower() in _TRAIT_KEYS_SET:
        normalised = {}
        for trait, file_scores in raw.items():
            if not isinstance(file_scores, dict):
                continue
            for fname, score in file_scores.items():
                if fname not in normalised:
                    normalised[fname] = {}
                normalised[fname][trait] = float(score)
        return normalised
    return raw



# ─────────────────────────────────────────────────────────────────────────────
def check_split(split: str, data_root: Path, cache_dir: Path) -> dict:
    """Run all checks for one split. Returns a results dict."""
    results   = {"split": split, "issues": []}
    cache_path = cache_dir / split
    ann_file   = data_root / split / ANNOTATION_FILES[split]

    # ── 1. Annotation file ────────────────────────────────────────────────────
    if not ann_file.exists():
        results["issues"].append(f"MISSING annotation file: {ann_file}")
        return results

    with open(ann_file, "rb") as f:
        raw_annotations = pickle.load(f, encoding="latin1")
    annotations    = _normalise_annotations(raw_annotations)
    expected_count = len(annotations)

    # ── 2. Cache completeness ─────────────────────────────────────────────────
    pt_files = list(cache_path.glob("*.pt"))
    found    = len(pt_files)
    pct      = (found / expected_count * 100) if expected_count > 0 else 0
    results["cache_completeness"] = f"{found}/{expected_count} ({pct:.1f}%)"

    if pct < 95.0:
        results["issues"].append(
            f"Cache only {pct:.1f}% complete ({found}/{expected_count}). "
            f"Rerun build_cache.py."
        )

    # ── 3. Sample features for quality checks ────────────────────────────────
    sample_size = min(200, found)
    sample_files = pt_files[:sample_size]

    dim_errors   = {m: 0 for m in EXPECTED_DIMS}
    nan_counts   = {m: 0 for m in EXPECTED_DIMS}
    zero_counts  = {m: 0 for m in EXPECTED_DIMS}
    stds         = {m: [] for m in EXPECTED_DIMS}

    for pt_file in sample_files:
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=True)
            for mod, expected_dim in EXPECTED_DIMS.items():
                if mod not in data:
                    dim_errors[mod] += 1
                    continue
                t = data[mod].float().numpy()

                if t.shape[0] != expected_dim:
                    dim_errors[mod] += 1

                if np.isnan(t).any():
                    nan_counts[mod] += 1

                if np.all(t == 0):
                    zero_counts[mod] += 1

                stds[mod].append(float(np.std(t)))

        except Exception as e:
            results["issues"].append(f"Cannot load {pt_file.name}: {e}")

    # ── 4. Report per-modality ────────────────────────────────────────────────
    results["modality_checks"] = {}
    for mod in EXPECTED_DIMS:
        if sample_size == 0:
            results["modality_checks"][mod] = {
                "dim_errors": 0, "nan_rate": "N/A",
                "zero_rate": "N/A", "mean_std": "N/A",
            }
            continue
        nan_rate  = nan_counts[mod]  / sample_size * 100
        zero_rate = zero_counts[mod] / sample_size * 100
        mean_std  = float(np.mean(stds[mod])) if stds[mod] else 0.0

        results["modality_checks"][mod] = {
            "dim_errors":  dim_errors[mod],
            "nan_rate":    f"{nan_rate:.2f}%",
            "zero_rate":   f"{zero_rate:.2f}%",
            "mean_std":    f"{mean_std:.4f}",
        }

        if dim_errors[mod] > 0:
            results["issues"].append(
                f"{mod}: {dim_errors[mod]} files have wrong dimension "
                f"(expected {EXPECTED_DIMS[mod]})"
            )
        if nan_rate > 1.0:
            results["issues"].append(
                f"{mod}: NaN rate {nan_rate:.2f}% exceeds 1% threshold"
            )
        if zero_rate > 5.0:
            results["issues"].append(
                f"{mod}: Zero vector rate {zero_rate:.2f}% exceeds 5% threshold"
            )

    return results


# ─────────────────────────────────────────────────────────────────────────────
def plot_label_distributions(data_root: Path, out_dir: Path):
    """Plot OCEAN label distributions for the training split."""
    ann_file = data_root / "train" / ANNOTATION_FILES["train"]
    if not ann_file.exists():
        logger.warning("Cannot plot labels — annotation file missing")
        return

    with open(ann_file, "rb") as f:
        raw_annotations = pickle.load(f, encoding="latin1")
    annotations = _normalise_annotations(raw_annotations)

    all_labels = []
    for trait_dict in annotations.values():
        try:
            row = [float(trait_dict[t]) for t in TRAIT_KEYS]
            all_labels.append(row)
        except (KeyError, ValueError):
            pass

    labels = np.array(all_labels)   # (N, 5)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle("Training Set — OCEAN Label Distributions", fontsize=14, fontweight="bold")

    for i, (ax, trait) in enumerate(zip(axes, TRAIT_KEYS)):
        ax.hist(labels[:, i], bins=40, color="#2E75B6", edgecolor="white", alpha=0.85)
        ax.set_title(trait.capitalize(), fontsize=11)
        ax.set_xlabel("Score [0, 1]")
        ax.set_ylabel("Count" if i == 0 else "")
        ax.axvline(labels[:, i].mean(), color="red", linestyle="--",
                   linewidth=1.5, label=f"mean={labels[:,i].mean():.2f}")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    plt.tight_layout()
    out_path = out_dir / "label_distributions.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Label distribution plot saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Validate feature cache before training")
    parser.add_argument("--cache_dir", default="./feature_cache")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--out_dir",   default="./diagnostics")
    parser.add_argument("--splits",    nargs="+",
                        default=["train", "val", "test"],
                        choices=["train", "val", "test"])
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    data_root = Path(args.data_root)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_issues = []

    for split in args.splits:
        logger.info(f"\nChecking split: {split}")
        results = check_split(split, data_root, cache_dir)

        logger.info(f"  Cache completeness : {results.get('cache_completeness', 'N/A')}")
        for mod, checks in results.get("modality_checks", {}).items():
            logger.info(
                f"  {mod:8s} — dim_errors:{checks['dim_errors']}  "
                f"nan:{checks['nan_rate']}  zeros:{checks['zero_rate']}  "
                f"std:{checks['mean_std']}"
            )

        if results["issues"]:
            for issue in results["issues"]:
                logger.error(f"  ❌ ISSUE: {issue}")
            all_issues.extend(results["issues"])
        else:
            logger.info(f"  ✅ {split} — all checks passed")

    # ── Label distribution plot ───────────────────────────────────────────────
    plot_label_distributions(data_root, out_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    report_path = out_dir / "cache_report.txt"
    with open(report_path, "w") as f:
        if all_issues:
            f.write("ISSUES FOUND — do not start training:\n")
            for issue in all_issues:
                f.write(f"  - {issue}\n")
        else:
            f.write("ALL CHECKS PASSED — safe to start training.\n")

    print(f"\n{'='*60}")
    if all_issues:
        print(f"❌  {len(all_issues)} issue(s) found. Fix before training.")
        print(f"    See {report_path}")
        sys.exit(1)
    else:
        print("✅  All diagnostics passed. Safe to start Stage 2 training.")
        print(f"    Report saved to {report_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()