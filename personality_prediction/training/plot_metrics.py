"""
training/plot_metrics.py
Generates all 6 required plots for the professor presentation.

Run from project root after E13 completes:
    python training/plot_metrics.py ^
        --history     ./experiments/E13_TACFN_final/history.json ^
        --test        ./experiments/E13_TACFN_final/test_results.json ^
        --results_csv ./experiments/global_results.csv ^
        --out_dir     ./plots
"""

import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive — works on Windows without display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRAIT_KEYS  = ["openness","conscientiousness","extraversion","agreeableness","neuroticism"]
TRAIT_SHORT = ["O", "C", "E", "A", "N"]

BLUE   = "#2E75B6"
ORANGE = "#E07B39"
GREEN  = "#2E8B57"
RED    = "#C0392B"
PURPLE = "#6C3483"
GREY   = "#95A5A6"
COLORS = [BLUE, ORANGE, GREEN, RED, PURPLE, GREY, "#1ABC9C", "#F39C12"]


def save(fig, path: Path, name: str):
    out = path / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Training curves (loss + accuracy vs epoch for E13)
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history: dict, out_dir: Path):
    train_acc  = [e["mean_accuracy"] for e in history["train"]]
    val_acc    = [e["mean_accuracy"] for e in history["val"]]
    train_loss = [e["loss"]          for e in history["train"]]
    val_loss   = [e["loss"]          for e in history["val"]]
    epochs     = list(range(1, len(train_acc) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("TACFN (E13) — Training Curves", fontsize=14, fontweight="bold")

    ax1.plot(epochs, train_acc, color=BLUE,   label="Train", linewidth=2)
    ax1.plot(epochs, val_acc,   color=ORANGE, label="Val",   linewidth=2)
    best_epoch = int(np.argmax(val_acc)) + 1
    ax1.axvline(best_epoch, color=RED, linestyle="--", linewidth=1.2,
                label=f"Best epoch {best_epoch}")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Mean Accuracy (1−MAE)")
    ax1.set_title("Accuracy"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_loss, color=BLUE,   label="Train", linewidth=2)
    ax2.plot(epochs, val_loss,   color=ORANGE, label="Val",   linewidth=2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.set_title("Loss"); ax2.legend(); ax2.grid(alpha=0.3)

    save(fig, out_dir, "training_curves_E13.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Modality contribution bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_modality_contribution(results: pd.DataFrame, out_dir: Path):
    ids    = ["E2", "E3", "E4", "E5", "E6", "E7", "E9"]
    labels = ["Text\nonly", "Audio\nonly", "Visual\nonly",
              "Audio\n+Text", "Audio\n+Visual", "Text\n+Visual",
              "All 3\n(concat)"]

    accs = []
    for eid in ids:
        row = results[results["exp_name"].str.startswith(eid)]
        accs.append(float(row["test_acc"].values[0]) if len(row) else 0.0)

    colors = [BLUE, ORANGE, GREEN, PURPLE, RED, "#1ABC9C", GREY]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, accs, color=colors, edgecolor="white", width=0.6)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Mean Accuracy (1−MAE)", fontsize=12)
    ax.set_title("Modality Contribution — Unimodal vs Pairwise vs Trimodal", fontsize=13, fontweight="bold")
    ax.set_ylim(min(accs) - 0.02, max(accs) + 0.02)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(accs[0], color=BLUE, linestyle=":", alpha=0.5, linewidth=1)

    save(fig, out_dir, "modality_contribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Normalisation ablation (E8 vs E9)
# ─────────────────────────────────────────────────────────────────────────────
def plot_norm_ablation(results: pd.DataFrame, out_dir: Path):
    def get_acc(prefix):
        row = results[results["exp_name"].str.startswith(prefix)]
        return float(row["test_acc"].values[0]) if len(row) else 0.0

    e8 = get_acc("E8"); e9 = get_acc("E9")
    delta = (e9 - e8) * 100

    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.bar(["E8\n(No L2 Norm)", "E9\n(With L2 Norm)"],
                  [e8, e9], color=[RED, GREEN], edgecolor="white", width=0.5)

    for bar, acc in zip(bars, [e8, e9]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Mean Accuracy (1−MAE)", fontsize=12)
    ax.set_title(f"L2 Normalisation Ablation\nImprovement: +{delta:.2f}%", fontsize=13, fontweight="bold")
    ax.set_ylim(min(e8, e9) - 0.03, max(e8, e9) + 0.03)
    ax.grid(axis="y", alpha=0.3)
    ax.annotate(f"+{delta:.2f}%", xy=(1, e9), xytext=(0.5, (e8 + e9) / 2),
                arrowprops=dict(arrowstyle="->", color="black"), fontsize=12, color=GREEN)

    save(fig, out_dir, "normalization_ablation.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Fusion method comparison (E9, E10, E11, E12, E13)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fusion_comparison(results: pd.DataFrame, out_dir: Path):
    ids    = ["E9", "E10", "E11", "E12", "E13"]
    labels = ["Concat\n(E9)", "Late\n(E10)", "Cross-Attn\n(E11)",
              "Transformer\n(E12)", "TACFN\n(E13)"]
    colors = [GREY, ORANGE, GREEN, BLUE, PURPLE]

    accs = []
    for eid in ids:
        row = results[results["exp_name"].str.startswith(eid)]
        accs.append(float(row["test_acc"].values[0]) if len(row) else 0.0)

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, accs, color=colors, edgecolor="white", width=0.6)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Mean Accuracy (1−MAE)", fontsize=12)
    ax.set_title("Fusion Method Comparison — All Trimodal Architectures", fontsize=13, fontweight="bold")
    ax.set_ylim(min(a for a in accs if a > 0) - 0.02, max(accs) + 0.02)
    ax.grid(axis="y", alpha=0.3)

    save(fig, out_dir, "fusion_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — OCEAN radar chart (E13 per-trait accuracy)
# ─────────────────────────────────────────────────────────────────────────────
def plot_ocean_radar(test_results: dict, out_dir: Path):
    accs = [1.0 - test_results[t] for t in TRAIT_KEYS]
    labels = [t.capitalize() for t in TRAIT_KEYS]

    N      = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    accs   += accs[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    ax.plot(angles, accs, color=BLUE, linewidth=2.5)
    ax.fill(angles, accs, color=BLUE, alpha=0.25)

    ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], labels, fontsize=12)
    ax.set_ylim(0.8, 1.0)
    ax.set_yticks([0.84, 0.88, 0.92, 0.96, 1.0])
    ax.set_yticklabels(["0.84", "0.88", "0.92", "0.96", "1.00"], fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("TACFN (E13) — Per-Trait Accuracy\n(OCEAN Radar)", fontsize=13,
                 fontweight="bold", pad=20)

    mean_acc = float(np.mean(accs[:-1]))
    ax.text(0, 0.7, f"Mean\n{mean_acc:.3f}", ha="center", va="center",
            fontsize=12, fontweight="bold", color=BLUE)

    save(fig, out_dir, "ocean_radar_E13.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 — Dropout sensitivity (H1 sweep)
# ─────────────────────────────────────────────────────────────────────────────
def plot_dropout_sensitivity(results: pd.DataFrame, out_dir: Path):
    h1_rows = results[results["exp_name"].str.startswith("H1_")].copy()

    if len(h1_rows) == 0:
        logger.warning("No H1 rows in global_results.csv — skipping dropout plot")
        return

    h1_rows = h1_rows.sort_values("dropout")
    dropouts = h1_rows["dropout"].astype(float).tolist()
    accs     = h1_rows["test_acc"].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dropouts, accs, marker="o", color=BLUE, linewidth=2.5,
            markersize=8, markerfacecolor="white", markeredgewidth=2)

    best_idx = int(np.argmax(accs))
    ax.scatter([dropouts[best_idx]], [accs[best_idx]], color=RED, s=120,
               zorder=5, label=f"Best: dropout={dropouts[best_idx]}, acc={accs[best_idx]:.4f}")

    for d, a in zip(dropouts, accs):
        ax.text(d, a + 0.001, f"{a:.3f}", ha="center", fontsize=9)

    ax.set_xlabel("Dropout Rate", fontsize=12)
    ax.set_ylabel("Test Mean Accuracy", fontsize=12)
    ax.set_title("Dropout Rate Sensitivity (H1 Sweep)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    save(fig, out_dir, "dropout_sensitivity.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Generate all 6 required plots")
    p.add_argument("--history",     required=True, help="Path to E13 history.json")
    p.add_argument("--test",        required=True, help="Path to E13 test_results.json")
    p.add_argument("--results_csv", required=True, help="Path to global_results.csv")
    p.add_argument("--out_dir",     default="./plots")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.history)  as f: history      = json.load(f)
    with open(args.test)     as f: test_results = json.load(f)
    results = pd.read_csv(args.results_csv)

    logger.info("Generating Plot 1 — Training curves")
    plot_training_curves(history, out_dir)

    logger.info("Generating Plot 2 — Modality contribution")
    plot_modality_contribution(results, out_dir)

    logger.info("Generating Plot 3 — Normalisation ablation")
    plot_norm_ablation(results, out_dir)

    logger.info("Generating Plot 4 — Fusion comparison")
    plot_fusion_comparison(results, out_dir)

    logger.info("Generating Plot 5 — OCEAN radar")
    plot_ocean_radar(test_results, out_dir)

    logger.info("Generating Plot 6 — Dropout sensitivity")
    plot_dropout_sensitivity(results, out_dir)

    print(f"\n✅  All plots saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()