"""
training/train_tacfn.py
TACFN Training Script — E13 (Ultimate Model)

Changes from original:
  - Two-phase training added (borrowed from emotion recognition module):
      Phase 1 (warmup_epochs): projection layers frozen, only fusion + regressor train
      Phase 2 (remaining epochs): all layers unfreeze, end-to-end fine-tuning
  - weight_decay = 0.01 (Rule 6 — mandatory)
  - LR = 5e-5 with linear warmup (adaptive gates need gentle early updates)
  - Gradient accumulation for larger effective batch size
  - Early stopping patience = 10 (TACFN converges more slowly than E12)

Run from project root:
    python training/train_tacfn.py ^
        --data_root ./data ^
        --cache_dir ./feature_cache ^
        --out_dir   ./experiments ^
        --exp_name  E13_TACFN_final

Windows: use ^ for line continuation, or put on one line.
"""

import os
import sys
import argparse
import logging
import json
import csv
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset  import get_dataloaders, TRAIT_KEYS
from training.tacfn_model import TACFN, set_seed, count_parameters
from training.losses   import CombinedLoss
from training.metrics  import compute_metrics, format_metrics

# ── Seed FIRST — Rule 1 ───────────────────────────────────────────────────────
set_seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 10):
        self.patience   = patience
        self.best       = None
        self.counter    = 0
        self.best_epoch = 0

    def __call__(self, val_acc: float, epoch: int) -> bool:
        if self.best is None or val_acc > self.best + 1e-4:
            self.best, self.counter, self.best_epoch = val_acc, 0, epoch
            return False
        self.counter += 1
        logger.info(f"  EarlyStopping {self.counter}/{self.patience}")
        return self.counter >= self.patience


# ─────────────────────────────────────────────────────────────────────────────
# LR Warmup
# ─────────────────────────────────────────────────────────────────────────────

def apply_warmup(optimizer, epoch: int, warmup_epochs: int, base_lr: float):
    """Linearly ramp LR from 0 → base_lr over warmup_epochs."""
    if epoch <= warmup_epochs:
        lr = base_lr * epoch / warmup_epochs
        for g in optimizer.param_groups:
            g["lr"] = lr


# ─────────────────────────────────────────────────────────────────────────────
# Two-Phase Training Helpers
# ─────────────────────────────────────────────────────────────────────────────

def freeze_projections(model: TACFN):
    """
    Phase 1: Freeze projection layers so only fusion + regressor train.
    This protects the projection weights from random fusion gradients early on.
    """
    for proj in [model.audio_proj, model.text_proj, model.visual_proj]:
        for param in proj.parameters():
            param.requires_grad = False
    logger.info("Phase 1: Projection layers FROZEN — training fusion + regressor only")


def unfreeze_all(model: TACFN, optimizer: optim.Optimizer, lr: float):
    """
    Phase 2: Unfreeze projection layers with a lower LR than the fusion layers.

    Differential LR rationale:
      - Fusion + regressor have been training since Phase 1 — they are already
        partially converged.  Keep their LR unchanged.
      - Projection layers were frozen the whole of Phase 1 — they have never
        seen a gradient.  Starting them at the same LR as an already-warmed-up
        fusion would cause large, destabilising updates in the first few steps.
      - Using proj_lr = 0.4 × base_lr is the standard "layer-wise LR decay"
        pattern and keeps the newly-unfrozen layers from disrupting the
        converged fusion weights.
    """
    proj_lr = lr * 0.4          # e.g. 5e-5 base → 2e-5 for projections

    frozen_params = []
    for proj in [model.audio_proj, model.text_proj, model.visual_proj]:
        for param in proj.parameters():
            if not param.requires_grad:
                param.requires_grad = True
                frozen_params.append(param)

    if frozen_params:
        optimizer.add_param_group({
            "params":       frozen_params,
            "lr":           proj_lr,
            "weight_decay": 0.01,
        })
        logger.info(
            f"Phase 2: Projection LR = {proj_lr:.2e}  "
            f"(0.4× base {lr:.2e}) — differential LR applied"
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Phase 2: ALL layers UNFROZEN — {trainable:,} trainable parameters")


# ─────────────────────────────────────────────────────────────────────────────
# Train / Eval
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler,
                grad_accum, warmup_epochs, base_lr, epoch):
    model.train()
    apply_warmup(optimizer, epoch, warmup_epochs, base_lr)

    total_loss, all_p, all_t = 0.0, [], []
    optimizer.zero_grad()

    for i, (features, labels) in enumerate(loader):
        a = features["audio"].to(DEVICE)
        t = features["text"].to(DEVICE)
        v = features["visual"].to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            preds = model(a, t, v)
            loss  = criterion(preds, labels) / grad_accum

        scaler.scale(loss).backward()

        if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        all_p.append(preds.detach().cpu().numpy())
        all_t.append(labels.cpu().numpy())

    m = compute_metrics(np.vstack(all_p), np.vstack(all_t))
    m["loss"] = total_loss / len(loader)
    return m


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, all_p, all_t = 0.0, [], []

    for features, labels in loader:
        a = features["audio"].to(DEVICE)
        t = features["text"].to(DEVICE)
        v = features["visual"].to(DEVICE)
        labels = labels.to(DEVICE)

        preds = model(a, t, v)
        total_loss += criterion(preds, labels).item()
        all_p.append(preds.cpu().numpy())
        all_t.append(labels.cpu().numpy())

    m = compute_metrics(np.vstack(all_p), np.vstack(all_t))
    m["loss"] = total_loss / len(loader)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# CSV Logging
# ─────────────────────────────────────────────────────────────────────────────

def log_to_csv(csv_path: str, row: dict):
    fields = [
        "exp_name", "model", "modalities", "dropout", "proj_dim", "lr",
        "num_layers", "warmup_epochs", "grad_accum",
        "train_acc", "val_acc", "test_acc",
        "O_mae", "C_mae", "E_mae", "A_mae", "N_mae",
        "best_epoch", "timestamp",
    ]
    exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)
    logger.info(f"✅ Appended to {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    logger.info(f"\n{'='*65}")
    logger.info(f"Experiment  : {args.exp_name}")
    logger.info(f"Model       : TACFN (Liu et al. 2025)")
    logger.info(f"LR          : {args.lr} (warmup {args.warmup_epochs} epochs)")
    logger.info(f"Grad accum  : {args.grad_accum}  (effective batch = {args.batch_size * args.grad_accum})")
    logger.info(f"Weight dec  : 0.01  (Rule 6)")
    logger.info(f"Two-phase   : Phase 1 = {args.warmup_epochs} epochs (projections frozen)")
    logger.info(f"Device      : {DEVICE}")
    logger.info(f"{'='*65}\n")

    out_dir = Path(args.out_dir) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {**vars(args), "seed": 42, "device": DEVICE,
              "model": "TACFN", "timestamp": datetime.now().isoformat()}
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_root, args.cache_dir, args.batch_size, args.num_workers
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TACFN(
        proj_dim=args.proj_dim, num_heads=args.num_heads, ff_dim=args.ff_dim,
        num_layers=args.num_layers, dropout=args.dropout,
    ).to(DEVICE)
    logger.info(f"TACFN parameters: {count_parameters(model):,}")

    # ── Phase 1: freeze projections ───────────────────────────────────────────
    freeze_projections(model)
    trainable_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Phase 1 trainable parameters: {trainable_p1:,}")

    # ── Optimiser / Scheduler — Rule 6: weight_decay = 0.01 ──────────────────
    criterion  = CombinedLoss(alpha=0.5)
    optimizer  = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01
    )
    scheduler  = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                   patience=3)
    scaler     = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    early_stop = EarlyStopping(patience=args.patience)

    best_val_acc = 0.0
    history      = {"train": [], "val": []}
    phase        = 1

    for epoch in range(1, args.epochs + 1):

        # ── Switch to Phase 2 after warmup ────────────────────────────────────
        if phase == 1 and epoch > args.warmup_epochs:
            phase = 2
            unfreeze_all(model, optimizer, args.lr)
            logger.info(f"\n--- Switching to Phase 2 at epoch {epoch} ---\n")

        tr = train_epoch(model, train_loader, optimizer, criterion, scaler,
                         args.grad_accum, args.warmup_epochs, args.lr, epoch)
        vl = eval_epoch(model, val_loader, criterion)

        # Only step scheduler after warmup
        if epoch > args.warmup_epochs:
            scheduler.step(vl["mean_accuracy"])

        phase_tag = f"[P{phase}]"
        logger.info(
            f"[{epoch:3d}]{phase_tag} "
            f"Train:{tr['mean_accuracy']:.4f}  Val:{vl['mean_accuracy']:.4f}  "
            f"|  {format_metrics(vl)}"
        )

        history["train"].append(tr)
        history["val"].append(vl)

        if vl["mean_accuracy"] > best_val_acc:
            best_val_acc = vl["mean_accuracy"]
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics":      vl,
                "config":           config,
            }, out_dir / "best_model.pt")
            logger.info(f"  ✅ Best saved (val_acc={best_val_acc:.4f})")

        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if early_stop(vl["mean_accuracy"], epoch):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # ── Test evaluation — Rule 3: once, on best checkpoint ───────────────────
    ckpt = torch.load(out_dir / "best_model.pt", map_location=DEVICE,
                      weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m = eval_epoch(model, test_loader, criterion)

    logger.info(f"\n{'='*65}")
    logger.info(f"TACFN FINAL TEST — {args.exp_name}")
    logger.info(f"Mean Accuracy : {test_m['mean_accuracy']:.4f}")
    logger.info(f"Per-trait MAE : {format_metrics(test_m)}")
    logger.info(f"Best epoch    : {ckpt['epoch']}")
    logger.info(f"{'='*65}")

    with open(out_dir / "test_results.json", "w") as f:
        json.dump(test_m, f, indent=2)

    best_epoch = ckpt["epoch"]
    log_to_csv(str(Path(args.out_dir) / "global_results.csv"), {
        "exp_name":      args.exp_name,
        "model":         "TACFN",
        "modalities":    "audio+text+visual",
        "dropout":       args.dropout,
        "proj_dim":      args.proj_dim,
        "lr":            args.lr,
        "num_layers":    args.num_layers,
        "warmup_epochs": args.warmup_epochs,
        "grad_accum":    args.grad_accum,
        "train_acc":     round(history["train"][best_epoch - 1]["mean_accuracy"], 4),
        "val_acc":       round(best_val_acc, 4),
        "test_acc":      round(test_m["mean_accuracy"], 4),
        "O_mae":         round(test_m["openness"], 4),
        "C_mae":         round(test_m["conscientiousness"], 4),
        "E_mae":         round(test_m["extraversion"], 4),
        "A_mae":         round(test_m["agreeableness"], 4),
        "N_mae":         round(test_m["neuroticism"], 4),
        "best_epoch":    best_epoch,
        "timestamp":     datetime.now().isoformat(),
    })

    return test_m


# ─────────────────────────────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train TACFN — E13 final model")
    p.add_argument("--data_root",     required=True)
    p.add_argument("--cache_dir",     default="./feature_cache")
    p.add_argument("--out_dir",       default="./experiments")
    p.add_argument("--exp_name",      required=True)
    p.add_argument("--proj_dim",      type=int,   default=256)
    p.add_argument("--num_heads",     type=int,   default=8)
    p.add_argument("--ff_dim",        type=int,   default=512)
    p.add_argument("--num_layers",    type=int,   default=3)
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--lr",            type=float, default=5e-5)
    p.add_argument("--epochs",        type=int,   default=60)
    p.add_argument("--patience",      type=int,   default=10)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--num_workers",   type=int,   default=0)
    p.add_argument("--warmup_epochs", type=int,   default=5)
    p.add_argument("--grad_accum",    type=int,   default=2)
    train(p.parse_args())