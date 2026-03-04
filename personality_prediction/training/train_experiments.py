"""
training/train_experiments.py
Main training script for ALL experiments E2–E12 and hyperparameter sweeps H1–H4.

One script, all architectures — controlled entirely by command-line flags.
See Section 2 of the training guide for every run command.

Run from project root:
    python training/train_experiments.py ^
        --data_root ./data --cache_dir ./feature_cache --out_dir ./experiments ^
        --exp_name E2_text_only --modalities text --fusion none ^
        --dropout 0.3 --proj_dim 256 --lr 1e-4 --epochs 50 --patience 7 --batch_size 32

Windows: use ^ for line continuation, or put the whole command on one line.
"""

import os
import sys
import json
import csv
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import get_dataloaders, TRAIT_KEYS
from training.losses  import CombinedLoss
from training.metrics import compute_metrics, format_metrics

# ── Seed first — Rule 1 ───────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger  = logging.getLogger(__name__)
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURES
# Each is controlled by --fusion flag.
# All apply F.normalize(p=2) as first op in forward() — Rule 2.
# Exception: UnimodalMLP with no_l2_norm=True (only for E8 ablation).
# ─────────────────────────────────────────────────────────────────────────────

class UnimodalMLP(nn.Module):
    """
    Single-modality MLP head.
    Used for E2 (text), E3 (audio), E4 (visual).
    Architecture: L2norm → Linear(dim→proj) → LN → GELU → Dropout
                  → Linear(proj→64) → GELU → Linear(64→5) → Sigmoid
    """
    def __init__(self, in_dim: int, proj_dim: int = 256,
                 dropout: float = 0.3, use_l2_norm: bool = True):
        super().__init__()
        self.use_l2_norm = use_l2_norm
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 64),
            nn.GELU(),
            nn.Linear(64, 5),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_l2_norm:
            x = F.normalize(x, p=2, dim=-1)
        return self.net(x)


class ConcatFusion(nn.Module):
    """
    Early concatenation of 2 or 3 modalities.
    Used for E5, E6, E7 (pairwise) and E8, E9 (trimodal).
    E8: no_l2_norm=True  (ablation)
    E9: no_l2_norm=False (correct)
    """
    def __init__(self, modality_dims: List[int], proj_dim: int = 256,
                 dropout: float = 0.3, use_l2_norm: bool = True):
        super().__init__()
        self.use_l2_norm = use_l2_norm
        n = len(modality_dims)

        # One projection per modality
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ) for d in modality_dims
        ])

        # MLP head on concatenated features
        self.head = nn.Sequential(
            nn.Linear(proj_dim * n, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 64),
            nn.GELU(),
            nn.Linear(64, 5),
            nn.Sigmoid(),
        )

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        projected = []
        for x, proj in zip(inputs, self.projections):
            if self.use_l2_norm:
                x = F.normalize(x, p=2, dim=-1)
            projected.append(proj(x))
        fused = torch.cat(projected, dim=-1)
        return self.head(fused)


class CrossAttentionFusion(nn.Module):
    """
    6 directed cross-attention pairs, no transformer encoder.
    Used for E11.
    """
    def __init__(self, proj_dim: int = 256, num_heads: int = 8,
                 dropout: float = 0.3):
        super().__init__()
        D = proj_dim

        self.audio_proj  = self._proj(768,  D, dropout)
        self.text_proj   = self._proj(768,  D, dropout)
        self.visual_proj = self._proj(2048, D, dropout)

        # 6 directed cross-attention pairs
        def ca():
            return nn.MultiheadAttention(D, num_heads, dropout=dropout, batch_first=True)

        self.a_from_t = ca(); self.a_from_v = ca()
        self.t_from_a = ca(); self.t_from_v = ca()
        self.v_from_a = ca(); self.v_from_t = ca()

        self.norm_a = nn.LayerNorm(D)
        self.norm_t = nn.LayerNorm(D)
        self.norm_v = nn.LayerNorm(D)

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

        fused = torch.cat([a_out, t_out, v_out], dim=-1)
        return self.head(fused)


class TransformerFusion(nn.Module):
    """
    6 cross-attention pairs + TransformerEncoder over 3 modality tokens.
    Used for E12 and all H1–H4 sweeps.
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

        tokens = torch.stack([a_out, t_out, v_out], dim=1)   # (B, 3, D)
        tokens = self.transformer(tokens)                      # (B, 3, D)
        fused  = tokens.reshape(tokens.size(0), -1)           # (B, 3D)
        return self.head(fused)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────

DIM_MAP = {"audio": 768, "text": 768, "visual": 2048}


def build_model(args) -> nn.Module:
    """Build the correct model from command-line args."""
    mods   = args.modalities
    fusion = args.fusion
    D      = args.proj_dim
    drop   = args.dropout
    use_l2 = not args.no_l2_norm

    if fusion == "none":
        assert len(mods) == 1, "--fusion none requires exactly one modality"
        return UnimodalMLP(DIM_MAP[mods[0]], proj_dim=D, dropout=drop,
                           use_l2_norm=use_l2)

    if fusion == "concat":
        dims = [DIM_MAP[m] for m in mods]
        return ConcatFusion(dims, proj_dim=D, dropout=drop, use_l2_norm=use_l2)

    if fusion == "late":
        # Late fusion: loads E2/E3/E4 checkpoints — no new model needed
        return None

    if fusion == "cross_attention":
        assert len(mods) == 3, "--fusion cross_attention requires all 3 modalities"
        return CrossAttentionFusion(proj_dim=D, num_heads=args.num_heads, dropout=drop)

    if fusion == "transformer":
        assert len(mods) == 3, "--fusion transformer requires all 3 modalities"
        return TransformerFusion(proj_dim=D, num_heads=args.num_heads,
                                 ff_dim=args.ff_dim, num_layers=args.num_layers,
                                 dropout=drop)

    raise ValueError(f"Unknown fusion: {fusion}")


# ─────────────────────────────────────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 7):
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
# BATCH HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_inputs(features: dict, modalities: List[str]) -> List[torch.Tensor]:
    """Extract the correct modality tensors from a batch features dict."""
    return [features[m].to(DEVICE) for m in modalities]


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL LOOPS
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler, modalities):
    model.train()
    total_loss, all_p, all_t = 0.0, [], []

    for features, labels in loader:
        inputs = get_inputs(features, modalities)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            preds = model(*inputs)
            loss  = criterion(preds, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_p.append(preds.detach().cpu().numpy())
        all_t.append(labels.cpu().numpy())

    m = compute_metrics(np.vstack(all_p), np.vstack(all_t))
    m["loss"] = total_loss / len(loader)
    return m


@torch.no_grad()
def eval_epoch(model, loader, criterion, modalities):
    model.eval()
    total_loss, all_p, all_t = 0.0, [], []

    for features, labels in loader:
        inputs = get_inputs(features, modalities)
        labels = labels.to(DEVICE)
        preds  = model(*inputs)
        total_loss += criterion(preds, labels).item()
        all_p.append(preds.cpu().numpy())
        all_t.append(labels.cpu().numpy())

    m = compute_metrics(np.vstack(all_p), np.vstack(all_t))
    m["loss"] = total_loss / len(loader)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# LATE FUSION (E10)
# Loads checkpoints from E2, E3, E4 and averages predictions.
# No new training — inference only.
# ─────────────────────────────────────────────────────────────────────────────

def run_late_fusion(args, train_loader, val_loader, test_loader, out_dir):
    logger.info("Late fusion: loading E2, E3, E4 checkpoints")

    exp_dir  = Path(args.out_dir)
    ckpt_map = {
        "text":   exp_dir / "E2_text_only"   / "best_model.pt",
        "audio":  exp_dir / "E3_audio_only"  / "best_model.pt",
        "visual": exp_dir / "E4_visual_only" / "best_model.pt",
    }

    models = {}
    for mod, ckpt_path in ckpt_map.items():
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Late fusion needs {ckpt_path}. "
                f"Run E2, E3, E4 first."
            )
        ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        dim   = DIM_MAP[mod]
        m     = UnimodalMLP(dim, proj_dim=args.proj_dim, dropout=args.dropout)
        m.load_state_dict(ckpt["model_state_dict"])
        m.to(DEVICE).eval()
        models[mod] = m
        logger.info(f"  Loaded {mod} model from {ckpt_path}")

    criterion = CombinedLoss()

    @torch.no_grad()
    def eval_late(loader):
        all_p, all_t, total_loss = [], [], 0.0
        for features, labels in loader:
            labels = labels.to(DEVICE)
            preds_list = []
            for mod, m in models.items():
                x = features[mod].to(DEVICE)
                preds_list.append(m(x))
            preds = torch.stack(preds_list).mean(dim=0)
            total_loss += criterion(preds, labels).item()
            all_p.append(preds.cpu().numpy())
            all_t.append(labels.cpu().numpy())
        met = compute_metrics(np.vstack(all_p), np.vstack(all_t))
        met["loss"] = total_loss / len(loader)
        return met

    val_m  = eval_late(val_loader)
    test_m = eval_late(test_loader)
    logger.info(f"Val  : {format_metrics(val_m)}")
    logger.info(f"Test : {format_metrics(test_m)}")

    with open(out_dir / "test_results.json", "w") as f:
        json.dump(test_m, f, indent=2)

    return val_m, test_m


# ─────────────────────────────────────────────────────────────────────────────
# CSV LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def log_to_csv(csv_path: str, row: dict):
    fields = [
        "exp_name", "modalities", "fusion", "no_l2_norm",
        "dropout", "proj_dim", "num_layers", "lr",
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
    logger.info(f"Result appended to {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    logger.info(f"\n{'='*65}")
    logger.info(f"Experiment  : {args.exp_name}")
    logger.info(f"Modalities  : {args.modalities}")
    logger.info(f"Fusion      : {args.fusion}")
    logger.info(f"L2 norm     : {not args.no_l2_norm}")
    logger.info(f"Device      : {DEVICE}")
    logger.info(f"{'='*65}\n")

    out_dir = Path(args.out_dir) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {**vars(args), "seed": 42, "device": DEVICE,
              "timestamp": datetime.now().isoformat()}
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_root, args.cache_dir, args.batch_size, args.num_workers
    )

    # ── Late fusion (no training needed) ──────────────────────────────────────
    if args.fusion == "late":
        val_m, test_m = run_late_fusion(
            args, train_loader, val_loader, test_loader, out_dir
        )
        log_to_csv(str(Path(args.out_dir) / "global_results.csv"), {
            "exp_name":   args.exp_name,
            "modalities": "+".join(args.modalities),
            "fusion":     args.fusion,
            "no_l2_norm": args.no_l2_norm,
            "dropout":    args.dropout,
            "proj_dim":   args.proj_dim,
            "num_layers": args.num_layers,
            "lr":         args.lr,
            "train_acc":  "",
            "val_acc":    round(val_m["mean_accuracy"], 4),
            "test_acc":   round(test_m["mean_accuracy"], 4),
            "O_mae":      round(test_m["openness"], 4),
            "C_mae":      round(test_m["conscientiousness"], 4),
            "E_mae":      round(test_m["extraversion"], 4),
            "A_mae":      round(test_m["agreeableness"], 4),
            "N_mae":      round(test_m["neuroticism"], 4),
            "best_epoch": "N/A",
            "timestamp":  datetime.now().isoformat(),
        })
        return test_m

    # ── Model, optimiser, loss ────────────────────────────────────────────────
    model     = build_model(args).to(DEVICE)
    criterion = CombinedLoss(alpha=0.5)
    # Rule 6: weight_decay = 0.01
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                  patience=3, verbose=True)
    scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    stopper   = EarlyStopping(patience=args.patience)

    best_val_acc = 0.0
    best_epoch   = 0
    history      = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, criterion,
                         scaler, args.modalities)
        vl = eval_epoch(model, val_loader, criterion, args.modalities)
        scheduler.step(vl["mean_accuracy"])

        logger.info(
            f"[{epoch:3d}] Train:{tr['mean_accuracy']:.4f}  "
            f"Val:{vl['mean_accuracy']:.4f}  |  {format_metrics(vl)}"
        )

        history["train"].append(tr)
        history["val"].append(vl)

        if vl["mean_accuracy"] > best_val_acc:
            best_val_acc = vl["mean_accuracy"]
            best_epoch   = epoch
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics":      vl,
                "config":           config,
            }, out_dir / "best_model.pt")
            logger.info(f"  ✅ Best saved (val_acc={best_val_acc:.4f})")

        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if stopper(vl["mean_accuracy"], epoch):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # ── Test evaluation (Rule 3: once, on best checkpoint) ───────────────────
    ckpt = torch.load(out_dir / "best_model.pt", map_location=DEVICE,
                      weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m = eval_epoch(model, test_loader, criterion, args.modalities)

    logger.info(f"\n{'='*65}")
    logger.info(f"FINAL TEST — {args.exp_name}")
    logger.info(f"Mean Accuracy : {test_m['mean_accuracy']:.4f}")
    logger.info(f"Best epoch    : {best_epoch}")
    logger.info(f"{'='*65}")

    with open(out_dir / "test_results.json", "w") as f:
        json.dump(test_m, f, indent=2)

    log_to_csv(str(Path(args.out_dir) / "global_results.csv"), {
        "exp_name":   args.exp_name,
        "modalities": "+".join(args.modalities),
        "fusion":     args.fusion,
        "no_l2_norm": args.no_l2_norm,
        "dropout":    args.dropout,
        "proj_dim":   args.proj_dim,
        "num_layers": args.num_layers,
        "lr":         args.lr,
        "train_acc":  round(history["train"][best_epoch - 1]["mean_accuracy"], 4),
        "val_acc":    round(best_val_acc, 4),
        "test_acc":   round(test_m["mean_accuracy"], 4),
        "O_mae":      round(test_m["openness"], 4),
        "C_mae":      round(test_m["conscientiousness"], 4),
        "E_mae":      round(test_m["extraversion"], 4),
        "A_mae":      round(test_m["agreeableness"], 4),
        "N_mae":      round(test_m["neuroticism"], 4),
        "best_epoch": best_epoch,
        "timestamp":  datetime.now().isoformat(),
    })

    return test_m


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train personality prediction experiments E2–E12")
    p.add_argument("--data_root",   required=True)
    p.add_argument("--cache_dir",   default="./feature_cache")
    p.add_argument("--out_dir",     default="./experiments")
    p.add_argument("--exp_name",    required=True)
    p.add_argument("--modalities",  nargs="+", required=True,
                   choices=["audio", "text", "visual"])
    p.add_argument("--fusion",      required=True,
                   choices=["none", "concat", "late", "cross_attention", "transformer"])
    p.add_argument("--no_l2_norm",  action="store_true",
                   help="Disable L2 norm — only for E8 ablation")
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--proj_dim",    type=int,   default=256)
    p.add_argument("--num_layers",  type=int,   default=3)
    p.add_argument("--num_heads",   type=int,   default=8)
    p.add_argument("--ff_dim",      type=int,   default=512)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--patience",    type=int,   default=7)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--num_workers", type=int,   default=0)
    train(p.parse_args())