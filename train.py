import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

from dataset import get_loaders
from model import build_model

# ── single swap point for Kaggle ─────────────────────────────────────────────
DATA_ROOT = Path("..")                              # Kaggle: Path("/kaggle/input/isic-2019")
IMAGE_DIR = DATA_ROOT / "ISIC_2019_Training_Input"
CSV_PATH  = DATA_ROOT / "ISIC_2019_Training_GroundTruth.csv"
CKPT_DIR  = Path("checkpoints")

# ── hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE   = 64      # split across 2 GPUs by DataParallel → 32 per card
NUM_EPOCHS   = 20
LR           = 3e-4
WEIGHT_DECAY = 1e-2
VAL_SPLIT    = 0.15
NUM_WORKERS  = 4
SEED         = 42


# ── helpers ───────────────────────────────────────────────────────────────────

def compute_pos_weight(csv_path: Path) -> torch.Tensor:
    """pos_weight = n_benign / n_malignant for BCEWithLogitsLoss."""
    df    = pd.read_csv(csv_path)
    df    = df[df["UNK"] != 1.0]
    n_pos = int(df[["MEL", "BCC", "AK", "SCC"]].any(axis=1).sum())
    n_neg = len(df) - n_pos
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)


def train_one_epoch(
    model, loader, optimizer, scaler, criterion, device, use_amp: bool
) -> float:
    model.train()
    total_loss = 0.0

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp: bool) -> tuple[float, float, float]:
    model.eval()
    total_loss             = 0.0
    all_logits: list[float] = []
    all_labels: list[int]   = []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.float32, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        all_logits.extend(logits.cpu().float().tolist())
        all_labels.extend(labels.cpu().int().tolist())

    logits_arr = np.array(all_logits, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int32)

    n_bad = int((~np.isfinite(logits_arr)).sum())
    if n_bad:
        print(f"  Warning: {n_bad} NaN/Inf logits - model weights may be corrupted")
        logits_arr = np.nan_to_num(logits_arr, nan=0.0, posinf=88.0, neginf=-88.0)

    probs   = 1.0 / (1.0 + np.exp(-np.clip(logits_arr, -88.0, 88.0)))
    preds   = (probs >= 0.5).astype(np.int32)
    auroc   = roc_auc_score(labels_arr, probs)
    bal_acc = balanced_accuracy_score(labels_arr, preds)
    return total_loss / len(loader.dataset), auroc, bal_acc


def save_checkpoint(model, optimizer, epoch: int, auroc: float, path: Path) -> None:
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "val_auroc":            auroc,
        },
        path,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke_test", action="store_true",
                   help="Quick sanity run: batch=4, 2000 train / 500 val samples, 1 epoch")
    return p.parse_args()


def _trim_loader(loader: DataLoader, n: int, batch_size: int, *, shuffle: bool) -> DataLoader:
    """Return a new DataLoader capped at n samples, reusing the existing dataset."""
    subset = Subset(loader.dataset, range(min(n, len(loader.dataset))))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        persistent_workers=loader.num_workers > 0,
    )


def main() -> None:
    args  = _parse_args()
    smoke = args.smoke_test

    batch_size = 4          if smoke else BATCH_SIZE
    num_epochs = 1          if smoke else NUM_EPOCHS

    torch.manual_seed(SEED)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Device: {device}  |  GPUs visible: {n_gpus}")
    if smoke:
        print("[smoke_test] batch=4 | train<=2000 | val<=500 | 1 epoch | img=288x288")

    # ── data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_loaders(
        IMAGE_DIR,
        CSV_PATH,
        val_split=VAL_SPLIT,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        seed=SEED,
    )
    if smoke:
        train_loader = _trim_loader(train_loader, 2000, batch_size, shuffle=True)
        val_loader   = _trim_loader(val_loader,    500, batch_size, shuffle=False)

    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    print(f"Train: {n_train} samples  |  Val: {n_val} samples")

    # ── loss with class weighting ──────────────────────────────────────────────
    pos_weight = compute_pos_weight(CSV_PATH).to(device)
    print(f"pos_weight (benign/malignant): {pos_weight.item():.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(pretrained=True).to(device)
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Wrapped in DataParallel across {n_gpus} GPUs")

    # ── optimiser + schedule + AMP ────────────────────────────────────────────
    # AMP disabled in smoke mode: small batch (4) + fp16 causes BN variance
    # instability on Pascal GPUs; fp32 is fine for a pipeline correctness check.
    use_amp = not smoke
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── training loop ─────────────────────────────────────────────────────────
    best_auroc = 0.0
    header = f"{'Epoch':>5}  {'TrainLoss':>9}  {'ValLoss':>7}  {'AUROC':>6}  {'BalAcc':>6}  {'LR':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    if smoke:
        _, pre_auroc, pre_bal_acc = evaluate(model, val_loader, criterion, device, use_amp)
        print(f"{'pre':>5}  {'---':>9}  {'---':>7}  {pre_auroc:>6.4f}  {pre_bal_acc:>6.4f}  {'---':>8}  <- pretrained baseline")

    for epoch in range(1, num_epochs + 1):
        train_loss                = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, use_amp)
        val_loss, auroc, bal_acc  = evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"{epoch:>5}  {train_loss:>9.4f}  {val_loss:>7.4f}"
            f"  {auroc:>6.4f}  {bal_acc:>6.4f}  {current_lr:>8.2e}"
        )

        if auroc > best_auroc:
            best_auroc = auroc
            save_checkpoint(model, optimizer, epoch, auroc, CKPT_DIR / "best.pt")
            print(f"        ^ new best -> checkpoint saved (AUROC {auroc:.4f})")

    print(f"\nDone. Best val AUROC: {best_auroc:.4f}  Checkpoint: {CKPT_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
