"""Zero-shot evaluation of the ISIC 2019 checkpoint on external test sets.

Loads checkpoints/best.pt and runs inference on ISIC 2016 (heldout of 2019),
ISIC 2020, and ISIC 2024. Reports AUROC and balanced accuracy for each; for
ISIC 2024 additionally reports partial AUROC above 80% TPR — the Kaggle
competition metric of record given the ~0.1% positive rate.

Missing datasets are skipped, not fatal, so the same script works on Kaggle
(where 2020 is attached) and locally (where it isn't).
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from eval_datasets import (
    ISIC2020_CSV,
    ISIC2020_IMG_DIR,
    ISIC2024_CSV,
    ISIC2024_IMG_DIR,
    ISIC2020Dataset,
    ISIC2024Dataset,
    get_eval_loader,
    load_isic2016_heldout,
)
from model import build_model
from setup_data import IS_KAGGLE

BATCH_SIZE  = 128
NUM_WORKERS = 4

CKPT_PATH = Path("/kaggle/working/checkpoints/best.pt") if IS_KAGGLE else Path("checkpoints/best.pt")
OUT_PATH  = Path("/kaggle/working/eval_results.json")    if IS_KAGGLE else Path("eval_results.json")


def load_checkpoint(ckpt_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    cleaned = OrderedDict()
    for k, v in state_dict.items():
        cleaned[k[len("module."):] if k.startswith("module.") else k] = v

    model = build_model(pretrained=False).to(device)
    model.load_state_dict(cleaned)
    model.eval()
    return model


@torch.no_grad()
def predict(model, loader, device, use_amp: bool) -> tuple[np.ndarray, np.ndarray]:
    all_logits: list[float] = []
    all_labels: list[int]   = []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(imgs).squeeze(1)
        all_logits.extend(logits.cpu().float().tolist())
        all_labels.extend(labels.cpu().int().tolist())

    logits = np.asarray(all_logits, dtype=np.float32)
    labels = np.asarray(all_labels, dtype=np.int32)

    n_bad = int((~np.isfinite(logits)).sum())
    if n_bad:
        print(f"  Warning: {n_bad} non-finite logits — clamping")
        logits = np.nan_to_num(logits, nan=0.0, posinf=88.0, neginf=-88.0)

    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -88.0, 88.0)))
    return probs, labels


def pauc_above_tpr(y_true: np.ndarray, y_score: np.ndarray, min_tpr: float = 0.80) -> float:
    """ISIC 2024 competition metric: partial AUC above `min_tpr`.

    Perfect classifier scores 1 - min_tpr (= 0.20 at the 80% TPR cutoff).
    Mirrors the Kaggle scorer: flip labels/scores into an FPR-restricted
    AUROC (max_fpr = 1 - min_tpr) and undo McClish's [0.5, 1.0] rescale to
    recover the raw partial area.
    """
    v_gt   = 1 - y_true
    v_pred = -y_score
    max_fpr = 1 - min_tpr
    scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    return 0.5 * max_fpr ** 2 + (scaled - 0.5) * max_fpr * (1 - 0.5 * max_fpr)


def try_build(name: str, factory):
    try:
        ds = factory()
    except Exception as e:  # noqa: BLE001 — every failure mode here == "skip this dataset"
        print(f"[{name}] SKIP: {type(e).__name__}: {e}")
        return None
    if len(ds) == 0:
        print(f"[{name}] SKIP: empty dataset")
        return None
    return ds


def evaluate_dataset(name: str, ds, model, device, use_amp: bool, *, is_2024: bool = False) -> dict:
    n_pos = int(sum(ds.labels))
    n_neg = len(ds.labels) - n_pos
    print(f"[{name}] evaluating n={len(ds)}  malignant={n_pos}  benign={n_neg}")

    loader = get_eval_loader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    probs, labels = predict(model, loader, device, use_amp)

    preds = (probs >= 0.5).astype(np.int32)
    out = {
        "n":                 len(ds),
        "malignant":         n_pos,
        "benign":            n_neg,
        "auroc":             float(roc_auc_score(labels, probs)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
    }
    if is_2024:
        out["pauc_above_80tpr"] = float(pauc_above_tpr(labels, probs, min_tpr=0.80))
    return out


def print_summary(results: dict[str, dict], skipped: list[str]) -> None:
    bar = "=" * 78
    print()
    print(bar)
    print("Zero-shot evaluation summary")
    print(bar)
    header = f"{'Dataset':<22} {'n':>8} {'AUROC':>7} {'BalAcc':>7} {'pAUC@80TPR':>12}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        pauc = f"{r['pauc_above_80tpr']:.4f}" if "pauc_above_80tpr" in r else "-"
        print(f"{name:<22} {r['n']:>8} {r['auroc']:>7.4f} {r['balanced_accuracy']:>7.4f} {pauc:>12}")
    if skipped:
        print("-" * len(header))
        print(f"Skipped: {', '.join(skipped)}")
    print(bar)


def main() -> None:
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}  AMP: {use_amp}  Checkpoint: {CKPT_PATH}")

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    model = load_checkpoint(CKPT_PATH, device)

    plan = [
        ("ISIC 2016 heldout", load_isic2016_heldout,                                     False),
        ("ISIC 2020",         lambda: ISIC2020Dataset(ISIC2020_IMG_DIR, ISIC2020_CSV),   False),
        ("ISIC 2024",         lambda: ISIC2024Dataset(ISIC2024_IMG_DIR, ISIC2024_CSV),   True),
    ]

    results: dict[str, dict] = {}
    skipped: list[str] = []
    for name, factory, is_2024 in plan:
        ds = try_build(name, factory)
        if ds is None:
            skipped.append(name)
            continue
        results[name] = evaluate_dataset(name, ds, model, device, use_amp, is_2024=is_2024)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(
            {
                "checkpoint":  str(CKPT_PATH),
                "batch_size":  BATCH_SIZE,
                "results":     results,
                "skipped":     skipped,
            },
            f,
            indent=2,
        )
    print(f"\nSaved -> {OUT_PATH}")

    print_summary(results, skipped)


if __name__ == "__main__":
    main()
