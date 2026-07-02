"""Profile the data-loading and augmentation pipeline.

Measures things that transfer across machines (CPU aug cost, worker/prefetch
behaviour) and things that don't (GPU utilization on this specific card).
Numbers below reflect the machine this runs on — Kaggle T4x2 will differ.
"""
from __future__ import annotations

import os
import random
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, ".")
from dataset import SyntheticHair, get_loaders

CSV = Path("C:/ISIC/ISIC_2019_Training_GroundTruth.csv")
IMG = Path("C:/ISIC/ISIC_2019_Training_Input")


# ── 1. CPU count ─────────────────────────────────────────────────────────────
def section_cpu() -> None:
    print("=" * 78)
    print("#1  CPU CORES")
    print("=" * 78)
    print(f"os.cpu_count()          : {os.cpu_count()}")
    print(f"len(os.sched_getaffinity(0)) : "
          + (str(len(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else "N/A on Windows"))
    print(f"torch.get_num_threads() : {torch.get_num_threads()}")
    print(f"NUM_WORKERS in dataset  : 4  (from train.py)")


# ── 2. Hair augmentation cost ────────────────────────────────────────────────
def section_hair() -> None:
    print()
    print("=" * 78)
    print("#2  SyntheticHair COST (p=1.0, 100 iters, real ISIC 2019 image)")
    print("=" * 78)
    # Pick a real image at typical resolution
    import pandas as pd
    df = pd.read_csv(CSV)
    iid = df["image"].iloc[0]
    src = Image.open(IMG / f"{iid}.jpg").convert("RGB")
    print(f"Source image            : {iid}  {src.size}")

    aug = SyntheticHair(p=1.0, count_range=(4, 8))  # forced-fire, mid-heavy

    # warmup
    for _ in range(5):
        aug(src)

    ms = []
    for _ in range(100):
        t0 = time.perf_counter()
        aug(src)
        ms.append((time.perf_counter() - t0) * 1000)

    print(f"per-image  mean={statistics.mean(ms):.2f} ms   "
          f"median={statistics.median(ms):.2f} ms   "
          f"p95={sorted(ms)[94]:.2f} ms   "
          f"max={max(ms):.2f} ms")
    print(f"Amortized at p=0.4      : {statistics.mean(ms) * 0.4:.2f} ms/image (avg over training)")


# ── 3. nvidia-smi background sampler ─────────────────────────────────────────
class GpuSampler:
    def __init__(self, interval_s: float = 0.2):
        self.interval = interval_s
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            )
            try:
                self.samples.append(int(r.stdout.strip().splitlines()[0]))
            except (ValueError, IndexError):
                pass
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self) -> str:
        if not self.samples:
            return "no samples"
        s = self.samples
        return (f"n={len(s)}  mean={statistics.mean(s):.1f}%  "
                f"median={statistics.median(s):.1f}%  "
                f"min={min(s)}%  max={max(s)}%  "
                f"pct_above_50={sum(1 for x in s if x > 50)/len(s)*100:.0f}%")


# ── 4. batch timing at different prefetch_factors ───────────────────────────
def time_loader(prefetch_factor: int, model, batch_size: int,
                warmup: int = 3, timed: int = 20) -> tuple[float, GpuSampler]:
    """Return (avg seconds/batch, GPU util summary). Uses real B3 forward+backward
    so GPU util samples reflect an actual training step, not a trivial op."""
    from dataset import ISICDataset, _subset_with_transform, TRAIN_TRANSFORM
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split

    full = ISICDataset(IMG, CSV, transform=None)
    idx  = list(range(len(full)))
    train_idx, _ = train_test_split(
        idx, test_size=0.15, stratify=full.labels, random_state=42,
    )
    train_ds = _subset_with_transform(full, train_idx, TRAIN_TRANSFORM)

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )

    device    = torch.device("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler    = torch.amp.GradScaler("cuda")
    criterion = torch.nn.BCEWithLogitsLoss()
    it        = iter(loader)

    def step(imgs, labels):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    for _ in range(warmup):
        step(*next(it))
    torch.cuda.synchronize()

    sampler = GpuSampler(interval_s=0.15)
    sampler.start()

    t0 = time.perf_counter()
    for _ in range(timed):
        step(*next(it))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    sampler.stop()
    del it
    del loader
    return elapsed / timed, sampler


def section_prefetch() -> None:
    from model import build_model
    # MX150 (2GB) can't fit B3 fwd+bwd at bs=64. Use bs=16 locally; report both.
    batch_size = 16
    print()
    print("=" * 78)
    print(f"#3-4  prefetch_factor SWEEP + GPU UTILIZATION  (bs={batch_size}, workers=4, "
          f"real B3 fwd+bwd+AMP, 20 timed batches)")
    print("=" * 78)
    # Build model ONCE outside the sweep so we don't re-download weights each iter
    device = torch.device("cuda")
    model  = build_model(pretrained=True).to(device)
    print(f"{'pref':>6}  {'s/batch':>8}  {'imgs/s':>8}   GPU utilization")
    print("-" * 78)
    for pf in [2, 4, 6]:
        sec_per_batch, sampler = time_loader(pf, model, batch_size)
        imgs_per_s = batch_size / sec_per_batch
        print(f"{pf:>6}  {sec_per_batch:>8.3f}  {imgs_per_s:>8.1f}   {sampler.summary()}")


def main() -> None:
    section_cpu()
    section_hair()
    if torch.cuda.is_available():
        section_prefetch()
    else:
        print("\n(GPU not available — skipping prefetch/GPU-util section)")


if __name__ == "__main__":
    main()
