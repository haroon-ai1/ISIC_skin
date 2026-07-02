"""Dataset inspection: contact sheets + image dimension distributions.

Produces one 5x4 grid PNG per dataset (2016/2019/2024) with 20 random labeled
images, and prints width/height/aspect-ratio stats. For 2024 the dimension
stats are estimated on a 2000-image sample (full dataset is 401k images).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, ".")
from eval_datasets import (
    ISIC2016_CSV, ISIC2016_IMG_DIR,
    ISIC2024_CSV, ISIC2024_IMG_DIR,
)

SEED = 42
random.seed(SEED)

ISIC2019_CSV     = Path("../ISIC_2019_Training_GroundTruth.csv")
ISIC2019_IMG_DIR = Path("../ISIC_2019_Training_Input")

OUT_DIR = Path("dataset_inspection")
OUT_DIR.mkdir(exist_ok=True)


def _load_2016() -> tuple[list[str], list[int], Path]:
    df = pd.read_csv(ISIC2016_CSV, header=None, names=["image_id", "label"])
    ids    = df["image_id"].tolist()
    labels = (df["label"] == "malignant").astype(int).tolist()
    return ids, labels, ISIC2016_IMG_DIR


def _load_2019() -> tuple[list[str], list[int], Path]:
    df = pd.read_csv(ISIC2019_CSV)
    df = df[df["UNK"] != 1.0].reset_index(drop=True)
    ids    = df["image"].tolist()
    labels = df[["MEL", "BCC", "AK", "SCC"]].any(axis=1).astype(int).tolist()
    return ids, labels, ISIC2019_IMG_DIR


def _load_2024() -> tuple[list[str], list[int], Path]:
    df = pd.read_csv(ISIC2024_CSV)
    ids    = df["isic_id"].tolist()
    labels = df["malignant"].astype(int).tolist()
    return ids, labels, ISIC2024_IMG_DIR


def sample_indices(labels: list[int], n: int, seed: int) -> list[int]:
    """Random sample of `n` indices, capped at len(labels)."""
    rng = random.Random(seed)
    return rng.sample(range(len(labels)), min(n, len(labels)))


def contact_sheet(name: str, ids: list[str], labels: list[int], img_dir: Path,
                  indices: list[int], out_path: Path) -> None:
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    n_mal = sum(labels[i] for i in indices)
    n_ben = len(indices) - n_mal
    fig.suptitle(f"{name}  |  sample of {len(indices)}  "
                 f"(malignant={n_mal}, benign={n_ben})", fontsize=14, y=0.995)

    for ax, idx in zip(axes.flat, indices):
        img = Image.open(img_dir / f"{ids[idx]}.jpg").convert("RGB")
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        tag = "malignant" if labels[idx] else "benign"
        color = "red" if labels[idx] else "green"
        ax.set_title(f"{ids[idx]}\n{tag}", fontsize=8, color=color)

    plt.tight_layout(rect=(0, 0, 1, 0.98))
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def dimension_stats(name: str, ids: list[str], img_dir: Path,
                    sample_size: int | None = None) -> None:
    if sample_size and sample_size < len(ids):
        sampled = random.Random(SEED + 1).sample(ids, sample_size)
        note = f" (sample of {sample_size} / {len(ids)})"
    else:
        sampled = ids
        note    = f" (full n={len(ids)})"

    widths, heights = [], []
    for iid in sampled:
        with Image.open(img_dir / f"{iid}.jpg") as im:
            widths.append(im.size[0]); heights.append(im.size[1])

    w = np.array(widths); h = np.array(heights); ar = w / h
    print(f"\n{name}{note}")
    print(f"  width  : min={w.min():>5}  max={w.max():>5}  mean={w.mean():>7.1f}  std={w.std():>6.1f}")
    print(f"  height : min={h.min():>5}  max={h.max():>5}  mean={h.mean():>7.1f}  std={h.std():>6.1f}")
    print(f"  aspect : min={ar.min():>5.2f}  max={ar.max():>5.2f}  mean={ar.mean():>7.3f}  std={ar.std():>6.3f}")
    portrait_frac  = (ar < 0.95).mean()
    landscape_frac = (ar > 1.05).mean()
    square_frac    = 1 - portrait_frac - landscape_frac
    print(f"  shape mix: {landscape_frac*100:.1f}% landscape  {portrait_frac*100:.1f}% portrait  {square_frac*100:.1f}% ~square")


def main() -> None:
    for name, loader, out_name, dim_sample in [
        ("ISIC 2016",     _load_2016, "contact_2016.png",  None),
        ("ISIC 2019",     _load_2019, "contact_2019.png",  2000),
        ("ISIC 2024",     _load_2024, "contact_2024.png",  2000),
    ]:
        ids, labels, img_dir = loader()
        idxs = sample_indices(labels, 20, seed=SEED)
        print(f"\n=== {name} ({len(ids)} images) ===")
        contact_sheet(name, ids, labels, img_dir, idxs, OUT_DIR / out_name)
        dimension_stats(name, ids, img_dir, sample_size=dim_sample)


if __name__ == "__main__":
    main()
