"""Visualize the SyntheticHair augmentation on real ISIC 2019 images.

Renders 6 (original, augmented) pairs into a single PNG so we can eyeball
whether the synthetic hair reads as reasonable dermoscopy artifact vs.
obvious digital paste-on.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

sys.path.insert(0, ".")
from dataset import SyntheticHair

SEED = 7  # picked so we get a range of skin tones / lesion types
random.seed(SEED)

CSV = Path("../ISIC_2019_Training_GroundTruth.csv")
IMG = Path("../ISIC_2019_Training_Input")
OUT = Path("dataset_inspection/hair_aug_samples.png")


def main() -> None:
    df  = pd.read_csv(CSV)
    df  = df[df["UNK"] != 1.0].reset_index(drop=True)
    ids = df["image"].tolist()
    picks = random.sample(ids, 6)

    # Force augmentation to fire (bypass p) and use the full expected count range.
    # Two variants: light hair (2-4) and heavy hair (5-8) so we see both regimes.
    aug_light = SyntheticHair(p=1.0, count_range=(2, 4))
    aug_heavy = SyntheticHair(p=1.0, count_range=(5, 8))

    fig, axes = plt.subplots(6, 2, figsize=(9, 22))
    fig.suptitle("SyntheticHair augmentation — original (left) vs augmented (right)",
                 fontsize=13, y=0.995)

    for row, iid in enumerate(picks):
        img = Image.open(IMG / f"{iid}.jpg").convert("RGB")
        aug = (aug_light if row < 3 else aug_heavy)(img)
        density = "light (2-4 hairs)" if row < 3 else "heavy (5-8 hairs)"

        axes[row, 0].imshow(img); axes[row, 0].set_title(f"{iid} original", fontsize=10)
        axes[row, 1].imshow(aug); axes[row, 1].set_title(f"{iid} + {density}", fontsize=10)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(rect=(0, 0, 1, 0.99))
    OUT.parent.mkdir(exist_ok=True)
    plt.savefig(OUT, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
