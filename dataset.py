import math
import os
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

MALIGNANT = {"MEL", "BCC", "AK", "SCC"}
BENIGN = {"NV", "BKL", "DF", "VASC"}

NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))


def resolve_nested(root: str | Path) -> Path:
    """Auto-descend one level if `root` contains exactly one same-named subdirectory.

    Handles the double-nesting pattern from official ISIC zip extractions
    (e.g., .../ISIC_2024_Training_Input/ISIC_2024_Training_Input/). Only
    subdirectories are counted; stray files at the top level are ignored.
    Returns `root` unchanged if it doesn't exist, isn't a directory, or the
    pattern doesn't match.
    """
    root = Path(root)
    if not root.exists() or not root.is_dir():
        return root
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if len(subdirs) == 1 and subdirs[0].name == root.name:
        return subdirs[0]
    return root

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class SyntheticHair:
    """Draw randomly-curved dark hair-like lines onto a PIL image.

    Applied before Resize so lines get anti-aliased when the image is
    downsampled — this makes them look softer, closer to real dermoscopy
    hair than crisp 1-px pixel lines drawn at the target resolution.

    Fires with probability `p`. Each application draws
    `count_range[0]..count_range[1]` hairs. TRAIN-ONLY: never add this to
    VAL_TRANSFORM or any eval transform.
    """

    def __init__(
        self,
        p: float = 0.4,
        count_range: tuple[int, int] = (0, 8),
        thickness_range: tuple[int, int] = (1, 3),
        alpha_range: tuple[float, float] = (0.5, 0.9),
    ):
        self.p = p
        self.count_range = count_range
        self.thickness_range = thickness_range
        self.alpha_range = alpha_range

    def _draw_one_hair(self, draw: ImageDraw.ImageDraw, W: int, H: int) -> None:
        # Dark brown/black with slight jitter; alpha for blend
        base = random.randint(0, 35)
        color = (
            base + random.randint(0, 20),
            base + random.randint(0, 12),
            base + random.randint(0, 8),
            random.randint(
                int(255 * self.alpha_range[0]),
                int(255 * self.alpha_range[1]),
            ),
        )
        thickness = random.randint(*self.thickness_range)

        # Start slightly off-canvas so hairs can enter from any edge
        margin = max(W, H) // 4
        x0 = random.randint(-margin, W + margin)
        y0 = random.randint(-margin, H + margin)

        # Length is a fraction of the image diagonal
        diag = math.hypot(W, H)
        length = random.uniform(diag * 0.2, diag * 0.9)
        angle = random.uniform(0, 2 * math.pi)

        # Build a curved polyline: many short segments with gradual angle drift
        segments = 24
        step = length / segments
        pts = [(x0, y0)]
        cur_angle = angle
        for _ in range(segments):
            cur_angle += random.uniform(-0.12, 0.12)  # gentle curve
            dx = step * math.cos(cur_angle)
            dy = step * math.sin(cur_angle)
            pts.append((pts[-1][0] + dx, pts[-1][1] + dy))

        draw.line(pts, fill=color, width=thickness)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        n_hairs = random.randint(*self.count_range)
        if n_hairs == 0:
            return img

        W, H = img.size
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for _ in range(n_hairs):
            self._draw_one_hair(draw, W, H)

        composited = Image.alpha_composite(img.convert("RGBA"), overlay)
        return composited.convert("RGB")


TRAIN_TRANSFORM = transforms.Compose([
    SyntheticHair(p=0.4),
    transforms.Resize((288, 288)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


class ISICDataset(Dataset):
    """ISIC 2019 binary classification dataset (malignant vs. benign).

    Reads image IDs from the CSV — does not rely on directory listing —
    so stray non-image files in the image folder are ignored.
    UNK rows are dropped. Labels: 1 = malignant (MEL/BCC/AK/SCC),
    0 = benign (NV/BKL/DF/VASC).
    """

    def __init__(
        self,
        image_dir: str | Path,
        csv_path: str | Path,
        transform=None,
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform

        df = pd.read_csv(csv_path)
        df = df[df["UNK"] != 1.0].reset_index(drop=True)

        malignant_cols = list(MALIGNANT)
        self.image_ids: list[str] = df["image"].tolist()
        self.labels: list[int] = df[malignant_cols].any(axis=1).astype(int).tolist()

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_path = self.image_dir / f"{self.image_ids[idx]}.jpg"
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(self.labels[idx], dtype=torch.long)

    def with_transform(self, transform) -> "_TransformedSubset":
        """Return a view of this dataset with a different transform applied."""
        return _TransformedSubset(self, transform=transform)


class _TransformedSubset(Dataset):
    """A view over an ISICDataset that overrides the transform."""

    def __init__(self, base: ISICDataset, transform):
        self._base = base
        self._transform = transform

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        img_path = self._base.image_dir / f"{self._base.image_ids[idx]}.jpg"
        image = Image.open(img_path).convert("RGB")
        if self._transform is not None:
            image = self._transform(image)
        return image, torch.tensor(self._base.labels[idx], dtype=torch.long)


def _subset_with_transform(base: ISICDataset, indices: list[int], transform) -> Dataset:
    """Return a Subset of base restricted to indices, with a fresh transform."""
    wrapped = _TransformedSubset(base, transform)
    return Subset(wrapped, indices)


def get_loaders(
    image_dir: str | Path,
    csv_path: str | Path,
    val_split: float = 0.15,
    batch_size: int = 32,
    num_workers: int = NUM_WORKERS,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) with a stratified split on the binary label."""
    full = ISICDataset(image_dir, csv_path, transform=None)

    indices = list(range(len(full)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        stratify=full.labels,
        random_state=seed,
    )

    train_ds = _subset_with_transform(full, train_idx, TRAIN_TRANSFORM)
    val_ds = _subset_with_transform(full, val_idx, VAL_TRANSFORM)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
