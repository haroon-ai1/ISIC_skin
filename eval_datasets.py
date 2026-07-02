from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataset import VAL_TRANSFORM

# ── single swap point for Kaggle (mirrors train.py) ──────────────────────────
DATA_ROOT = Path("..")  # Kaggle: Path("/kaggle/input/isic-eval") or similar

ISIC2019_CSV = DATA_ROOT / "ISIC_2019_Training_GroundTruth.csv"

ISIC2016_IMG_DIR = DATA_ROOT / "ISBI2016_ISIC_Part3_Training_Data"
ISIC2016_CSV     = DATA_ROOT / "ISBI2016_ISIC_Part3_Training_GroundTruth.csv"

ISIC2024_IMG_DIR = DATA_ROOT / "ISIC_2024_Training_Input" / "ISIC_2024_Training_Input"
ISIC2024_CSV     = DATA_ROOT / "ISIC_2024_Training_GroundTruth.csv"


class ISIC2016Dataset(Dataset):
    """ISIC 2016 (ISBI Part 3) eval dataset.

    CSV has no header: col 0 = image_id, col 1 = 'benign' | 'malignant'.
    Labels: 0 = benign, 1 = malignant.

    IMPORTANT: 568 of 900 IDs also appear in ISIC 2019. For a true zero-shot
    evaluation of a 2019-trained model, pass `exclude_ids=<set of 2019 IDs>`
    (see load_isic2016_heldout()).
    """

    def __init__(
        self,
        image_dir: str | Path,
        csv_path: str | Path,
        transform=VAL_TRANSFORM,
        exclude_ids: set[str] | None = None,
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform

        df = pd.read_csv(csv_path, header=None, names=["image_id", "label"])
        if exclude_ids:
            df = df[~df["image_id"].isin(exclude_ids)].reset_index(drop=True)

        self.image_ids: list[str] = df["image_id"].tolist()
        self.labels: list[int]    = (df["label"] == "malignant").astype(int).tolist()

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_path = self.image_dir / f"{self.image_ids[idx]}.jpg"
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(self.labels[idx], dtype=torch.long)


class ISIC2024Dataset(Dataset):
    """ISIC 2024 eval dataset.

    CSV columns: isic_id, malignant (0.0 / 1.0 float). No relabeling needed.
    Labels: 0 = benign, 1 = malignant.

    Note: images live one directory deeper than the top-level folder.
    """

    def __init__(self, image_dir: str | Path, csv_path: str | Path, transform=VAL_TRANSFORM):
        self.image_dir = Path(image_dir)
        self.transform = transform

        df = pd.read_csv(csv_path)
        self.image_ids: list[str] = df["isic_id"].tolist()
        self.labels: list[int]    = df["malignant"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_path = self.image_dir / f"{self.image_ids[idx]}.jpg"
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(self.labels[idx], dtype=torch.long)


def _read_isic2019_ids(csv_path: str | Path = ISIC2019_CSV) -> set[str]:
    return set(pd.read_csv(csv_path)["image"].tolist())


def load_isic2016_heldout(
    image_dir: str | Path = ISIC2016_IMG_DIR,
    csv_path: str | Path = ISIC2016_CSV,
    isic2019_csv: str | Path = ISIC2019_CSV,
) -> ISIC2016Dataset:
    """ISIC 2016 with any IDs appearing in ISIC 2019 removed.

    This is the true zero-shot subset for evaluating a 2019-trained model
    (~332 of the 900 images survive; 568 overlap with 2019 training).
    """
    train_ids = _read_isic2019_ids(isic2019_csv)
    return ISIC2016Dataset(image_dir, csv_path, exclude_ids=train_ids)


def get_eval_loader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoader:
    """Inference-mode DataLoader: no shuffle, pin_memory, persistent workers."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
