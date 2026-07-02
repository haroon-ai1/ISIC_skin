from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataset import ISICDataset, VAL_TRANSFORM, resolve_nested
from setup_data import DATA_ROOT

# 2019 (training) — repeated here so the summary can report it in one place.
ISIC2019_IMG_DIR = resolve_nested(DATA_ROOT / "ISIC_2019_Training_Input")
ISIC2019_CSV     = DATA_ROOT / "ISIC_2019_Training_GroundTruth.csv"

# 2016
ISIC2016_IMG_DIR = resolve_nested(DATA_ROOT / "ISBI2016_ISIC_Part3_Training_Data")
ISIC2016_CSV     = DATA_ROOT / "ISBI2016_ISIC_Part3_Training_GroundTruth.csv"

# 2024
ISIC2024_IMG_DIR = resolve_nested(DATA_ROOT / "ISIC_2024_Training_Input")
ISIC2024_CSV     = DATA_ROOT / "ISIC_2024_Training_GroundTruth.csv"

# 2020 — Kaggle-only absolute path (won't exist locally; that's fine)
_ISIC2020_ROOT   = Path("/kaggle/input/datasets/nischaydnk/isic-2020-jpg-256x256-resized")
ISIC2020_IMG_DIR = resolve_nested(_ISIC2020_ROOT / "train-image" / "image")
ISIC2020_CSV     = _ISIC2020_ROOT / "train-metadata.csv"


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


class ISIC2020Dataset(Dataset):
    """ISIC 2020 eval dataset (Kaggle: nischaydnk/isic-2020-jpg-256x256-resized).

    Image column: 'image_name'. Label preference: 'target' (0/1) first, then
    fall back to 'benign_malignant' (string) if 'target' isn't present.
    Labels: 0 = benign, 1 = malignant.
    """

    def __init__(self, image_dir: str | Path, csv_path: str | Path, transform=VAL_TRANSFORM):
        self.image_dir = Path(image_dir)
        self.transform = transform

        df = pd.read_csv(csv_path)
        self.image_ids: list[str] = df["image_name"].tolist()

        if "target" in df.columns:
            self.labels: list[int] = df["target"].astype(int).tolist()
        elif "benign_malignant" in df.columns:
            self.labels = (df["benign_malignant"] == "malignant").astype(int).tolist()
        else:
            raise KeyError(
                f"Expected 'target' or 'benign_malignant' column in {csv_path}. "
                f"Got: {df.columns.tolist()}"
            )

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

    ~332 of 900 images survive (568 overlap with 2019 training).
    Use this — not the full 900-image set — for reporting zero-shot 2016 results.
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


# ── startup sanity check ─────────────────────────────────────────────────────

def _try_summarize(
    name: str,
    img_dir: Path,
    csv_path: Path,
    spec_img_dir: Path,
    factory,
) -> dict:
    """Build a status dict for the summary table by actually instantiating the dataset."""
    entry = {
        "name":     name,
        "img_dir":  str(img_dir),
        "csv_path": str(csv_path),
        "nested":   img_dir != spec_img_dir,
        "status":   "OK",
        "n":        "-",
        "mal":      "-",
        "ben":      "-",
    }
    if not Path(csv_path).exists():
        entry["status"] = "CSV missing"
        return entry
    if not Path(img_dir).exists():
        entry["status"] = "IMG missing"
        return entry
    try:
        ds = factory()
        entry["n"]   = len(ds)
        entry["mal"] = sum(ds.labels)
        entry["ben"] = len(ds.labels) - entry["mal"]
    except Exception as e:  # noqa: BLE001 — surface any load error in the table
        entry["status"] = f"ERROR: {type(e).__name__}: {e}"
    return entry


def print_dataset_summary() -> None:
    """Print resolved paths + label counts for all configured datasets.

    Called at train startup so a wrong mount point is obvious immediately.
    Gracefully handles missing datasets (e.g., 2020 running locally).
    """
    entries = [
        _try_summarize(
            "ISIC 2019 (train)",
            ISIC2019_IMG_DIR, ISIC2019_CSV,
            spec_img_dir=DATA_ROOT / "ISIC_2019_Training_Input",
            factory=lambda: ISICDataset(ISIC2019_IMG_DIR, ISIC2019_CSV),
        ),
        _try_summarize(
            "ISIC 2016 (full)",
            ISIC2016_IMG_DIR, ISIC2016_CSV,
            spec_img_dir=DATA_ROOT / "ISBI2016_ISIC_Part3_Training_Data",
            factory=lambda: ISIC2016Dataset(ISIC2016_IMG_DIR, ISIC2016_CSV),
        ),
        _try_summarize(
            "ISIC 2016 (heldout)",
            ISIC2016_IMG_DIR, ISIC2016_CSV,
            spec_img_dir=DATA_ROOT / "ISBI2016_ISIC_Part3_Training_Data",
            factory=lambda: load_isic2016_heldout(),
        ),
        _try_summarize(
            "ISIC 2024",
            ISIC2024_IMG_DIR, ISIC2024_CSV,
            spec_img_dir=DATA_ROOT / "ISIC_2024_Training_Input",
            factory=lambda: ISIC2024Dataset(ISIC2024_IMG_DIR, ISIC2024_CSV),
        ),
        _try_summarize(
            "ISIC 2020",
            ISIC2020_IMG_DIR, ISIC2020_CSV,
            spec_img_dir=_ISIC2020_ROOT / "train-image" / "image",
            factory=lambda: ISIC2020Dataset(ISIC2020_IMG_DIR, ISIC2020_CSV),
        ),
    ]

    bar = "=" * 92
    print(bar)
    print("Dataset resolution summary")
    print(bar)

    for e in entries:
        head = f"{e['name']:<22}  status={e['status']}"
        if e["status"] == "OK":
            head += (
                f"  n={e['n']:>7}  malignant={e['mal']:>6}  benign={e['ben']:>7}"
            )
        print(head)
        nested = "  (auto-descended)" if e["nested"] else ""
        print(f"  images : {e['img_dir']}{nested}")
        print(f"  labels : {e['csv_path']}")

    nested_names = [e["name"] for e in entries if e["nested"] and e["status"] == "OK"]
    print()
    if nested_names:
        print(f"Nested-folder auto-descend fired for: {', '.join(nested_names)}")
    else:
        print("No nested-folder auto-descend fired.")
    print("Note: 'ISIC 2016 (heldout)' = 2016 minus any IDs present in ISIC 2019.")
    print(bar)


if __name__ == "__main__":
    print_dataset_summary()
