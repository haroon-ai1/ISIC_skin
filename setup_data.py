"""Idempotent data setup for the ISIC pipeline.

Ensures ISIC 2019 / 2016 / 2024 image folders and CSVs exist under a single
fixed DATA_ROOT. Downloads from the official S3 URLs and extracts to
DATA_ROOT with any double-nested folder pattern (e.g.
ISIC_2019_Training_Input/ISIC_2019_Training_Input/) flattened at extract
time — no runtime auto-detect for the freshly-downloaded copy.

ISIC 2020 is skipped: attach it as a Kaggle Input dataset instead.

Call `setup_data()` once at the start of a training run. It is a no-op when
every dataset is already present with the expected file count.
"""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

from dataset import resolve_nested

# ── environment detection + fixed roots ──────────────────────────────────────
IS_KAGGLE = Path("/kaggle/working").exists()
DATA_ROOT = Path("/kaggle/working/data") if IS_KAGGLE else Path("C:/ISIC")

# ── dataset specs ────────────────────────────────────────────────────────────
_S3 = "https://isic-archive.s3.amazonaws.com/challenges"

DATASETS: dict[str, dict] = {
    "isic2019": {
        "img_url":      f"{_S3}/2019/ISIC_2019_Training_Input.zip",
        "csv_url":      f"{_S3}/2019/ISIC_2019_Training_GroundTruth.csv",
        "img_dir_name": "ISIC_2019_Training_Input",
        "csv_name":     "ISIC_2019_Training_GroundTruth.csv",
        "expected_n":   25331,
    },
    "isic2016": {
        "img_url":      f"{_S3}/2016/ISBI2016_ISIC_Part3_Training_Data.zip",
        "csv_url":      f"{_S3}/2016/ISBI2016_ISIC_Part3_Training_GroundTruth.csv",
        "img_dir_name": "ISBI2016_ISIC_Part3_Training_Data",
        "csv_name":     "ISBI2016_ISIC_Part3_Training_GroundTruth.csv",
        "expected_n":   900,
    },
    "isic2024": {
        "img_url":      f"{_S3}/2024/ISIC_2024_Training_Input.zip",
        "csv_url":      f"{_S3}/2024/ISIC_2024_Training_GroundTruth.csv",
        "img_dir_name": "ISIC_2024_Training_Input",
        "csv_name":     "ISIC_2024_Training_GroundTruth.csv",
        "expected_n":   401059,
    },
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _count_jpgs(d: Path) -> int:
    """Count jpg files in d, transparently descending through the nested-folder
    quirk in case the user has pre-existing double-nested data (e.g. local
    Windows dev before setup_data has flattened anything)."""
    d = resolve_nested(d)
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if p.suffix.lower() == ".jpg")


def _wget(url: str, out_path: Path) -> None:
    """Download url to out_path via wget. Uses a `.partial` marker to avoid
    treating a half-downloaded file as complete on the next run."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(out_path.suffix + ".partial")
    print(f"  Downloading {url}")
    subprocess.run(
        ["wget", "-q", "--show-progress", "-O", str(partial), url],
        check=True,
    )
    partial.rename(out_path)


def _extract_and_flatten(zip_path: Path, target_dir: Path) -> None:
    """Extract zip into target_dir. If the archive nests a single top-level
    subdirectory (the known ISIC quirk), flatten it so files end up directly
    under target_dir. Uses a sibling temp directory to keep the check clean."""
    print(f"  Extracting {zip_path.name}  (may take a while for large archives)")
    tmp = target_dir.parent / f"_extract_{target_dir.name}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)

    entries = list(tmp.iterdir())
    subdirs = [p for p in entries if p.is_dir()]
    files   = [p for p in entries if p.is_file()]
    if len(subdirs) == 1 and not files:
        source = subdirs[0]
        print(f"  Flattening nested '{source.name}/'")
    else:
        source = tmp

    target_dir.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        item.rename(target_dir / item.name)
    shutil.rmtree(tmp)


def _fetch_dataset(key: str, spec: dict) -> None:
    """Download + extract images (if needed) and CSV (if needed) for one dataset."""
    img_dir  = DATA_ROOT / spec["img_dir_name"]
    csv_path = DATA_ROOT / spec["csv_name"]
    expected = spec["expected_n"]

    n_jpgs = _count_jpgs(img_dir)
    csv_ok = csv_path.is_file() and csv_path.stat().st_size > 0
    img_ok = n_jpgs >= expected

    if img_ok and csv_ok:
        print(f"[{key}] OK  ({n_jpgs} images, csv present)")
        return

    print(
        f"[{key}] MISSING or incomplete "
        f"— images={n_jpgs}/{expected}, csv={'OK' if csv_ok else 'missing'}"
    )

    if not img_ok:
        if img_dir.exists():
            print(f"  Clearing incomplete {img_dir}")
            shutil.rmtree(img_dir)
        zip_path = DATA_ROOT / f"{spec['img_dir_name']}.zip"
        try:
            _wget(spec["img_url"], zip_path)
            _extract_and_flatten(zip_path, img_dir)
        finally:
            if zip_path.exists():
                zip_path.unlink()

    if not csv_ok:
        _wget(spec["csv_url"], csv_path)

    n_jpgs = _count_jpgs(img_dir)
    if n_jpgs >= expected and csv_path.is_file():
        print(f"[{key}] OK  ({n_jpgs} images downloaded)")
    else:
        print(f"[{key}] WARNING: after download, images={n_jpgs}/{expected}, "
              f"csv={'present' if csv_path.is_file() else 'missing'}")


# ── public entry point ───────────────────────────────────────────────────────

def setup_data() -> None:
    """Ensure all downloadable datasets exist under DATA_ROOT. Idempotent.

    2020 is skipped: attach the Kaggle dataset instead.
    """
    print(f"[setup_data] IS_KAGGLE={IS_KAGGLE}  DATA_ROOT={DATA_ROOT}")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for key, spec in DATASETS.items():
        _fetch_dataset(key, spec)
    print("[setup_data] 2020 skipped (attach as Kaggle Input)")
    print("[setup_data] done\n")


if __name__ == "__main__":
    setup_data()
