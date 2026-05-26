"""Download DIV2K archives and extract them into the project datasets folder.

This avoids relying on Hugging Face legacy dataset scripts, which are no
longer supported by recent versions of `datasets`.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


BASE_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/"
URLS = {
    "train_hr": BASE_URL + "DIV2K_train_HR.zip",
    "valid_hr": BASE_URL + "DIV2K_valid_HR.zip",
    "train_bicubic_x2": BASE_URL + "DIV2K_train_LR_bicubic_X2.zip",
    "train_unknown_x2": BASE_URL + "DIV2K_train_LR_unknown_X2.zip",
    "valid_bicubic_x2": BASE_URL + "DIV2K_valid_LR_bicubic_X2.zip",
    "valid_unknown_x2": BASE_URL + "DIV2K_valid_LR_unknown_X2.zip",
    "train_bicubic_x3": BASE_URL + "DIV2K_train_LR_bicubic_X3.zip",
    "train_unknown_x3": BASE_URL + "DIV2K_train_LR_unknown_X3.zip",
    "valid_bicubic_x3": BASE_URL + "DIV2K_valid_LR_bicubic_X3.zip",
    "valid_unknown_x3": BASE_URL + "DIV2K_valid_LR_unknown_X3.zip",
    "train_bicubic_x4": BASE_URL + "DIV2K_train_LR_bicubic_X4.zip",
    "train_unknown_x4": BASE_URL + "DIV2K_train_LR_unknown_X4.zip",
    "valid_bicubic_x4": BASE_URL + "DIV2K_valid_LR_bicubic_X4.zip",
    "valid_unknown_x4": BASE_URL + "DIV2K_valid_LR_unknown_X4.zip",
    "train_bicubic_x8": BASE_URL + "DIV2K_train_LR_x8.zip",
    "valid_bicubic_x8": BASE_URL + "DIV2K_valid_LR_x8.zip",
    "train_realistic_mild_x4": BASE_URL + "DIV2K_train_LR_mild.zip",
    "valid_realistic_mild_x4": BASE_URL + "DIV2K_valid_LR_mild.zip",
    "train_realistic_difficult_x4": BASE_URL + "DIV2K_train_LR_difficult.zip",
    "valid_realistic_difficult_x4": BASE_URL + "DIV2K_valid_LR_difficult.zip",
    "train_realistic_wild_x4": BASE_URL + "DIV2K_train_LR_wild.zip",
    "valid_realistic_wild_x4": BASE_URL + "DIV2K_valid_LR_wild.zip",
}

VALID_CONFIGS = (
    "bicubic_x2",
    "bicubic_x3",
    "bicubic_x4",
    "bicubic_x8",
    "unknown_x2",
    "unknown_x3",
    "unknown_x4",
    "realistic_mild_x4",
    "realistic_difficult_x4",
    "realistic_wild_x4",
)
VALID_SPLITS = ("train", "validation", "all")


def download_file(url: str, dest: Path) -> None:
    """Download a file unless it already exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Using existing archive: {dest.name}")
        return

    print(f"Downloading {url}")

    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            downloaded = block_num * block_size / (1024 * 1024)
            print(f"\rDownloaded: {downloaded:.1f} MB", end="", flush=True)
            return

        downloaded = min(block_num * block_size, total_size)
        remaining = max(total_size - downloaded, 0)
        percent = downloaded / total_size * 100
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        remaining_mb = remaining / (1024 * 1024)
        print(
            f"\r{percent:5.1f}%  downloaded {downloaded_mb:8.1f} MB / {total_mb:8.1f} MB"
            f"  left {remaining_mb:8.1f} MB",
            end="",
            flush=True,
        )

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()


def extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    """Extract a ZIP archive and return the top-level extracted directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        top_levels = {
            Path(name).parts[0]
            for name in archive.namelist()
            if name and not name.endswith("/") and Path(name).parts
        }
        archive.extractall(dest_dir)

    if len(top_levels) == 1:
        return dest_dir / next(iter(top_levels))
    return dest_dir


def move_contents(src_dir: Path, dest_dir: Path) -> int:
    """Move image files from src_dir into dest_dir, flattening nested folders."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    moved = 0

    for image_path in sorted(src_dir.rglob("*.png")):
        target = dest_dir / image_path.name
        if target.exists():
            moved += 1
            continue
        shutil.move(str(image_path), str(target))
        moved += 1

    return moved


def resolve_split_key(split: str) -> str:
    return "valid" if split == "validation" else split


def download_split(config: str, split: str, cache_dir: Path, export_base_dir: Path, no_export: bool) -> None:
    """Download one DIV2K split and optionally extract it."""
    split_key = resolve_split_key(split)
    hr_key = f"{split_key}_hr"
    lr_key = f"{split_key}_{config}"

    hr_url = URLS[hr_key]
    lr_url = URLS[lr_key]

    hr_zip = cache_dir / Path(hr_url).name
    lr_zip = cache_dir / Path(lr_url).name
    export_dir = export_base_dir / f"div2k_{config}_{split}"

    print(f"Downloading DIV2K config={config} split={split}")
    print(f"Archive directory: {cache_dir}")

    download_file(hr_url, hr_zip)
    download_file(lr_url, lr_zip)

    if no_export:
        print("Archive download complete. Skipping extraction.")
        return

    hr_extract_root = export_dir / "_raw_hr"
    lr_extract_root = export_dir / "_raw_lr"
    hr_final_dir = export_dir / "hr"
    lr_final_dir = export_dir / "lr"

    print(f"Extracting into: {export_dir}")
    hr_src = extract_zip(hr_zip, hr_extract_root)
    lr_src = extract_zip(lr_zip, lr_extract_root)

    hr_count = move_contents(hr_src, hr_final_dir)
    lr_count = move_contents(lr_src, lr_final_dir)

    print(f"HR images ready at {hr_final_dir} ({hr_count} files)")
    print(f"LR images ready at {lr_final_dir} ({lr_count} files)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download DIV2K into LearnedCompression/datasets."
    )
    parser.add_argument(
        "--config",
        default="bicubic_x2",
        choices=VALID_CONFIGS,
        help="DIV2K variant to download.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=VALID_SPLITS,
        help="Dataset split to download. Use all for both train and validation.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(Path(__file__).resolve().parents[1] / "datasets" / "div2k_archives"),
        help="Directory for downloaded ZIP archives.",
    )
    parser.add_argument(
        "--export-dir",
        default=None,
        help="Directory to extract images into. Defaults to datasets/div2k_<config>_<split>.",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Only download the ZIP archives and skip extraction.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    export_base_dir = (
        Path(args.export_dir).resolve()
        if args.export_dir
        else cache_dir.parent
    )
    splits = ("train", "validation") if args.split == "all" else (args.split,)

    for split in splits:
        download_split(args.config, split, cache_dir, export_base_dir, args.no_export)


if __name__ == "__main__":
    main()
