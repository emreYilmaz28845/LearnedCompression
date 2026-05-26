"""Dataset utilities for learned image compression.

- CLICDataset: CLIC 2020 training images with random 256x256 crops
- KodakDataset: Kodak PhotoCD evaluation images (padded to multiple of 64)
- Download helpers for both datasets
"""

import glob
import os
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F


class CLICDataset(Dataset):
    """Image-folder dataset for CLIC-style compression training images."""

    def __init__(self, root, patch_size=256, training=True):
        self.root = Path(root)
        self.patch_size = patch_size
        self.training = training
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        self.images = []
        for ext in exts:
            self.images.extend(glob.glob(str(self.root / "**" / ext), recursive=True))
        self.images.sort()

        if not self.images:
            raise FileNotFoundError(f"No images found in {root}")

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        img = self._pad_if_needed(img)
        if self.training:
            img = transforms.RandomCrop(self.patch_size)(img)
            img = transforms.RandomHorizontalFlip()(img)
        else:
            img = transforms.CenterCrop(self.patch_size)(img)
        return self.to_tensor(img)

    def _pad_if_needed(self, img):
        w, h = img.size
        pad_w = max(self.patch_size - w, 0)
        pad_h = max(self.patch_size - h, 0)
        if pad_w == 0 and pad_h == 0:
            return img
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        return F.pad(img, [left, top, right, bottom], padding_mode="reflect")


class KodakDataset(Dataset):
    """Kodak PhotoCD dataset for evaluation (24 images, 768x512)."""

    def __init__(self, root):
        self.root = Path(root)
        self.images = sorted(glob.glob(str(self.root / "*.png")))
        if not self.images:
            raise FileNotFoundError(f"No PNG images found in {root}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        w, h = img.size
        pad_h = (64 - h % 64) % 64
        pad_w = (64 - w % 64) % 64
        transform = transforms.Compose([
            transforms.Pad((0, 0, pad_w, pad_h), padding_mode="reflect"),
            transforms.ToTensor(),
        ])
        tensor = transform(img)
        return tensor, (h, w)


def download_kodak(dest_dir):
    """Download the 24 Kodak PhotoCD images."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    base_url = "https://r0k.us/graphics/kodak/kodak/"
    for i in range(1, 25):
        fname = f"kodim{i:02d}.png"
        fpath = dest / fname
        if fpath.exists():
            continue
        url = base_url + fname
        print(f"Downloading {fname}...")
        urllib.request.urlretrieve(url, fpath)
    print(f"Kodak dataset ready at {dest} ({len(list(dest.glob('*.png')))} images)")


def download_clic(dest_dir):
    """Download the CLIC 2020 professional training set.

    Note: CLIC images are hosted on various mirrors. This downloads the
    professional training split (~1200 images). If the URL becomes
    unavailable, manually download from https://www.compression.cc/tasks/
    and place images in dest_dir.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    existing = list(dest.rglob("*.png")) + list(dest.rglob("*.jpg"))
    if len(existing) >= 100:
        print(f"CLIC dataset already has {len(existing)} images at {dest}")
        return

    url = "https://data.vision.ee.ethz.ch/cvl/clic/professional_train_2020.zip"
    zip_path = dest / "clic_train.zip"

    print("Downloading CLIC 2020 training set...")
    print(f"URL: {url}")
    print("This may take a while (~2GB)...")

    try:
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest)
        zip_path.unlink()
        existing = list(dest.rglob("*.png")) + list(dest.rglob("*.jpg"))
        print(f"CLIC dataset ready at {dest} ({len(existing)} images)")
    except Exception as e:
        print(f"Auto-download failed: {e}")
        print("Please manually download CLIC 2020 professional training images from:")
        print("  https://www.compression.cc/tasks/")
        print(f"  and place them in: {dest}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--dataset", choices=["kodak", "clic", "all"], default="all")
    args = parser.parse_args()

    if args.dataset in ("kodak", "all"):
        download_kodak(os.path.join(args.data_dir, "kodak"))
    if args.dataset in ("clic", "all"):
        download_clic(os.path.join(args.data_dir, "clic"))
