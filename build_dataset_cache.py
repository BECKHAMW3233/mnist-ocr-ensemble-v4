#!/usr/bin/env python3
"""
build_dataset_cache.py
=======================
Standalone script, run once by William on his own system. Precomputes and
saves to disk the deterministic (non-random) preprocessing for every real
dataset source this project uses: decode + orientation-correction (EMNIST-
family sources only, undoing a real torchvision bug) at native resolution,
plus upscaled copies at every resolution tier that source is actually used
at. Random augmentation (RandomRotation/RandomAffine/ColorJitter/
GaussianBlur) is never cached — it has to run fresh every access by
definition, so it isn't part of what this script produces.

Cache location: DATA_DIR / "_precomputed_cache" / "<source>_<tag>.npz"
  <source>_native.npz    — decoded (+ orientation-corrected), original size
  <source>_<res>.npz     — same, resized to that resolution (res in 16/28/32/64/128)

Only the *_native.npz files are actually read back during training (see the
matching supplementary_data.py change) — that's what keeps the transform
pipeline's order (augment, then resize) exactly as it is today. The
per-resolution upscaled files are produced because they were explicitly
asked for; they are precomputed deterministic artifacts sitting on disk,
not something the live training path loads.

Sources covered, and the resolutions each is actually used at in this
project (confirmed by reading digit_sources_for_tier() and each calling
script, not assumed):
    mnist            (supplementary MNIST copy, via MNISTDataset): 28,32,64,128
    emnist_digits                                                : 28,32,64,128
    usps             (native 16x16; supplementary at 32/64/128, not 28): 16,32,64,128
    svhn                                                          : 28,32,64,128
    ardis                                                         : 28,32,64,128
    emnist_balanced  (router only, all 5 router resolutions)      : 16,28,32,64,128
    emnist_byclass   (letter models; no 16x16 letter tier)        : 28,32,64,128

Usage:
    python build_dataset_cache.py              # build everything listed above
    python build_dataset_cache.py --list        # show what would be built, do nothing
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import EMNIST, MNIST, USPS, SVHN

from supplementary_data import (
    DATA_DIR, BALANCED_TO_BYCLASS, _correct_emnist_orientation, ARDISDataset,
)

CACHE_DIR = Path(__file__).resolve().parent / "_precomputed_cache"

# (tag, resolutions) — matches supplementary_data.py's per-source cache tags exactly
SOURCES = {
    "mnist":           [28, 32, 64, 128],
    "emnist_digits":   [28, 32, 64, 128],
    "usps":            [16, 32, 64, 128],
    "svhn":            [28, 32, 64, 128],
    "ardis":           [28, 32, 64, 128],
    "emnist_balanced": [16, 28, 32, 64, 128],
    "emnist_byclass":  [28, 32, 64, 128],
}


def _save(tag: str, suffix: str, images: np.ndarray, labels: np.ndarray):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{tag}_{suffix}.npz"
    np.savez(str(path), images=images, labels=labels)
    print(f"  [SAVED] {path}  ({len(labels):,} samples, {images.nbytes / 1024**2:.1f} MB)")


def _build_native(tag: str) -> tuple:
    """Decode (+ orientation-correct where applicable) once, at native resolution."""
    if tag == "mnist":
        base = MNIST(root=str(DATA_DIR), train=True, download=True, transform=None)
        imgs = np.stack([np.array(img, dtype=np.uint8) for img, _ in base])
        labels = np.array([int(label) for _, label in base], dtype=np.int64)

    elif tag == "emnist_digits":
        base = EMNIST(root=str(DATA_DIR), split="digits", train=True, download=True, transform=None)
        imgs, labels = [], []
        for img, label in base:
            img = _correct_emnist_orientation(img)
            imgs.append(np.array(img, dtype=np.uint8))
            labels.append(int(label))
        imgs = np.stack(imgs)
        labels = np.array(labels, dtype=np.int64)

    elif tag == "usps":
        base = USPS(root=str(DATA_DIR), train=True, download=True, transform=None)
        imgs, labels = [], []
        for img, label in base:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            imgs.append(np.array(img, dtype=np.uint8))
            labels.append(int(label))
        imgs = np.stack(imgs)
        labels = np.array(labels, dtype=np.int64)

    elif tag == "svhn":
        base = SVHN(root=str(DATA_DIR), split="train", download=True, transform=None)
        imgs, labels = [], []
        for i in range(len(base)):
            img, _ = base[i]
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = img.convert("L")
            imgs.append(np.array(img, dtype=np.uint8))
            labels.append(int(base.labels[i]) % 10)
        imgs = np.stack(imgs)
        labels = np.array(labels, dtype=np.int64)

    elif tag == "ardis":
        ds = ARDISDataset(transform=None)  # already loads from its own cached .npy internally
        imgs = ds.images.astype(np.uint8)
        labels = np.array(ds.remapped_labels, dtype=np.int64)

    elif tag == "emnist_balanced":
        base = EMNIST(root=str(DATA_DIR), split="balanced", train=True, download=True, transform=None)
        imgs, labels = [], []
        for img, label in base:
            label = int(label)
            if label in BALANCED_TO_BYCLASS:
                img = _correct_emnist_orientation(img)
                imgs.append(np.array(img, dtype=np.uint8))
                labels.append(BALANCED_TO_BYCLASS[label])
        imgs = np.stack(imgs)
        labels = np.array(labels, dtype=np.int64)

    elif tag == "emnist_byclass":
        base = EMNIST(root=str(DATA_DIR), split="byclass", train=True, download=True, transform=None)
        imgs, labels = [], []
        for img, label in base:
            img = _correct_emnist_orientation(img)
            imgs.append(np.array(img, dtype=np.uint8))
            labels.append(int(label))
        imgs = np.stack(imgs)
        labels = np.array(labels, dtype=np.int64)

    else:
        raise ValueError(f"Unknown source tag: {tag}")

    return imgs, labels


def _resize_all(images: np.ndarray, resolution: int) -> np.ndarray:
    """Deterministic resize only — no augmentation involved, matches Resize((res,res))."""
    out = np.empty((len(images), resolution, resolution), dtype=np.uint8)
    for i, arr in enumerate(images):
        img = Image.fromarray(arr, mode="L").resize((resolution, resolution))
        out[i] = np.array(img, dtype=np.uint8)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="Show what would be built, do nothing")
    args = parser.parse_args()

    if args.list:
        for tag, resolutions in SOURCES.items():
            print(f"{tag}: native + {resolutions}")
        return

    for tag, resolutions in SOURCES.items():
        print(f"=== {tag} ===")
        native_path = CACHE_DIR / f"{tag}_native.npz"
        if native_path.exists():
            data = np.load(str(native_path))
            images, labels = data["images"], data["labels"]
            print(f"  [SKIP] native already cached ({len(labels):,} samples)")
        else:
            images, labels = _build_native(tag)
            _save(tag, "native", images, labels)

        for res in resolutions:
            res_path = CACHE_DIR / f"{tag}_{res}.npz"
            if res_path.exists():
                print(f"  [SKIP] {res}x{res} already cached")
                continue
            resized = _resize_all(images, res)
            _save(tag, str(res), resized, labels)
        print()

    print("Done. Cache directory:", CACHE_DIR)


if __name__ == "__main__":
    main()
