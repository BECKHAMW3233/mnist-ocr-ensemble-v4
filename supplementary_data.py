"""
supplementary_data.py
=====================
Shared supplementary dataset loader — v4. Extends the v2 module of the
same name; the loading pattern (per-source Dataset wrapper classes,
graceful skip-on-missing, WeightedRandomSampler-friendly label
extraction) is unchanged, not replaced. Four additions for v3:

  1. digit_sources_for_tier(img_size) — the per-source resolution-ladder
     split (see Part 1 of this restructure / v3_CHANGELOG.md): MNIST, EMNIST
     Digits, SVHN, and ARDIS IV ladder 28->32->64->128; USPS ladders
     16->32->64->128 (its own native resolution, skipping 28x28 — not
     native or meaningfully closer to native for USPS than either of its
     real neighbors). All five sources are present at 32/64/128; only the
     28x28 tier excludes USPS. This function is the single place that
     encodes that split so every digit training script and the router
     stay consistent with each other by construction rather than by each
     one re-deriving it.

  2. BALANCED_TO_BYCLASS now correctly covers EMNIST Balanced's full
     47-class label space (previously only 0-35 — see the comment above
     the dict below). This was found while re-enabling EMNIST Balanced
     as the v3 router's letter source (Part 3) — BalancedEMNISTDataset
     itself is otherwise unchanged.

  3. _correct_emnist_orientation() — torchvision's EMNIST dataset class
     (every split: digits, balanced, byclass, etc.) returns images
     rotated 90 degrees and mirrored relative to upright — a real,
     still-open torchvision bug (github.com/pytorch/vision/issues/8783),
     confirmed by reading torchvision 0.28.0's own mnist.py directly:
     EMNIST.__getitem__ inherits MNIST.__getitem__ unchanged, which
     applies zero rotation/flip correction. Applied in
     EMNISTDigitsDataset, BalancedEMNISTDataset, and the new
     EMNISTByClassDataset below, so training images match the upright
     orientation real scanned character crops have at inference time.
     Added 2026-08-02, per direct user follow-up after this was found
     while building the v3 letter-identity models (Part 4) — see
     v3_CHANGELOG.md.

  4. EMNISTByClassDataset / load_base_emnist_letters() — EMNIST ByClass
     (all 62 classes, not previously used anywhere in this project — only
     "digits" and "balanced" were), added as the sole data source for the
     v3 uppercase/lowercase letter-identity models
     (v4_mnist_letter_{uc,lc}_{optimizer}_{res}.py). Chosen over EMNIST
     Balanced specifically because Balanced only has 11 distinct lowercase
     classes (the other 15 are merged into their uppercase counterpart by
     the dataset's own creators — see BALANCED_TO_BYCLASS's comment
     above); ByClass has all 26. Raw byclass label integers already match
     this project's own 0-9/10-35/36-61 convention directly (confirmed by
     reading torchvision's own EMNIST.classes_split_dict) — no remapping
     bug, unlike BALANCED_TO_BYCLASS above.

Provides digit-only supplementary data sources for MNIST digit recognition:

  DIGIT-SPECIFIC (addresses systematic digit→letter misclassification):
  1. EMNIST Digits     — 280,000 samples, digits 0-9 only
  2. MNIST             — 70,000 samples, digits 0-9, clean centered
  3. USPS              — 9,298 samples, digits 0-9, scanned envelopes
  4. SVHN              — 73,257 samples, digits 0-9, street sign photos
  5. ARDIS IV          — 7,600 samples, digits 0-9, Swedish historical church records
                         Non-NIST writer population, 19th–20th century handwriting.
                         Auto-downloaded from the official ARDISDataset/ARDIS GitHub
                         repo on first use if not already present locally (see
                         ARDISDataset below) — no separate download script needed.
                         If auto-download fails (no network, missing extraction
                         tool, etc.), it prints manual download/setup instructions.

  LETTER DATASETS:
  6. EMNIST Balanced   — re-enabled in v3 as the router's UC/LC training
                         source (see load_supplementary()'s use_balanced
                         flag, still False by default — the DIGIT ensemble
                         stays digits-only; the router imports
                         BalancedEMNISTDataset directly instead, mirroring
                         how the v2 digit gate self-contained its own
                         letter-loading rather than changing this shared
                         module's defaults — see v4_mnist_router_ranger_*.py)
  7. Kaggle A-Z, Chars74K, PG-HWLD — still excluded (use_kaggle/
                         use_chars_hnd/use_chars_img/use_pghwld default
                         False); optional supplementary letter sources,
                         not required for the router per this restructure)
  8. EMNIST ByClass    — 814,255 samples across all 62 classes; the sole
                         source for the v3 letter-identity models
                         (uppercase/lowercase, 26 classes each — see
                         EMNISTByClassDataset / load_base_emnist_letters()
                         above). Not wired into load_supplementary() at
                         all — the letter scripts call
                         load_base_emnist_letters() directly, mirroring
                         how load_base_mnist()/load_base_usps() sit
                         outside load_supplementary() too.

Class index mapping (byclass, 62 classes):
    0-9   : digits 0-9
    10-35 : uppercase A-Z
    36-61 : lowercase a-z

Chars74K Sample folder mapping:
    Sample001-010 → digits 0-9     → byclass 0-9
    Sample011-036 → uppercase A-Z  → byclass 10-35
    Sample037-062 → lowercase a-z  → byclass 36-61

PG-HWLD folder mapping:
    Folder name = single uppercase letter (A, B, C ... Z)
    Maps directly to byclass 10-35 (A=10, B=11, ... Z=35)

Class balance rationale (digit ensemble only — NUM_CLASSES/DIGIT_BOOST
below describe load_supplementary()'s digits-only default use; the
router does its own 3-class digit/UC/LC weighting, see
v4_mnist_router_ranger_*.py):
    DIGIT_BOOST=1.0 (neutral). Weighting is pure inverse frequency across 10 classes.

    Raw digit samples (training-time total): ~388,148 (EMNIST Digits
    train 240,000 + MNIST train 60,000 + USPS train 7,291 + SVHN train
    73,257 + ARDIS IV 7,600 — train-split counts, not published dataset
    totals; see EMNISTDigitsDataset/USPSDataset docstrings for the
    corrected per-source splits).
    All sources map to class indices 0-9 directly.
"""

from pathlib import Path
from typing import Optional
import glob
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import numpy as np

import torch
from torch.utils.data import Dataset, ConcatDataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import EMNIST, MNIST, USPS, SVHN
from PIL import Image, ImageOps

# =============================================================================
# Paths — EDIT THESE FOR YOUR OWN SYSTEM
# =============================================================================
# Every constant below is a real, hardcoded location on the original
# project machine. None of them exist on a fresh checkout — before running
# anything that touches this file, point each one at wherever you want
# these datasets to actually live on YOUR disk. They don't need to match
# each other or follow this exact layout; each is used independently.

# DATA_DIR: root folder for the torchvision-managed sources (EMNIST Digits,
# base MNIST, USPS, SVHN, EMNIST Balanced). These auto-download here via
# torchvision's own download=True mechanism — you just need this to point
# at a folder that exists (it's created automatically if missing) and has
# enough free space (a few GB across all of them combined). EMNIST
# Balanced (re-enabled in v3 for the router — see module docstring) reuses
# this same DATA_DIR, same as its digit-only sibling EMNIST Digits; it
# does not get its own path constant.
DATA_DIR      = Path(r"E:\CSC-114\emnist-model\datasets\pytorch")

# KAGGLE_DIR / KAGGLE_IMGS / KAGGLE_LBLS: Kaggle A-Z letters dataset.
# NOT used by the digit ensemble (use_kaggle=False everywhere it's
# called) or by the v3 router (which uses EMNIST Balanced instead — see
# module docstring) — safe to leave pointed at a nonexistent path unless
# you specifically opt into this as a supplementary letter source, in
# which case note the path below is specific to the original machine
# and will need changing.
KAGGLE_DIR    = Path(r"E:\CSC-114\emnist-model\datasets\kaggle")
KAGGLE_IMGS   = KAGGLE_DIR / "az_images.npy"
KAGGLE_LBLS   = KAGGLE_DIR / "az_labels.npy"

# CHARS74K_HND / CHARS74K_IMG: Chars74K handwritten/image letter datasets.
# Also NOT used by the digit ensemble or the v3 router by default — same
# as above, safe to leave as-is unless you opt into this as a
# supplementary letter source — both paths below are specific to the
# original machine and will need changing if you do.
CHARS74K_HND  = Path(r"E:\CSC-114\emnist-model\datasets\EnglishHnd\English\Hnd\Img")
CHARS74K_IMG  = Path(r"E:\CSC-114\emnist-model\datasets\EnglishImg\English\Img\GoodImg\Bmp")

# ARDIS_DIR / ARDIS_IMGS / ARDIS_LBLS: Swedish historical digit dataset —
# ACTUALLY USED by every digit training script and the router
# (use_ardis=True). Point ARDIS_DIR at wherever you want it stored;
# ARDISDataset (below) will auto-download and prepare ARDIS_IMGS/
# ARDIS_LBLS the first time it's constructed if they aren't already there.
ARDIS_DIR     = Path(r"E:\CSC-114\emnist-model\datasets\ardis")
ARDIS_IMGS    = ARDIS_DIR / "ardis_images.npy"
ARDIS_LBLS    = ARDIS_DIR / "ardis_labels.npy"

# PGHWLD_DIR: Pashto handwritten letters dataset. NOT used by the digit
# ensemble or the v3 router by default (use_pghwld=False everywhere) —
# safe to leave as-is unless you opt into this as a supplementary letter
# source, in which case the path below is specific to the original
# machine and will need changing.
PGHWLD_DIR    = Path(r"E:\CSC-114\emnist-model\datasets\pg_hwld")

NUM_CLASSES = 10

# DIGIT_BOOST: no letter classes to counterbalance against in the digit
# ensemble's own 10-class training (the router handles digit-vs-UC-vs-LC
# balance separately — see v4_mnist_router_ranger_*.py). Set to 1.0 —
# pure inverse frequency weighting across the 10 digit classes.
DIGIT_BOOST = 1.0

# Index mappings
#
# EMNIST Balanced has 47 classes: digits 0-9, uppercase A-Z (36 classes
# so far), plus ONLY the 11 lowercase letters NOT visually merged with
# their uppercase counterpart by the dataset's own creators — the other
# 15 lowercase letters (c, i, j, k, l, m, o, p, s, u, v, w, x, y, z) have
# no EMNIST Balanced class at all. Per the dataset's own published
# emnist-balanced-mapping.txt, EMNIST Balanced indices 36-46 are, in
# order: a, b, d, e, f, g, h, n, q, r, t.
#
# The v2 version of this dict only mapped indices 0-35 (digits +
# uppercase) — BalancedEMNISTDataset's own `if label in
# BALANCED_TO_BYCLASS` filter silently dropped every lowercase sample as
# a result, with no error or warning (BalancedEMNISTDataset was never
# actually used anywhere in v2 — use_balanced was False at every call
# site — so this had no behavioral effect until now). Found while
# re-enabling EMNIST Balanced as the v3 router's letter source (Part 3),
# which genuinely needs lowercase representation; fixed here so
# BalancedEMNISTDataset returns complete, correct byclass labels (0-61)
# for all 47 of its actual classes, matching CHARS74K_TO_BYCLASS's own
# already-correct byclass 36-61 = a-z convention below.
_EMNIST_BALANCED_LOWERCASE = "abdefghnqrt"  # EMNIST Balanced indices 36-46, in order
BALANCED_TO_BYCLASS = {
    **{i: i for i in range(10)},              # digits 0-9      -> byclass 0-9
    **{10 + i: 10 + i for i in range(26)},    # uppercase A-Z   -> byclass 10-35
    **{36 + i: 36 + (ord(ch) - ord('a'))      # the 11 distinct lowercase classes
       for i, ch in enumerate(_EMNIST_BALANCED_LOWERCASE)},  # -> byclass 36-61
}
KAGGLE_TO_BYCLASS   = {i: i + 10 for i in range(26)}
CHARS74K_TO_BYCLASS = {f"Sample{i+1:03d}": i for i in range(62)}
# Digits 0-9 map directly to byclass 0-9 for MNIST, USPS, SVHN, EMNIST Digits
DIGIT_TO_BYCLASS    = {i: i for i in range(10)}


# =============================================================================
# Native-resolution disk cache (2026-08-08, per direct user follow-up)
# =============================================================================
# Reads the native-resolution cache files build_dataset_cache.py produces
# (William runs that script separately — see its own docstring). Only the
# *_native.npz file is ever read here; the per-resolution upscaled files
# that script also produces are not read here, since applying them here
# would mean resize happens before augmentation instead of after —
# explicitly ruled out. If no cache exists yet (script hasn't been run),
# each class below falls back to its original from-scratch behavior
# unchanged — this is purely opportunistic, not a requirement.

_CACHE_DIR = Path(__file__).resolve().parent / "_precomputed_cache"


def _load_native_cache(tag: str):
    """Returns (images, labels) numpy arrays from the native-resolution
    cache for this tag, or None if it hasn't been built yet."""
    path = _CACHE_DIR / f"{tag}_native.npz"
    if not path.exists():
        return None
    data = np.load(str(path))
    return data["images"], data["labels"]


# =============================================================================
# Dataset wrappers
# =============================================================================

def _correct_emnist_orientation(img):
    """
    torchvision's EMNIST dataset class (every split — digits, balanced,
    byclass, etc.) returns images rotated 90 degrees and mirrored relative
    to upright. This is a real, still-open torchvision bug
    (github.com/pytorch/vision/issues/8783), confirmed by reading
    torchvision 0.28.0's own mnist.py directly: EMNIST.__getitem__
    inherits MNIST.__getitem__ unchanged, which applies zero rotation/flip
    correction before handing back the PIL image. Undoes it so training
    images match the upright orientation real scanned character crops
    have at inference time. Added 2026-08-02, per direct user follow-up —
    see v3_CHANGELOG.md.
    """
    return ImageOps.mirror(img.rotate(-90, expand=True))


class EMNISTDigitsDataset(Dataset):
    """
    EMNIST Digits split — 240,000 training + 40,000 test samples, digits 0-9.
    Digits 0-9 map directly to byclass indices 0-9.
    """
    def __init__(self, train: bool = True, transform=None):
        self.transform = transform
        cached = _load_native_cache("emnist_digits") if train else None
        if cached is not None:
            self.images, labels_arr = cached
            self.remapped_labels = labels_arr.tolist()
            self.base = None
        else:
            self.base = EMNIST(
                root=str(DATA_DIR), split="digits", train=train,
                download=True, transform=None,
            )
            self.images = None
            self.remapped_labels = [int(label) for _, label in self.base]

    def __len__(self):
        return len(self.images) if self.images is not None else len(self.base)

    def __getitem__(self, idx):
        if self.images is not None:
            img = Image.fromarray(self.images[idx], mode="L")
        else:
            img, _ = self.base[idx]
            img = _correct_emnist_orientation(img)
        label = self.remapped_labels[idx]
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label

    def get_labels(self):
        return self.remapped_labels


class MNISTDataset(Dataset):
    """
    MNIST — 70,000 samples, digits 0-9, clean centered 28x28.
    Different writer pool than EMNIST — additional stroke variation.
    Digits 0-9 map directly to byclass indices 0-9.
    """
    def __init__(self, train: bool = True, transform=None):
        self.transform = transform
        cached = _load_native_cache("mnist") if train else None
        if cached is not None:
            self.images, labels_arr = cached
            self.remapped_labels = labels_arr.tolist()
            self.base = None
        else:
            self.base = MNIST(
                root=str(DATA_DIR), train=train,
                download=True, transform=None,
            )
            self.images = None
            self.remapped_labels = [int(label) for _, label in self.base]

    def __len__(self):
        return len(self.images) if self.images is not None else len(self.base)

    def __getitem__(self, idx):
        if self.images is not None:
            img = Image.fromarray(self.images[idx], mode="L")
        else:
            img, _ = self.base[idx]
        label = self.remapped_labels[idx]
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label

    def get_labels(self):
        return self.remapped_labels


class USPSDataset(Dataset):
    """
    USPS — 7,291 training + 2,007 test samples, digits 0-9.
    Scanned from US Postal Service envelopes — real-world handwriting,
    different stroke characteristics than EMNIST/MNIST. Native resolution
    16x16 — see digit_sources_for_tier() below for why this source is
    absent from the 28x28 resolution tier specifically.
    Digits 0-9 map directly to byclass indices 0-9.
    """
    def __init__(self, train: bool = True, transform=None):
        self.transform = transform
        cached = _load_native_cache("usps") if train else None
        if cached is not None:
            self.images, labels_arr = cached
            self.remapped_labels = labels_arr.tolist()
            self.base = None
        else:
            self.base = USPS(
                root=str(DATA_DIR), train=train,
                download=True, transform=None,
            )
            self.images = None
            self.remapped_labels = [int(label) for _, label in self.base]

    def __len__(self):
        return len(self.images) if self.images is not None else len(self.base)

    def __getitem__(self, idx):
        if self.images is not None:
            img = Image.fromarray(self.images[idx], mode="L")
        else:
            img, _ = self.base[idx]
            # USPS returns numpy arrays — convert to PIL
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
        label = self.remapped_labels[idx]
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label

    def get_labels(self):
        return self.remapped_labels


class SVHNDataset(Dataset):
    """
    SVHN (Street View House Numbers) — 73,257 training samples, digits 0-9.
    Photographed from real street signs — very different domain to EMNIST.
    Adds real-world domain shift robustness to digit recognition.
    Digits 0-9 map directly to byclass indices 0-9.
    Note: SVHN labels are 1-10 (10=0) — remapped to 0-9 byclass indices.
    """
    def __init__(self, train: bool = True, transform=None):
        split = "train" if train else "test"
        self.transform = transform
        cached = _load_native_cache("svhn") if train else None
        if cached is not None:
            self.images, labels_arr = cached
            self.remapped_labels = labels_arr.tolist()
            self.base = None
        else:
            self.base = SVHN(
                root=str(DATA_DIR), split=split,
                download=True, transform=None,
            )
            self.images = None
            # SVHN uses label 10 for digit 0 — remap to standard 0-9
            self.remapped_labels = [int(label) % 10 for label in self.base.labels]

    def __len__(self):
        return len(self.images) if self.images is not None else len(self.base)

    def __getitem__(self, idx):
        if self.images is not None:
            img = Image.fromarray(self.images[idx], mode="L")
        else:
            img, _ = self.base[idx]
            # SVHN returns RGB — convert to grayscale to match other datasets
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = img.convert("L")
        label = self.remapped_labels[idx]
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label

    def get_labels(self):
        return self.remapped_labels


class KaggleAZDataset(Dataset):
    """Kaggle A-Z — uppercase only (byclass 10-35)."""
    def __init__(self, transform=None):
        if not KAGGLE_IMGS.exists() or not KAGGLE_LBLS.exists():
            raise FileNotFoundError("Kaggle data not found. Run download_datasets.py first.")
        self.images    = np.load(str(KAGGLE_IMGS))
        raw_labels     = np.load(str(KAGGLE_LBLS))
        self.labels    = np.array([KAGGLE_TO_BYCLASS[l] for l in raw_labels], dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img   = self.images[idx]
        label = int(self.labels[idx])
        pil   = Image.fromarray(img, mode="L")
        if self.transform:
            img_t = self.transform(pil)
        else:
            img_t = transforms.ToTensor()(pil)
        return img_t, label


class BalancedEMNISTDataset(Dataset):
    """
    EMNIST Balanced — 47 classes remapped to byclass indices (0-9 digits,
    10-35 uppercase, 36-61 the 11 EMNIST-Balanced-distinct lowercase
    letters — see BALANCED_TO_BYCLASS above). Re-enabled in v3 as the
    router's letter source (both uppercase and lowercase); the digit
    ensemble does not use this class (use_balanced stays False in every
    load_supplementary() call the digit ensemble makes).
    """
    def __init__(self, train: bool = True, transform=None):
        self.transform = transform
        cached = _load_native_cache("emnist_balanced") if train else None
        if cached is not None:
            self.images, labels_arr = cached
            self.remapped_labels = labels_arr.tolist()
            self.base = None
            self.valid_indices = None
        else:
            self.base = EMNIST(
                root=str(DATA_DIR), split="balanced", train=train,
                download=True, transform=None,
            )
            self.images = None
            valid_indices   = []
            remapped_labels = []
            for i, (_, label) in enumerate(self.base):
                label = int(label)
                if label in BALANCED_TO_BYCLASS:
                    valid_indices.append(i)
                    remapped_labels.append(BALANCED_TO_BYCLASS[label])
            self.valid_indices   = valid_indices
            self.remapped_labels = remapped_labels

    def __len__(self):
        return len(self.images) if self.images is not None else len(self.valid_indices)

    def __getitem__(self, idx):
        if self.images is not None:
            img = Image.fromarray(self.images[idx], mode="L")
        else:
            orig_idx = self.valid_indices[idx]
            img, _   = self.base[orig_idx]
            img      = _correct_emnist_orientation(img)
        label = self.remapped_labels[idx]
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label

    def get_labels(self):
        return self.remapped_labels


class EMNISTByClassDataset(Dataset):
    """
    EMNIST ByClass — all 62 classes (0-9 digits, 10-35 uppercase, 36-61
    lowercase). Not used by the digit ensemble or the router — added in
    v3 as the sole data source for the uppercase/lowercase letter-identity
    models (v4_mnist_letter_{uc,lc}_{optimizer}_{res}.py). Chosen over
    EMNIST Balanced specifically because Balanced only has 11 distinct
    lowercase classes (see BALANCED_TO_BYCLASS's comment above); ByClass
    has real, distinct labels for all 26. Raw byclass label integers
    already match this project's own 0-9/10-35/36-61 convention directly
    (confirmed by reading torchvision's own EMNIST.classes_split_dict) —
    no remapping needed, unlike BALANCED_TO_BYCLASS.
    """
    def __init__(self, train: bool = True, transform=None):
        self.transform = transform
        cached = _load_native_cache("emnist_byclass") if train else None
        if cached is not None:
            self.images, labels_arr = cached
            self.remapped_labels = labels_arr.tolist()
            self.base = None
        else:
            self.base = EMNIST(
                root=str(DATA_DIR), split="byclass", train=train,
                download=True, transform=None,
            )
            self.images = None
            self.remapped_labels = [int(label) for _, label in self.base]

    def __len__(self):
        return len(self.images) if self.images is not None else len(self.base)

    def __getitem__(self, idx):
        if self.images is not None:
            img = Image.fromarray(self.images[idx], mode="L")
        else:
            img, _ = self.base[idx]
            img = _correct_emnist_orientation(img)
        label = self.remapped_labels[idx]
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label

    def get_labels(self):
        return self.remapped_labels


class _LetterOnlyDataset(Dataset):
    """
    Wraps EMNISTByClassDataset, keeping only one case's byclass range
    (10-35 uppercase / 36-61 lowercase) and remapping to a dense 0-25
    label for a 26-class classifier head (A=0..Z=25 or a=0..z=25).
    """
    def __init__(self, base: "EMNISTByClassDataset", case: str):
        offset = {"upper": 10, "lower": 36}[case]
        self.base = base
        self.indices = [i for i, lbl in enumerate(base.remapped_labels)
                         if offset <= lbl < offset + 26]
        self.dense_labels = [base.remapped_labels[i] - offset for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, _ = self.base[self.indices[idx]]
        return img, self.dense_labels[idx]

    def get_labels(self):
        return self.dense_labels


class Chars74KDataset(Dataset):
    """
    Chars74K dataset loader — works for both EnglishHnd and EnglishImg.
    Maps Sample001-062 directly to EMNIST byclass indices 0-61.
    Covers all 62 classes including both upper AND lowercase.
    """
    def __init__(self, root: Path, transform=None):
        self.transform = transform
        self.samples   = []

        if not root.exists():
            raise FileNotFoundError(f"Chars74K not found at {root}")

        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            folder_name = folder.name
            if folder_name not in CHARS74K_TO_BYCLASS:
                continue
            label = CHARS74K_TO_BYCLASS[folder_name]
            for ext in ("*.png", "*.jpg", "*.bmp"):
                for img_path in sorted(folder.glob(ext)):
                    self.samples.append((img_path, label))

        if not self.samples:
            raise FileNotFoundError(f"No images found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label

    def get_labels(self):
        return [s[1] for s in self.samples]


# =============================================================================
# ARDIS IV — auto-download and preparation
# =============================================================================
# ARDIS_DATASET_IV.rar is the official, author-published archive from the
# ARDIS paper (Kusetogullari et al., "ARDIS: a Swedish historical
# handwritten digit dataset", Neural Computing and Applications, 2019),
# hosted directly in the authors' own GitHub repo. Confirmed present at
# this exact path via that repo's file listing.
_ARDIS_RAR_URL = "https://github.com/ARDISDataset/ARDIS/raw/master/ARDIS_DATASET_IV.rar"

# Inside the archive, per the dataset's own documented usage example
# (np.loadtxt on each file, then reshape to 28x28), the four files are:
#   ARDIS_train_2828.csv, ARDIS_train_labels.csv   (6,600 samples)
#   ARDIS_test_2828.csv,  ARDIS_test_labels.csv    (1,000 samples)
# combined here into one 7,600-sample supplementary source (train/test
# split for THIS project's own MNIST+supplementary data happens later,
# in load_base_mnist()/the calling script — ARDIS isn't pre-split there).

_ARDIS_MANUAL_INSTRUCTIONS = f"""
[ARDIS] Automatic download/setup failed. To set it up manually:
  1. Download: {_ARDIS_RAR_URL}
  2. Extract ARDIS_DATASET_IV.rar (Windows: 7-Zip or WinRAR both open
     .rar files; Linux/Mac: `unrar x` or `unar`) — inside you'll find
     ARDIS_train_2828.csv, ARDIS_train_labels.csv, ARDIS_test_2828.csv,
     ARDIS_test_labels.csv.
  3. Run this in a Python shell, with ARDIS_DIR pointed at where you
     want it stored permanently (matching this file's ARDIS_DIR):
         import numpy as np
         from pathlib import Path
         extracted = Path(r"<folder you extracted the csvs into>")
         train_x = np.loadtxt(extracted / "ARDIS_train_2828.csv")
         test_x  = np.loadtxt(extracted / "ARDIS_test_2828.csv")
         train_y = np.loadtxt(extracted / "ARDIS_train_labels.csv")
         test_y  = np.loadtxt(extracted / "ARDIS_test_labels.csv")
         if train_y.ndim == 2:  # one-hot encoded — convert to plain 0-9
             train_y = train_y.argmax(axis=1)
             test_y  = test_y.argmax(axis=1)
         images = np.concatenate([train_x, test_x]).reshape(-1, 28, 28).astype("uint8")
         labels = np.concatenate([train_y, test_y]).astype("int64")
         ardis_dir = Path(r"{ARDIS_DIR}")
         ardis_dir.mkdir(parents=True, exist_ok=True)
         np.save(ardis_dir / "ardis_images.npy", images)
         np.save(ardis_dir / "ardis_labels.npy", labels)
  Training will run without ARDIS (it's one of five digit sources) until
  this is done — it isn't required, just recommended for full coverage.
""".strip()


def _try_extract_rar(rar_path: Path, dest_dir: Path) -> bool:
    """
    Tries every extraction method likely to be available, in order, and
    returns True on the first success. No single tool is universally
    installed for .rar specifically (unlike .zip, which stdlib handles
    natively) — 7-Zip is the most common on Windows, unrar/unar on
    Linux/Mac, and the `rarfile` pip package is tried last since it still
    needs one of the same external binaries as a backend.
    """
    for exe, args in [
        ("7z",     ["x", "-y", f"-o{dest_dir}", str(rar_path)]),
        ("7za",    ["x", "-y", f"-o{dest_dir}", str(rar_path)]),
        ("unrar",  ["x", "-y", str(rar_path), str(dest_dir) + "/"]),
        ("unar",   ["-o", str(dest_dir), str(rar_path)]),
    ]:
        if shutil.which(exe) is None:
            continue
        try:
            subprocess.run([exe, *args], capture_output=True, timeout=120, check=True)
            return True
        except (subprocess.SubprocessError, OSError):
            continue
    try:
        import rarfile  # optional — still needs unrar/unar/7z as its own backend
        with rarfile.RarFile(str(rar_path)) as rf:
            rf.extractall(str(dest_dir))
        return True
    except Exception:
        return False


def _prepare_ardis_npy() -> bool:
    """
    Downloads and prepares ARDIS_IMGS/ARDIS_LBLS if they don't already
    exist. Returns True if they're ready to load afterward (either
    already present, or successfully prepared just now), False if
    preparation failed for any reason — network, missing extraction
    tool, unexpected archive contents, etc. Never raises: a failure here
    just means ARDISDataset's caller falls through to its existing
    FileNotFoundError, with manual instructions already printed.
    """
    if ARDIS_IMGS.exists() and ARDIS_LBLS.exists():
        return True

    print("[ARDIS] Not found locally — attempting one-time automatic download...")
    tmp_dir = Path(tempfile.mkdtemp(prefix="ardis_dl_"))
    try:
        rar_path = tmp_dir / "ARDIS_DATASET_IV.rar"
        try:
            urllib.request.urlretrieve(_ARDIS_RAR_URL, str(rar_path))
        except (urllib.error.URLError, OSError) as e:
            print(f"[ARDIS] Download failed ({e}).")
            print(_ARDIS_MANUAL_INSTRUCTIONS)
            return False

        extract_dir = tmp_dir / "extracted"
        extract_dir.mkdir()
        if not _try_extract_rar(rar_path, extract_dir):
            print("[ARDIS] Could not extract the .rar archive — no supported "
                  "extraction tool (7-Zip, unrar, unar) found on PATH.")
            print(_ARDIS_MANUAL_INSTRUCTIONS)
            return False

        def _find(patterns) -> Optional[Path]:
            for pat in patterns:
                hits = glob.glob(str(extract_dir / "**" / pat), recursive=True)
                if hits:
                    return Path(hits[0])
            return None

        train_x_path = _find(("ARDIS_train_2828.csv",))
        test_x_path  = _find(("ARDIS_test_2828.csv",))
        train_y_path = _find(("ARDIS_train_labels.csv",))
        test_y_path  = _find(("ARDIS_test_labels.csv",))
        if not all([train_x_path, test_x_path, train_y_path, test_y_path]):
            print("[ARDIS] Extracted archive didn't contain the expected CSV "
                  "filenames (ARDIS_{train,test}_2828.csv / "
                  "ARDIS_{train,test}_labels.csv) — archive contents may "
                  "have changed upstream.")
            print(_ARDIS_MANUAL_INSTRUCTIONS)
            return False

        train_x = np.loadtxt(str(train_x_path))
        test_x  = np.loadtxt(str(test_x_path))
        train_y = np.loadtxt(str(train_y_path))
        test_y  = np.loadtxt(str(test_y_path))

        # Label files are one-hot encoded in some ARDIS distributions and
        # plain integers in others — detect and normalize rather than
        # assume, since guessing wrong would silently corrupt every label.
        if train_y.ndim == 2:
            train_y = train_y.argmax(axis=1)
            test_y = test_y.argmax(axis=1)

        images = np.concatenate([train_x, test_x]).reshape(-1, 28, 28).astype(np.uint8)
        labels = np.concatenate([train_y, test_y]).astype(np.int64)

        ARDIS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(ARDIS_IMGS), images)
        np.save(str(ARDIS_LBLS), labels)
        print(f"[ARDIS] Prepared {len(images):,} samples → {ARDIS_DIR}")
        return True
    except Exception as e:
        print(f"[ARDIS] Automatic setup failed unexpectedly: {e}")
        print(_ARDIS_MANUAL_INSTRUCTIONS)
        return False
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)


class ARDISDataset(Dataset):
    """
    ARDIS IV — 7,600 handwritten digit images from Swedish church records.
    19th–20th century writer population, fully independent of NIST.
    28x28 grayscale MNIST-format. Digits 0-9 → byclass indices 0-9.
    Auto-downloads and prepares itself on first use if not already present
    locally (see _prepare_ardis_npy() above) — falls back to printing
    manual setup instructions if that fails for any reason.
    """
    def __init__(self, transform=None):
        _prepare_ardis_npy()
        if not ARDIS_IMGS.exists() or not ARDIS_LBLS.exists():
            raise FileNotFoundError(
                f"ARDIS IV not found at {ARDIS_DIR} and automatic setup "
                "did not complete — see the printed instructions above."
            )
        self.images          = np.load(str(ARDIS_IMGS))  # (N, 28, 28) uint8
        raw_labels           = np.load(str(ARDIS_LBLS))  # (N,) int64 0-9
        self.remapped_labels = [int(l) for l in raw_labels]  # digits → byclass 0-9 directly
        self.transform       = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img   = self.images[idx]
        label = self.remapped_labels[idx]
        pil   = Image.fromarray(img, mode="L")
        if self.transform:
            img_t = self.transform(pil)
        else:
            img_t = transforms.ToTensor()(pil)
        return img_t, label

    def get_labels(self):
        return self.remapped_labels


class PGHWLDDataset(Dataset):
    """
    PG-HWLD (Gdansk Tech Handwritten Letters Dataset) — 17,160 samples.
    26 uppercase letter classes (A-Z), 660 samples per class.
    Fully independent of NIST SD-19 — unique writer population and
    collection procedure. Maps to byclass indices 10-35 (uppercase A-Z).

    Expected content: exactly 17,160 total images across 26 single-letter
    (A-Z) folders, found ANYWHERE under PGHWLD_DIR at any nesting depth —
    e.g. both of these layouts are supported:
        PGHWLD_DIR/A/*.png                          (flat)
        PGHWLD_DIR/train-images/A/*.png             (nested under a split folder)
        PGHWLD_DIR/test-images/A/*.png
    If train-images/A and test-images/A both exist, their contents are
    pooled together as one class (this dataset has no standard train/test
    split).
    """
    # Map uppercase letter folder names to byclass indices 10-35
    PGHWLD_TO_BYCLASS = {chr(ord('A') + i): 10 + i for i in range(26)}

    def __init__(self, transform=None):
        self.transform = transform
        self.samples   = []  # list of (Path, byclass_label)

        if not PGHWLD_DIR.exists():
            raise FileNotFoundError(
                f"PG-HWLD not found at {PGHWLD_DIR}. "
                "See this project's README for manual download instructions."
            )

        # Recurse through every subdirectory at any depth, not just top-level,
        # and collect any folder whose name is exactly one of A-Z (case-insensitive).
        seen_paths = set()  # guard against the same file being matched twice
        for folder in sorted(PGHWLD_DIR.rglob("*")):
            if not folder.is_dir():
                continue
            folder_name = folder.name.upper()
            if folder_name not in self.PGHWLD_TO_BYCLASS:
                continue
            label = self.PGHWLD_TO_BYCLASS[folder_name]
            for ext in ("*.png", "*.jpg", "*.bmp", "*.PNG", "*.JPG"):
                for img_path in sorted(folder.glob(ext)):
                    resolved = img_path.resolve()
                    if resolved in seen_paths:
                        continue  # skip exact duplicate path (defensive, shouldn't normally trigger)
                    seen_paths.add(resolved)
                    self.samples.append((img_path, label))

        if not self.samples:
            raise FileNotFoundError(
                f"No images found in {PGHWLD_DIR}. "
                "Check folder structure: pg_hwld/A/*.png, pg_hwld/B/*.png ... "
                "(letter folders can be nested under a split folder, e.g. "
                "pg_hwld/train-images/A/*.png)"
            )

        actual_count = len(self.samples)
        if actual_count != 17160:
            print(f"  [PG-HWLD WARNING] Loaded {actual_count:,} samples, "
                  f"expected 17,160. Check for duplicate folders or an "
                  f"incomplete extraction before trusting this run's results.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label

    def get_labels(self):
        return [s[1] for s in self.samples]


# =============================================================================
# get_class_weights
# =============================================================================

def get_class_weights(dataset) -> torch.Tensor:
    """
    Compute WeightedRandomSampler weights for any dataset type.
    Applies DIGIT_BOOST multiplier to digit classes after inverse frequency weighting.
    For digits-only training DIGIT_BOOST=1.0 (no boost needed).
    """
    print("[Dataset] Computing class weights for balanced sampling...")
    targets       = _extract_targets(dataset)
    class_counts  = torch.bincount(targets, minlength=NUM_CLASSES).float()
    class_counts  = torch.clamp(class_counts, min=1)
    class_weights = 1.0 / class_counts

    print(f"[Dataset] Applying DIGIT_BOOST={DIGIT_BOOST}x weighting (1.0 = inverse frequency only)")
    class_weights[:10] *= DIGIT_BOOST

    sample_weights = class_weights[targets]
    print(f"[Dataset] Class weight range:  {class_weights.min():.6f} — {class_weights.max():.6f}")
    print(f"[Dataset] Digit weight range:  {class_weights[:10].min():.6f} — {class_weights[:10].max():.6f}")
    return sample_weights


def _extract_targets(dataset) -> torch.Tensor:
    """Recursively extract all labels from any dataset structure."""
    if isinstance(dataset, ConcatDataset):
        return torch.cat([_extract_targets(ds) for ds in dataset.datasets])

    if hasattr(dataset, 'indices') and hasattr(dataset, 'dataset'):
        parent = dataset.dataset
        if hasattr(parent, 'targets'):
            return torch.tensor(
                [int(parent.targets[i]) for i in dataset.indices], dtype=torch.long
            )
        return _extract_targets(parent)[list(dataset.indices)]

    if isinstance(dataset, (Chars74KDataset, PGHWLDDataset)):
        return torch.tensor(dataset.get_labels(), dtype=torch.long)

    if isinstance(dataset, KaggleAZDataset):
        return torch.tensor(dataset.labels.tolist(), dtype=torch.long)

    if isinstance(dataset, (EMNISTDigitsDataset, MNISTDataset, USPSDataset,
                             SVHNDataset, BalancedEMNISTDataset, ARDISDataset)):
        return torch.tensor(dataset.remapped_labels, dtype=torch.long)

    if hasattr(dataset, 'targets'):
        return torch.tensor([int(t) for t in dataset.targets], dtype=torch.long)

    raise ValueError(f"Cannot extract targets from dataset type: {type(dataset)}")


# =============================================================================
# Per-resolution-tier digit source selection (v3)
# =============================================================================

def digit_sources_for_tier(img_size: int) -> dict:
    """
    Returns the load_supplementary() keyword flags for which SUPPLEMENTARY
    digit sources are included at a given v3 resolution tier, per the
    per-source resolution ladder split:
        MNIST, EMNIST Digits, SVHN, ARDIS IV:  28 -> 32 -> 64 -> 128
        USPS:                                  16 -> 32 -> 64 -> 128

    Both ladders converge at 32x32, 64x64, and 128x128 — every source is
    included at those three tiers via load_supplementary() on top of
    load_base_mnist()'s own MNIST train/val/test spine (see each training
    script's load_mnist()/load_digit_data()). The two ladders' bottom
    tiers are each their OWN separate model — this is NOT a single
    "bottom tier" that picks one resolution:
      - 28x28: MNIST/EMNIST Digits/SVHN/ARDIS IV, USPS absent (16x16 is
        neither native nor meaningfully closer to native for USPS at
        28x28 than at its own 16x16, so it isn't upscaled in here).
      - 16x16: USPS's own native resolution. USPS is the ENTIRE spine at
        this tier — every training script uses load_base_usps() instead
        of load_base_mnist() when img_size == 16 (see load_base_usps()
        below), and this function returns every flag False for 16x16
        (there is nothing left for load_supplementary() to add on top —
        calling it anyway with all flags False is harmless, it just
        prints "no supplementary data available" and returns None).
    This means there are 5 resolution-tagged files per optimizer/router
    family (16, 28, 32, 64, 128), not 4 — see v3_CHANGELOG.md's resolution-
    ladder-split entry for the full reasoning, including the genuine
    ambiguity in the original task text this resolved.

    use_mnist here governs load_supplementary()'s OWN separate MNIST
    copy (on top of load_base_mnist()'s own train/val/test split) at the
    28/32/64/128 tiers — irrelevant at 16x16, where load_base_mnist()
    isn't called at all.
    """
    if img_size not in (16, 28, 32, 64, 128):
        raise ValueError(f"Unsupported v4 resolution tier: {img_size} "
                          f"(expected 16, 28, 32, 64, or 128)")
    if img_size == 16:
        # USPS is the entire spine at this tier via load_base_usps() —
        # nothing for load_supplementary() to add.
        return {
            "use_digits":    False,
            "use_mnist":     False,
            "use_usps":      False,
            "use_svhn":      False,
            "use_ardis":     False,
            "use_balanced":  False,
            "use_kaggle":    False,
            "use_chars_hnd": False,
            "use_chars_img": False,
            "use_pghwld":    False,
        }
    return {
        "use_digits":    True,             # EMNIST Digits — every 28+ tier
        "use_mnist":     True,             # MNIST          — every 28+ tier
        "use_usps":      img_size != 28,   # USPS — every 28+ tier EXCEPT 28x28 itself
        "use_svhn":      True,             # SVHN           — every 28+ tier
        "use_ardis":     True,             # ARDIS IV       — every 28+ tier
        "use_balanced":  False,
        "use_kaggle":    False,
        "use_chars_hnd": False,
        "use_chars_img": False,
        "use_pghwld":    False,
    }


# =============================================================================
# Main loader
# =============================================================================

def load_supplementary(
    transform,
    use_balanced:  bool = False,  # letters (47 classes) — digit ensemble leaves this False;
                                   # the router loads BalancedEMNISTDataset directly instead
                                   # (see module docstring)
    use_digits:    bool = True,
    use_mnist:     bool = True,
    use_usps:      bool = True,
    use_svhn:      bool = True,
    use_kaggle:    bool = False,  # A-Z uppercase — optional supplementary source, unused by default
    use_chars_hnd: bool = False,  # all 62 classes incl. letters — optional, unused by default
    use_chars_img: bool = False,  # all 62 classes incl. letters — optional, unused by default
    use_ardis:     bool = True,
    use_pghwld:    bool = False,  # A-Z uppercase — optional supplementary source, unused by default
    train:         bool = True,
) -> Optional[ConcatDataset]:
    """
    Load all available supplementary datasets.
    Gracefully skips any dataset that is missing or fails to load.

    Loading order:
      Digit datasets first (EMNIST Digits, MNIST, USPS, SVHN)
      then mixed/letter datasets (Balanced, Kaggle, Chars74K)

    Callers should pass **digit_sources_for_tier(img_size) for the
    use_digits/use_mnist/use_usps/use_svhn/use_ardis flags rather than
    hardcoding them, so every training script stays consistent with the
    v3 per-source resolution ladder split by construction.
    """
    datasets = []
    total    = 0

    # ── Digit datasets ────────────────────────────────────────────────────────
    if use_ardis:
        try:
            ds = ARDISDataset(transform=transform)
            datasets.append(ds)
            total += len(ds)
            print(f"  [Supplementary] ARDIS IV:        {len(ds):,} samples (Swedish historical digits)")
        except FileNotFoundError:
            print("  [Supplementary] ARDIS IV skipped — automatic download/setup "
                  "failed, see instructions printed above")
        except Exception as e:
            print(f"  [Supplementary] ARDIS IV skipped: {e}")

    if use_digits:
        try:
            ds = EMNISTDigitsDataset(train=train, transform=transform)
            datasets.append(ds)
            total += len(ds)
            print(f"  [Supplementary] EMNIST Digits:   {len(ds):,} samples")
        except Exception as e:
            print(f"  [Supplementary] EMNIST Digits skipped: {e}")

    if use_mnist:
        try:
            ds = MNISTDataset(train=train, transform=transform)
            datasets.append(ds)
            total += len(ds)
            print(f"  [Supplementary] MNIST:           {len(ds):,} samples")
        except Exception as e:
            print(f"  [Supplementary] MNIST skipped: {e}")

    if use_usps:
        try:
            ds = USPSDataset(train=train, transform=transform)
            datasets.append(ds)
            total += len(ds)
            print(f"  [Supplementary] USPS:            {len(ds):,} samples")
        except Exception as e:
            print(f"  [Supplementary] USPS skipped: {e}")

    if use_svhn and train:
        try:
            ds = SVHNDataset(train=train, transform=transform)
            datasets.append(ds)
            total += len(ds)
            print(f"  [Supplementary] SVHN:            {len(ds):,} samples")
        except Exception as e:
            print(f"  [Supplementary] SVHN skipped: {e}")

    # ── Mixed/letter datasets ─────────────────────────────────────────────────
    if use_balanced:
        try:
            ds = BalancedEMNISTDataset(train=train, transform=transform)
            datasets.append(ds)
            total += len(ds)
            print(f"  [Supplementary] EMNIST Balanced: {len(ds):,} samples")
        except Exception as e:
            print(f"  [Supplementary] EMNIST Balanced skipped: {e}")

    if use_kaggle and train:
        try:
            ds = KaggleAZDataset(transform=transform)
            datasets.append(ds)
            total += len(ds)
            print(f"  [Supplementary] Kaggle A-Z:      {len(ds):,} samples (uppercase only)")
        except FileNotFoundError:
            print("  [Supplementary] Kaggle A-Z skipped — dataset files not found")
        except Exception as e:
            print(f"  [Supplementary] Kaggle A-Z skipped: {e}")

    if use_chars_hnd and train:
        try:
            ds = Chars74KDataset(root=CHARS74K_HND, transform=transform)
            datasets.append(ds)
            total += len(ds)
            print(f"  [Supplementary] Chars74K Hnd:    {len(ds):,} samples (all 62 classes)")
        except FileNotFoundError:
            print(f"  [Supplementary] Chars74K Hnd skipped — not found at {CHARS74K_HND}")
        except Exception as e:
            print(f"  [Supplementary] Chars74K Hnd skipped: {e}")

    if use_chars_img and train:
        try:
            ds = Chars74KDataset(root=CHARS74K_IMG, transform=transform)
            datasets.append(ds)
            total += len(ds)
            print(f"  [Supplementary] Chars74K Img:    {len(ds):,} samples (all 62 classes)")
        except FileNotFoundError:
            print(f"  [Supplementary] Chars74K Img skipped — not found at {CHARS74K_IMG}")
        except Exception as e:
            print(f"  [Supplementary] Chars74K Img skipped: {e}")

    if use_pghwld and train:
        try:
            ds = PGHWLDDataset(transform=transform)
            datasets.append(ds)
            total += len(ds)
            print(f"  [Supplementary] PG-HWLD:         {len(ds):,} samples (non-NIST uppercase A-Z)")
        except FileNotFoundError:
            print(f"  [Supplementary] PG-HWLD skipped — manual download required")
        except Exception as e:
            print(f"  [Supplementary] PG-HWLD skipped: {e}")

    if not datasets:
        print("  [Supplementary] No supplementary data available — using base MNIST only")
        return None

    print(f"  [Supplementary] Total supplementary samples: {total:,}")
    return ConcatDataset(datasets)


def get_combined_weights(byclass_dataset, supplementary_dataset) -> torch.Tensor:
    """Compute weights for combined MNIST + supplementary digit dataset."""
    byclass_targets = _extract_targets(byclass_dataset)
    supp_targets    = _extract_targets(supplementary_dataset) if supplementary_dataset \
                      else torch.tensor([], dtype=torch.long)
    all_targets     = torch.cat([byclass_targets, supp_targets])
    class_counts    = torch.bincount(all_targets, minlength=NUM_CLASSES).float()
    class_counts    = torch.clamp(class_counts, min=1)
    class_weights   = 1.0 / class_counts
    class_weights[:10] *= DIGIT_BOOST
    sample_weights  = class_weights[all_targets]
    print(f"  [Weights] Combined: {len(all_targets):,} samples")
    print(f"  [Weights] Class weight range:  {class_weights.min():.6f} — {class_weights.max():.6f}")
    print(f"  [Weights] Digit weight range:  {class_weights[:10].min():.6f} — {class_weights[:10].max():.6f}")
    return sample_weights


# =============================================================================
# Base MNIST loader
# =============================================================================
# Every digit training script and the router fetch the base MNIST
# train/val/test split through this one shared function instead of each
# calling torchvision's MNIST class directly — one place all dataset
# loading goes through, consistent with how the digit supplementary
# sources (EMNIST/USPS/SVHN/ARDIS) already work via load_supplementary()
# above.
#
# Takes already-built transform objects rather than building them itself,
# since each optimizer script has its own augmentation strategy (see each
# script's own get_transforms()) — this function only handles the actual
# dataset fetch/split mechanics, not per-script augmentation choices.
#
# Known, accepted overlap (unchanged from v2, not introduced or fixed by
# this restructure): load_supplementary(use_mnist=True) loads a SECOND,
# full 60k-sample MNIST training copy as a "supplementary" source, on top
# of the 51k-sample train subset load_base_mnist() below already returns
# after its own 85/15 split — meaning some images that landed in
# load_base_mnist()'s held-out validation subset can still reach the
# model via that second copy during training. This predates v3 and Part
# 2 of this restructure is a structural refactor only (no behavior
# changes), so it's preserved exactly as-is; flagged here rather than
# silently carried forward unmentioned.

def load_base_mnist(data_dir: Path, train_transform, val_transform, test_transform,
                     validation_split: float = 0.15, split_seed: int = 42,
                     return_train_targets: bool = False):
    """
    Loads the base MNIST dataset (70,000 samples: 60k train, 10k test) and
    splits the training set into train/val.

    Args:
        data_dir: root directory MNIST is downloaded to/read from.
        train_transform: transform applied to the training subset (typically
            includes augmentation).
        val_transform: transform applied to the validation subset (typically
            no augmentation, matching test-time preprocessing).
        test_transform: transform applied to the held-out test set.
        validation_split: fraction of the 60k training samples held out for
            validation (default 0.15, matching every digit training script
            and the router).
        split_seed: seed for the train/val split (default 42, matching the
            existing GLOBAL_SEED convention used across the project).
        return_train_targets: if True, also returns a LongTensor of the
            training subset's labels (train_full.targets restricted to the
            train split's indices) — needed by callers that compute their
            own class weights from the raw labels (e.g. the SOAP scripts'
            fallback path when no supplementary data is available), since
            train_ds is a Subset wrapper and doesn't expose .targets
            directly. Default False keeps the original 3-tuple return for
            callers that don't need this.

    Returns:
        (train_ds, val_ds, test_ds) by default, or
        (train_ds, val_ds, test_ds, train_targets) if return_train_targets=True.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    print("[Dataset] Loading MNIST (via supplementary_data.load_base_mnist)...")

    train_full = MNIST(root=str(data_dir), train=True,
                        download=True, transform=train_transform)
    test_ds    = MNIST(root=str(data_dir), train=False,
                        download=True, transform=test_transform)

    total       = len(train_full)
    val_count   = int(total * validation_split)
    train_count = total - val_count

    generator = torch.Generator().manual_seed(split_seed)
    train_indices, val_indices = random_split(
        range(total), [train_count, val_count], generator=generator
    )

    train_ds = Subset(train_full, train_indices.indices)
    val_base = MNIST(root=str(data_dir), train=True,
                      download=False, transform=val_transform)
    val_ds   = Subset(val_base, val_indices.indices)

    print(f"[Dataset] Train: {train_count:,}  |  Val: {val_count:,}  |  "
          f"Test: {len(test_ds):,}")

    if return_train_targets:
        train_targets = torch.tensor(
            [int(train_full.targets[i]) for i in train_indices.indices], dtype=torch.long
        )
        return train_ds, val_ds, test_ds, train_targets

    return train_ds, val_ds, test_ds


# =============================================================================
# Base USPS loader (v3 — 16x16 tier)
# =============================================================================
# USPS is the ENTIRE training spine at the 16x16 resolution tier (see
# digit_sources_for_tier() above) — not a supplementary addition on top
# of load_base_mnist(), the way it is at 32/64/128. Every v3 digit
# training script and the router branch on `if img_size == 16:` to call
# this instead of load_base_mnist(); load_supplementary() is still
# called afterward for consistency with the other tiers, but
# digit_sources_for_tier(16) returns every flag False, so it's a no-op
# that gracefully returns None (matches load_supplementary()'s existing
# "no supplementary data available" fallback — no special-casing needed
# there).
#
# Mirrors load_base_mnist()'s structure exactly (same 85/15 default
# split, same GLOBAL_SEED-derived split_seed convention, same
# return_train_targets escape hatch) — the only real difference is the
# underlying dataset (USPSDataset instead of raw torchvision MNIST) and
# that USPSDataset stores labels as `.remapped_labels`, not `.targets`.

def load_base_usps(data_dir: Path, train_transform, val_transform, test_transform,
                    validation_split: float = 0.15, split_seed: int = 42,
                    return_train_targets: bool = False):
    """
    Loads the base USPS dataset (7,291 train + 2,007 test samples,
    digits 0-9, native 16x16 resolution) and splits the training set
    into train/val — the USPS-tier equivalent of load_base_mnist(),
    used as the digit ensemble's/router's entire training spine at the
    16x16 resolution tier instead of load_base_mnist() +
    load_supplementary(): at 16x16, USPS is the ONLY source in the
    ladder (see v3_CHANGELOG.md's Resolution ladder split section), so there's
    no separate "base spine" + "supplementary sources" split the way
    there is for the 28/32/64/128 tiers — USPS's own train/test split
    serves both roles directly.

    Args mirror load_base_mnist() exactly; same defaults, same
    return_train_targets behavior.

    Returns:
        (train_ds, val_ds, test_ds) by default, or
        (train_ds, val_ds, test_ds, train_targets) if return_train_targets=True.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    print("[Dataset] Loading USPS (via supplementary_data.load_base_usps)...")

    train_full = USPSDataset(train=True, transform=train_transform)
    test_ds    = USPSDataset(train=False, transform=test_transform)

    total       = len(train_full)
    val_count   = int(total * validation_split)
    train_count = total - val_count

    generator = torch.Generator().manual_seed(split_seed)
    train_indices, val_indices = random_split(
        range(total), [train_count, val_count], generator=generator
    )

    train_ds = Subset(train_full, train_indices.indices)
    val_base = USPSDataset(train=True, transform=val_transform)
    val_ds   = Subset(val_base, val_indices.indices)

    print(f"[Dataset] Train: {train_count:,}  |  Val: {val_count:,}  |  "
          f"Test: {len(test_ds):,}")

    if return_train_targets:
        train_targets = torch.tensor(
            [int(train_full.remapped_labels[i]) for i in train_indices.indices], dtype=torch.long
        )
        return train_ds, val_ds, test_ds, train_targets

    return train_ds, val_ds, test_ds


# =============================================================================
# Base EMNIST ByClass letter loader (v3 — uppercase/lowercase models)
# =============================================================================
# Unlike the digit ensemble's 5-source, per-resolution-tier ladder
# (digit_sources_for_tier()), the letter-identity models
# (v4_mnist_letter_{uc,lc}_{optimizer}_{res}.py) have exactly ONE data
# source — EMNIST ByClass — so there is no per-tier branching here: every
# resolution (28/32/64/128; there is no 16x16 letter tier, that was
# USPS-digit-only) calls this same function identically. Mirrors
# load_base_mnist()/load_base_usps()'s own train/val split mechanics
# exactly (same random_split + seeded Generator idiom, same
# return_train_targets escape hatch).

def load_base_emnist_letters(case: str, train_transform, val_transform, test_transform,
                              validation_split: float = 0.15, split_seed: int = 42,
                              return_train_targets: bool = False):
    """
    Loads EMNIST ByClass, filtered to one letter case, and splits its
    train portion into train/val. Labels are dense 0-25 (A=0..Z=25 for
    case="upper", a=0..z=25 for case="lower") — see _LetterOnlyDataset.

    Args mirror load_base_mnist()/load_base_usps(); same defaults, same
    return_train_targets behavior.

    Returns:
        (train_ds, val_ds, test_ds) by default, or
        (train_ds, val_ds, test_ds, train_targets) if return_train_targets=True.
    """
    if case not in ("upper", "lower"):
        raise ValueError(f"case must be 'upper' or 'lower', got {case!r}")

    print(f"[Dataset] Loading EMNIST ByClass ({case}) "
          f"(via supplementary_data.load_base_emnist_letters)...")

    train_full  = _LetterOnlyDataset(EMNISTByClassDataset(train=True,  transform=train_transform), case)
    train_noaug = _LetterOnlyDataset(EMNISTByClassDataset(train=True,  transform=val_transform),   case)
    test_ds     = _LetterOnlyDataset(EMNISTByClassDataset(train=False, transform=test_transform),  case)

    total       = len(train_full)
    val_count   = int(total * validation_split)
    train_count = total - val_count

    generator = torch.Generator().manual_seed(split_seed)
    train_indices, val_indices = random_split(
        range(total), [train_count, val_count], generator=generator
    )

    train_ds = Subset(train_full,  train_indices.indices)
    val_ds   = Subset(train_noaug, val_indices.indices)

    print(f"[Dataset] Train: {train_count:,}  |  Val: {val_count:,}  |  "
          f"Test: {len(test_ds):,}")

    if return_train_targets:
        train_targets = torch.tensor(
            [int(train_full.dense_labels[i]) for i in train_indices.indices], dtype=torch.long
        )
        return train_ds, val_ds, test_ds, train_targets

    return train_ds, val_ds, test_ds
