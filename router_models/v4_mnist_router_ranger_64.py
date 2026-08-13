"""
v4_mnist_router_ranger_64.py
====================
Router/classifier model — Stage 1 of the OCR pipeline. Adapted from v2's
mnist_digit_gate.py, which implemented a binary digit/not-digit gate;
this is a genuine scope expansion (2-class -> 4-class), not a rename —
see v3_CHANGELOG.md for the full before/after. Classifies each detected
character box into digit / UC (uppercase) / LC (lowercase) / [UNK]
(unclassifiable). Digit-classified boxes are handed to the digit
ensemble for actual digit reading; UC/LC are terminal labels for now (no
case-specific letter-identity model exists yet — see the module
docstring's "Scope note" in the original task description); [UNK] means
the router itself could not classify the box at all.

Architecture — OCRRouterNet (adapted from v2's OCRGateNet — same
lightweight conv stem, output width changed from 2 to 3 classes):
  Filter progression: 16→32→64→64, plain Conv+BatchNorm+ReLU blocks — no
  SE attention, no residual/stochastic-depth blocks. Deliberately far
  smaller than the digit ensemble's specialist models (this is a router,
  not a character recognizer).
  ~62.7K parameters, ~0.24 MB (float32).

Output: 3 logits — index 0 = digit, index 1 = UC (uppercase), index 2 =
LC (lowercase). No fourth "reject"/[UNK] class exists in the trained
model — [UNK] is produced purely by a confidence threshold at inference
time (ROUTER_UNSURE_FLOOR below), not a dedicated negative class. This
mirrors the v2 gate's own GATE_UNSURE_LOW/HIGH confidence-band approach
rather than introducing a second, different uncertainty mechanism —
chosen because it requires no negative/reject training data, which
doesn't exist anywhere in this project. See v3_CHANGELOG.md for the full
reasoning and what alternatives were considered.

Training data:
  digit (label=0): the SAME merged digit ensemble sources used by
    v4_mnist_digit_*.py at this resolution tier — USPS alone (via
    load_base_usps()) at 16x16, USPS's own native resolution and the
    only source in that ladder; MNIST + EMNIST Digits + SVHN + ARDIS IV
    (every source except USPS) at 28x28; every source including USPS at
    32/64/128 — via supplementary_data.py's load_base_mnist()/
    load_base_usps()/load_supplementary()/digit_sources_for_tier().
    Individual digit identity (0-9) is discarded; every digit source
    sample becomes label=0.
  UC (label=1) / LC (label=2): EMNIST Balanced (re-enabled for v3 — see
    supplementary_data.py's BalancedEMNISTDataset, whose byclass mapping
    was fixed this restructure to include lowercase — previously
    silently dropped, see v3_CHANGELOG.md). Balanced's own digit-labeled
    samples (byclass 0-9) are NOT used here — the router's digit class
    is sourced entirely from the merged digit ensemble data above, for
    consistency with the digit ensemble itself and to avoid diluting it
    with a third, lower-diversity digit source. byclass 10-35 (A-Z) ->
    UC; byclass 36-61 (the EMNIST-Balanced-distinct lowercase letters,
    a/b/d/e/f/g/h/n/q/r/t) -> LC. Unchanged across all 5 resolution
    tiers — EMNIST Balanced isn't part of the per-source resolution
    ladder split, only the digit class is.
  Class balance: WeightedRandomSampler, inverse-frequency across all
  three classes — same idiom as the rest of this project
  (get_class_weights()/WeightedRandomSampler), generalized here from the
  v2 gate's 2-class version to 3 classes (see compute_router_sample_weights()
  below — the letters block mixes UC and LC samples in shuffled order, so
  each sample needs its own weight, not one shared per-block weight like
  the gate's homogeneous 2-block case allowed).

Optimizer: Ranger (RAdam + Lookahead) — see section 2b below, in this
same file (implemented directly here, not in a separate shared module —
see v3_CHANGELOG.md's naming-and-modularization-scope correction).
Switched from the v2 gate's AdaBelief (see v3_CHANGELOG.md for the full
before/after of this optimizer swap, why the batch-size LR scaling and
warmup-length derivation were independently re-derived for Ranger rather
than assumed unchanged, and what alternatives were considered).
  Reference LR 1e-3 at REFERENCE_BATCH_SIZE=256, scaled LINEARLY (not
  sqrt) with batch size and capped at 4x — see scaled_learning_rate()
  below for why this differs from the gate's sqrt scaling.
  Lookahead k=6, alpha=0.5 (published Ranger defaults).
  A short, FIXED 100-step external warmup (not derived from
  steps-per-epoch like the gate's dynamic up-to-300-step warmup) — RAdam
  already performs its own analytic variance rectification in place of a
  hand-tuned warmup (that is RAdam's whole point), so the external warmup
  here only needs to give Lookahead's slow-weight mechanism a stable
  first few steps, not carry the full early-training variance-control
  burden the gate's warmup did for AdaBelief.
  PATIENCE=15

Resolution: matches the digit ensemble's own per-source ladder split —
16/28/32/64/128 (USPS alone at 16x16; MNIST/EMNIST/SVHN/ARDIS at 28x28;
every source including USPS at 32/64/128) — see v3_CHANGELOG.md's
Resolution ladder split section and supplementary_data.digit_sources_for_tier().

Box detection compatibility: get_boxes() in ocr_pipeline_mnist.py was
checked against realistic letter bounding-box shapes (ascender/descender
letters especially) before assuming no changes were needed for the
router to see letters at all — see ocr_pipeline_mnist.py's own docstring
and v3_CHANGELOG.md for what was verified.

Paths: DATA_DIR reuses supplementary_data.py's shared dataset root
(override with the ROUTER_DATA_DIR environment variable) rather than a
new hardcoded literal in this file — same pattern the v2 gate used
(GATE_DATA_DIR).

Output: ./v4_mnist_router_ranger_64/  (created next to this script)
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================
import os
import math
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # common/ and
# supplementary_data.py live at the project root, one level up from this
# script's own digit_models/uppercase_models/lowercase_models/router_models
# subfolder -- added when the project was reorganized into per-model-type
# folders, see v3_CHANGELOG.md.

from common.seeding import (
    apply_cublas_workspace_config, get_global_seed, set_all_seeds, reserve_cpu_threads,
    usable_cpu_count,
)

apply_cublas_workspace_config()

import torch

reserve_cpu_threads()
GLOBAL_SEED = get_global_seed()
set_all_seeds(GLOBAL_SEED)

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader, Dataset, Subset, random_split, WeightedRandomSampler, ConcatDataset,
)
from torchvision import transforms

from common.telemetry import (
    HardwareMonitor, setup_device, HAS_PSUTIL,
    check_vram_safety, check_cpu_ram_safety,
)
from common.distributed import (
    is_distributed, get_local_rank, is_main_process, setup_distributed,
    cleanup_distributed, wrap_model_ddp, unwrap_model, all_reduce_sum, get_world_size,
    DistributedWeightedRandomSampler,
)
from common.batch_sizing import determine_batch_size, cap_batch_size_for_min_steps, reduce_batch_size
from common.checkpointing import (
    EarlyStopping, save_resume_state, clear_resume_state, restore_optimizer_scheduler_state,
)
from common.cli_logging import _Tee, plot_history, save_log
from common.onnx_export import export_onnx
from common.scheduler import WarmupCosineScheduler
from common.amp import amp_train_step
from common.optimizers import build_ranger_optimizer


if HAS_PSUTIL:
    import psutil

try:
    from supplementary_data import (
        load_supplementary, load_base_mnist, load_base_usps, digit_sources_for_tier,
        BalancedEMNISTDataset, DATA_DIR as _SUPP_DATA_DIR,
    )
    HAS_SUPPLEMENTARY = True
except ImportError:
    HAS_SUPPLEMENTARY = False
    _SUPP_DATA_DIR = None
    print("[Warning] supplementary_data.py not found — this model requires "
          "it for both the digit and letter classes; training will fail without it.")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

NUM_CLASSES = 3  # 0 = digit, 1 = UC (uppercase), 2 = LC (lowercase)
ROUTER_DIGIT, ROUTER_UC, ROUTER_LC = 0, 1, 2
LABEL_MAP = ["digit", "UC", "LC"]

# [UNK] is NOT a trained class — see module docstring. This is the
# confidence floor applied to the router's own top-class softmax
# probability at inference time; below it, the pipeline outputs [UNK]
# instead of the top class. Chosen to sit meaningfully above the 3-class
# uniform-chance baseline (~0.33) without over-flagging ordinary
# confident predictions as unknown. Like the gate's original
# GATE_UNSURE_LOW/HIGH band, this is a starting point, not a value tuned
# against real-world data — see v3_CHANGELOG.md; real-world tuning is
# explicitly out of scope for this restructure.
ROUTER_UNSURE_FLOOR = 0.6

LEARNING_RATE        = 1e-3
REFERENCE_BATCH_SIZE = 256
WEIGHT_DECAY     = 1e-4
VALIDATION_SPLIT = 0.15
PATIENCE         = 15
MIN_STEPS_PER_EPOCH = 15  # floor on real gradient-update steps per epoch — see
                          # cap_batch_size_for_min_steps() in common/batch_sizing.py
NUM_WORKERS      = usable_cpu_count()  # auto-scales to this machine's real
                        # core count, leaving 25% for the OS (2026-07-28,
                        # per direct user follow-up) — was a hardcoded 10,
                        # which left cores idle on a many-core CPU and could
                        # be too many workers on a small one. DataLoader
                        # worker processes compete for CPU regardless of
                        # whether training itself runs on GPU or CPU, so
                        # this uses the same 25%-reserved-for-the-OS policy
                        # already applied to CPU-only torch.set_num_threads().
USE_AMP          = True

LOOKAHEAD_K     = 6
LOOKAHEAD_ALPHA = 0.5
WARMUP_STEPS    = 100  # fixed, not steps-per-epoch-derived — see module docstring
SCHEDULE_EPOCH_ESTIMATE = 100  # shapes the cosine LR curve only, not a hard epoch cap —
                                # PATIENCE-based early stopping is what actually ends training.

RAM_RESERVE_GB = 4.0

DATA_DIR = Path(os.environ.get(
    "ROUTER_DATA_DIR",
    str(_SUPP_DATA_DIR) if _SUPP_DATA_DIR is not None
    else str(Path(__file__).resolve().parent / "datasets"),
))
IMG_SIZE    = 64
# OUTPUT_ROOT is script-relative (Path(__file__).resolve().parent / ...) —
# safe as-is on any machine, unlike DATA_DIR above. It creates its own
# output folder next to wherever this script actually runs from, so it
# does not need to be personalized for a different system.
OUTPUT_ROOT = Path(__file__).resolve().parent / f"v4_mnist_router_ranger_{IMG_SIZE}"


def scaled_learning_rate(batch_size: int) -> float:
    """
    Linear (not sqrt) LR scaling relative to REFERENCE_BATCH_SIZE,
    capped at 4x the reference LR. Differs deliberately from the v2
    gate's sqrt scaling (scaled_learning_rate() there, for AdaBelief) —
    see module docstring for the full reasoning: RAdam's own analytic
    variance rectification already does the job AdaBelief's sqrt scaling
    was partly compensating for (damping an adaptive optimizer's
    early-training instability), so batch-size-to-LR scaling here follows
    the standard linear-scaling-rule literature (Goyal et al., 2017)
    instead, capped to avoid instability at this project's largest
    auto-detected batch sizes (into the thousands at low resolution).
    """
    raw_scale = batch_size / REFERENCE_BATCH_SIZE
    return LEARNING_RATE * min(raw_scale, 4.0)


def build_router_optimizer(params, lr: float):
    return build_ranger_optimizer(
        params, lr=lr, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=WEIGHT_DECAY, k=LOOKAHEAD_K, alpha=LOOKAHEAD_ALPHA,
    )


def build_config(img_size: int, batch_size: int) -> dict:
    prefix = f"v4_mnist_router_ranger_{img_size}"
    return {
        "img_size":    img_size,
        "batch_size":  batch_size,
        "checkpoint_path":  str(OUTPUT_ROOT / f"{prefix}_best.pt"),
        "final_model_path": str(OUTPUT_ROOT / f"{prefix}_final.pt"),
        "onnx_path":        str(OUTPUT_ROOT / f"{prefix}.onnx"),
        "log_path":         str(OUTPUT_ROOT / f"{prefix}_log.csv"),
        "plot_path":        str(OUTPUT_ROOT / f"{prefix}_curves.png"),
        "resume_path":      str(OUTPUT_ROOT / f"{prefix}_resume.pt"),
    }


# =============================================================================
# 2. DATA PIPELINE
# =============================================================================

def get_transforms(img_size: int, augment: bool = False) -> transforms.Compose:
    """Same standardized augmentation recipe used across the digit
    ensemble — kept identical since the router trains on the same image
    domain (cropped character boxes)."""
    aug_transforms = [
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=5),
        transforms.ColorJitter(contrast=0.3, brightness=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
    ] if augment else []
    base_transforms = [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
    return transforms.Compose(aug_transforms + base_transforms)


class _FixedLabelDataset(Dataset):
    """Wraps any (img, label)-yielding dataset, discarding its original
    label and returning one fixed label instead. Used to turn the merged
    digit sources into the router's single 'digit' class (label=0) —
    the router does not care which digit, only that it is one."""
    def __init__(self, base, label: int, indices=None):
        self.base    = base
        self.label   = label
        self.indices = list(indices) if indices is not None else None

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.base)

    def __getitem__(self, idx):
        real_idx = self.indices[idx] if self.indices is not None else idx
        img, _ = self.base[real_idx]
        return img, self.label


class _CaseLabelDataset(Dataset):
    """
    Wraps BalancedEMNISTDataset, keeping only byclass labels 10-61
    (uppercase + lowercase) — byclass 0-9 (digits-in-Balanced) are
    excluded, since the router's digit class is sourced from the merged
    digit ensemble data instead (see module docstring) — and remapping
    byclass -> router case label (ROUTER_UC for byclass 10-35,
    ROUTER_LC for byclass 36-61).
    """
    def __init__(self, base: BalancedEMNISTDataset):
        self.base = base
        labels = base.remapped_labels
        self.indices = [i for i, lbl in enumerate(labels) if lbl >= 10]
        self.case_labels = [ROUTER_UC if labels[i] < 36 else ROUTER_LC for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, _ = self.base[self.indices[idx]]
        return img, self.case_labels[idx]


def load_letters_split(img_size: int, validation_split: float, split_seed: int):
    """
    Loads EMNIST Balanced's uppercase+lowercase portion (via
    _CaseLabelDataset) and splits its train portion into train/val —
    mirrors load_base_mnist()'s own train/val split mechanics (same
    random_split + Generator pattern), matching the v2 gate's own
    load_letters_split() structure. Unaffected by the digit-side
    resolution-tier split (EMNIST Balanced isn't part of that ladder).
    """
    print("[Dataset] Loading EMNIST Balanced (UC/LC classes)...")
    train_aug   = _CaseLabelDataset(BalancedEMNISTDataset(train=True,  transform=get_transforms(img_size, augment=True)))
    train_noaug = _CaseLabelDataset(BalancedEMNISTDataset(train=True,  transform=get_transforms(img_size, augment=False)))
    test_ds     = _CaseLabelDataset(BalancedEMNISTDataset(train=False, transform=get_transforms(img_size, augment=False)))

    total       = len(train_aug)
    val_count   = int(total * validation_split)
    train_count = total - val_count
    generator   = torch.Generator().manual_seed(split_seed)
    train_indices, val_indices = random_split(range(total), [train_count, val_count], generator=generator)

    train_ds = Subset(train_aug,   train_indices.indices)
    val_ds   = Subset(train_noaug, val_indices.indices)

    print(f"[Dataset] EMNIST Balanced UC/LC — Train: {train_count:,}  |  "
          f"Val: {val_count:,}  |  Test: {len(test_ds):,}")
    return train_ds, val_ds, test_ds


def load_router_data(data_dir: Path, img_size: int):
    """
    Builds the combined digit (label=0) / UC (label=1) / LC (label=2)
    train/val/test sets. Digit class reuses supplementary_data.py's
    existing loaders exactly as the digit ensemble does, restricted to
    this resolution tier's source list: USPS alone (load_base_usps()) at
    16x16, base MNIST (load_base_mnist()) plus digit_sources_for_tier()'s
    supplementary sources everywhere else. UC/LC classes come from EMNIST
    Balanced (see load_letters_split() above) regardless of resolution
    tier.
    """
    base_loader = load_base_usps if img_size == 16 else load_base_mnist
    digit_train_ds, digit_val_ds, digit_test_ds = base_loader(
        data_dir=data_dir,
        train_transform=get_transforms(img_size, augment=True),
        val_transform=get_transforms(img_size, augment=False),
        test_transform=get_transforms(img_size, augment=False),
        validation_split=VALIDATION_SPLIT,
        split_seed=GLOBAL_SEED,
    )

    print("[Dataset] Loading digit supplementary data (digit class)...")
    supp_ds = load_supplementary(
        transform=get_transforms(img_size, augment=True),
        train=True,
        **digit_sources_for_tier(img_size),
    )
    if supp_ds is not None:
        digit_train_ds = ConcatDataset([digit_train_ds, supp_ds])
        print(f"[Dataset] Combined digit training set: {len(digit_train_ds):,} samples")

    letters_train_ds, letters_val_ds, letters_test_ds = load_letters_split(
        img_size, VALIDATION_SPLIT, GLOBAL_SEED,
    )

    digit_train = _FixedLabelDataset(digit_train_ds, label=ROUTER_DIGIT)
    digit_val   = _FixedLabelDataset(digit_val_ds,   label=ROUTER_DIGIT)
    digit_test  = _FixedLabelDataset(digit_test_ds,  label=ROUTER_DIGIT)

    train_ds = ConcatDataset([digit_train, letters_train_ds])
    val_ds   = ConcatDataset([digit_val,   letters_val_ds])
    test_ds  = ConcatDataset([digit_test,  letters_test_ds])

    print(f"[Dataset] Router train set: {len(digit_train):,} digit + "
          f"{len(letters_train_ds):,} UC/LC = {len(train_ds):,} total")
    print(f"[Dataset] Router val set:   {len(digit_val):,} digit + "
          f"{len(letters_val_ds):,} UC/LC = {len(val_ds):,} total")
    print(f"[Dataset] Router test set:  {len(digit_test):,} digit + "
          f"{len(letters_test_ds):,} UC/LC = {len(test_ds):,} total")

    return train_ds, val_ds, test_ds, digit_train, letters_train_ds


def compute_router_sample_weights(digit_train, letters_train_subset) -> torch.Tensor:
    """
    Inverse-frequency sample weights across all three classes, generalized
    from the v2 gate's 2-class version (which could use one shared weight
    per concatenated block, since its two blocks were each internally
    homogeneous). Here the letters block mixes UC and LC samples in
    shuffled order (random_split()'s output), so each letters sample needs
    its OWN weight based on its actual case label, not a block-uniform one.
    """
    n_digit = len(digit_train)
    base: _CaseLabelDataset = letters_train_subset.dataset
    idx = letters_train_subset.indices
    case_labels = [base.case_labels[i] for i in idx]
    n_uc = sum(1 for c in case_labels if c == ROUTER_UC)
    n_lc = sum(1 for c in case_labels if c == ROUTER_LC)

    print(f"[Dataset] Balancing sampler: digit={n_digit:,} (w={1.0/n_digit:.8f})  "
          f"UC={n_uc:,} (w={1.0/n_uc:.8f})  LC={n_lc:,} (w={1.0/n_lc:.8f})")

    digit_weights  = torch.full((n_digit,), 1.0 / n_digit, dtype=torch.double)
    letter_weights = torch.tensor(
        [1.0 / n_uc if c == ROUTER_UC else 1.0 / n_lc for c in case_labels], dtype=torch.double
    )
    return torch.cat([digit_weights, letter_weights])


def make_train_loader(train_ds, digit_train, letters_train_subset, batch_size: int) -> DataLoader:
    weights = compute_router_sample_weights(digit_train, letters_train_subset)
    if is_distributed():
        sampler = DistributedWeightedRandomSampler(weights, len(weights))
    else:
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0, drop_last=False,
    )


def make_eval_loader(dataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=torch.cuda.is_available())


# =============================================================================
# 3. MODEL ARCHITECTURE — OCRRouterNet
# =============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class OCRRouterNet(nn.Module):
    """
    Lightweight digit/UC/LC router. Adapted from v2's OCRGateNet — same
    conv stem, output width changed from 2 (digit/not-digit) to 3
    (digit/UC/LC).
    Input:  (batch, 1, H, W) — any resolution, GlobalAveragePooling makes
            the classifier head resolution-independent.
    Output: (batch, 3) logits — index 0 = digit, 1 = UC, 2 = LC.
    Filter progression: 16→32→64→64 — downsamples via stride=2 in each
    stage's conv (2026-08-07 fix: previously pooled via a separate
    MaxPool2d AFTER each full-resolution conv) — see v3_CHANGELOG.md.
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.stem   = ConvBlock(1, 16)
        self.stage1 = ConvBlock(16, 32, stride=2)
        self.stage2 = ConvBlock(32, 64, stride=2)
        self.stage3 = ConvBlock(64, 64, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)


# =============================================================================
# 4. TRAINING LOOP
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, scheduler,
                    epoch: int = None, img_size: int = None) -> tuple:
    model.train()
    total_loss = total_correct = total_samples = 0
    num_batches  = len(loader)
    log_interval = max(1, round(num_batches * 0.025))
    # Hardware telemetry (2026-08-07, per direct user follow-up): sampled
    # every _SAMPLE_INTERVAL batches (plus the first and last batch as
    # anchors) rather than a fixed 5 points, so sample density scales with
    # epoch length — batch counts in this project range from the enforced
    # 15-step floor (see MIN_STEPS_PER_EPOCH) up past 1000+ at larger
    # batch sizes, and a fixed sample count would under-resolve long
    # epochs. Every individual sample is written to the log as its own
    # row (see common/cli_logging.py's save_log()), not just reduced to
    # min/avg/max — epoch_summary() below still computes that reduction
    # too, for the epoch's own summary row.
    _hw_monitor = HardwareMonitor()
    _hw_samples = []
    _SAMPLE_INTERVAL = 25
    _sample_points = (set(range(0, num_batches, _SAMPLE_INTERVAL)) | {num_batches - 1}) if num_batches else set()
    # Data-wait vs. compute time: times the dataloader's next-batch yield
    # separately from the training step itself, so a GPU-starved-by-data
    # bottleneck (CPU/disk-bound) is distinguishable from a genuinely
    # GPU-bound run — utilization% alone can't tell those apart.
    _data_wait_s = 0.0
    _compute_s   = 0.0
    _t_prev = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        _t_data = time.time()
        _data_wait_s += _t_data - _t_prev
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        loss, logits, _stepped = amp_train_step(
            model, images, labels, criterion, optimizer, scaler, device,
            use_amp=USE_AMP, grad_clip_norm=1.0,
        )
        if _stepped:
            scheduler.step()

        _t_prev = time.time()
        _compute_s += _t_prev - _t_data

        if batch_idx in _sample_points:
            _hw_samples.append(_hw_monitor.sample())

        total_loss    += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += images.size(0)

        if is_main_process() and ((batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches):
            _running_loss = total_loss / total_samples
            _running_acc  = total_correct / total_samples
            _prefix = f"[{img_size}x{img_size}] " if img_size is not None else ""
            _epoch_str = f"Epoch {epoch:3d}  " if epoch is not None else ""
            print(f"  {_prefix}{_epoch_str}Batch {batch_idx + 1:5d}/{num_batches:5d}  "
                  f"loss: {_running_loss:.6f}  acc: {_running_acc:.6f}")

    if not _hw_samples:
        _hw_samples.append(_hw_monitor.sample())
    hw_summary = HardwareMonitor.epoch_summary(_hw_samples)
    hw_summary["data_wait_s"] = round(_data_wait_s, 2)
    hw_summary["compute_s"]   = round(_compute_s, 2)
    # DDP metric aggregation (2026-07-28, per direct user follow-up): each
    # rank only sees its own shard of the training data (via
    # DistributedWeightedRandomSampler in
    # make_train_loader() below), so total_loss/total_correct/
    # total_samples are LOCAL to this rank until summed across every rank
    # — without this, a distributed run's reported train_loss/train_acc
    # would silently reflect only one rank's slice of the epoch, not the
    # real global numbers a single-GPU run would show. No-op passthrough
    # when not running distributed — see common/distributed.py.
    total_loss    = all_reduce_sum(total_loss, device)
    total_correct = all_reduce_sum(total_correct, device)
    total_samples = all_reduce_sum(total_samples, device)
    # hw_summary is the epoch's min/avg/max-reduced summary (see
    # HardwareMonitor.epoch_summary()); _hw_samples is the raw list it was
    # reduced from — returned too (2026-08-07, per direct user follow-up)
    # so the caller can log each individual sample point as its own CSV
    # row, not just the reduction — see common/cli_logging.py's save_log().
    return total_loss / total_samples, total_correct / total_samples, hw_summary, _hw_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, per_class: bool = False) -> tuple:
    model.eval()
    total_loss = total_correct = total_samples = 0

    if per_class:
        class_correct = torch.zeros(NUM_CLASSES)
        class_total   = torch.zeros(NUM_CLASSES)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(1)

        total_loss    += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

        if per_class:
            for c in range(NUM_CLASSES):
                mask = labels == c
                class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                class_total[c]   += mask.sum().item()

    if per_class and class_total.sum() > 0:
        class_acc = class_correct / class_total.clamp(min=1)
        print("\n  [Per-Class] Accuracy by class:")
        for idx in range(NUM_CLASSES):
            print(f"    {LABEL_MAP[idx]:6s} (class {idx}): "
                  f"{class_acc[idx]*100:.2f}%  ({int(class_total[idx])} samples)")
        balanced_acc = class_acc.mean().item()
        print(f"    Balanced accuracy (mean of the three): {balanced_acc*100:.4f}%")

    return total_loss / total_samples, total_correct / total_samples


# =============================================================================
# 5. RUN TRAINING
# =============================================================================

def run_training(img_size: int, batch_override: int = None, gpu_id: int = None):
    if not HAS_SUPPLEMENTARY:
        print("[Error] supplementary_data.py is required for both the digit "
              "and letter classes and was not found — cannot train.")
        sys.exit(1)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = setup_device(use_amp=USE_AMP,
                          gpu_id=get_local_rank() if is_distributed() else gpu_id)
    setup_distributed(device)

    if batch_override is not None:
        batch_size = batch_override
        print(f"[Batch] Override: using batch size {batch_size} (skipping auto-detect)")
    else:
        batch_size = determine_batch_size(
            build_model=lambda: OCRRouterNet(NUM_CLASSES),
            build_optimizer=lambda params: build_router_optimizer(params, lr=scaled_learning_rate(256)),
            criterion=nn.CrossEntropyLoss(label_smoothing=0.05),
            img_size=img_size, num_classes=NUM_CLASSES, device=device,
            use_amp=USE_AMP, grad_clip_norm=1.0,
        )
    cfg = build_config(img_size, batch_size)

    print("=" * 60)
    print(f"  Router — Ranger (RAdam + Lookahead)  [{img_size}x{img_size}]")
    print(f"  PyTorch {torch.__version__}  |  AMP: {USE_AMP}")
    print(f"  Output: {OUTPUT_ROOT}")
    print(f"  Resolution: {img_size}x{img_size}  |  Batch: {cfg['batch_size']} {'(override)' if batch_override else '(auto-detected)'}")
    print(f"  Optimizer: Ranger  |  Scheduler: fixed-length warmup -> cosine (per-step)")
    print(f"  Output classes: {LABEL_MAP}  |  ROUTER_UNSURE_FLOOR={ROUTER_UNSURE_FLOOR}")
    print(f"  Digit sources this tier: {digit_sources_for_tier(img_size)}")
    print(f"  Normalization: [0,1]  (no mean/std shift)")
    print("=" * 60)

    train_ds, val_ds, test_ds, digit_train, letters_train_subset = load_router_data(DATA_DIR, img_size)

    # Minimum steps/epoch floor (2026-07-28, per direct user follow-up):
    # the VRAM probe above tests dummy tensors before the real dataset is
    # loaded, so it has no visibility into how many real training samples
    # exist — a small model (e.g. the router) or a small dataset (e.g. the
    # 16x16 USPS-only tier) can land on a batch size that gives only a
    # handful of real gradient-update steps per epoch. Capped here, now
    # that the real training set size is known, before the DataLoader is
    # built — never raises the batch size, only lowers it.
    _min_steps_batch = cap_batch_size_for_min_steps(
        cfg["batch_size"], len(train_ds), MIN_STEPS_PER_EPOCH,
        world_size=get_world_size(),
    )
    if _min_steps_batch < cfg["batch_size"]:
        print(f"[Batch] Capping batch size {cfg['batch_size']} -> {_min_steps_batch} "
              f"to guarantee >= {MIN_STEPS_PER_EPOCH} steps/epoch "
              f"({len(train_ds):,} train samples)")
        cfg["batch_size"] = _min_steps_batch
    train_loader = make_train_loader(train_ds, digit_train, letters_train_subset, cfg["batch_size"])
    val_loader   = make_eval_loader(val_ds,  cfg["batch_size"])
    test_loader  = make_eval_loader(test_ds, cfg["batch_size"])

    model = OCRRouterNet(NUM_CLASSES).to(device)
    model = wrap_model_ddp(model, device)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] OCRRouterNet — {img_size}x{img_size}")
    print(f"  Parameters : {total:,}")
    print(f"  Est. size  : {total * 4 / 1024**2:.2f} MB (float32)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    steps_per_epoch = len(train_loader)
    # DDP note (2026-07-28, per direct user follow-up): LR scaling uses the
    # GLOBAL effective batch size (this rank's batch * world_size), not
    # just this rank's own slice — the standard linear-scaling-rule
    # convention for distributed training (every rank's gradient is
    # already averaged across the global batch by DDP's backward hook, so
    # the LR should reflect the batch size that average was computed
    # over). get_world_size() is 1 for every non-distributed run, so
    # _global_batch_size == cfg["batch_size"] unchanged there.
    _global_batch_size = cfg["batch_size"] * get_world_size()
    scaled_lr       = scaled_learning_rate(_global_batch_size)
    total_steps     = SCHEDULE_EPOCH_ESTIMATE * steps_per_epoch

    optimizer = build_router_optimizer(model.parameters(), lr=scaled_lr)
    _ddp_note = (f" [{cfg['batch_size']}/rank x {get_world_size()} ranks]"
                if is_distributed() else "")
    print(f"[Optimizer] Ranger  lr={scaled_lr:.2e}  "
          f"(reference={LEARNING_RATE:.2e} @ batch={REFERENCE_BATCH_SIZE}, "
          f"scaled by min({_global_batch_size}/{REFERENCE_BATCH_SIZE}, 4.0)="
          f"{min(_global_batch_size / REFERENCE_BATCH_SIZE, 4.0):.3f}x{_ddp_note})  "
          f"wd={WEIGHT_DECAY}  betas=(0.9, 0.999)  lookahead_k={LOOKAHEAD_K}  alpha={LOOKAHEAD_ALPHA}")
    print(f"[Schedule] Fixed warmup: {WARMUP_STEPS} steps ({steps_per_epoch} steps/epoch)  ->  "
          f"cosine decay over ~{SCHEDULE_EPOCH_ESTIMATE} epochs "
          f"({total_steps:,} steps total — shape target only, PATIENCE governs the actual stop)")

    scheduler  = WarmupCosineScheduler(optimizer, WARMUP_STEPS, total_steps, eta_min=1e-4)
    scaler     = torch.amp.GradScaler('cuda', enabled=USE_AMP and device.type == "cuda")
    early_stop = EarlyStopping(patience=PATIENCE, path=cfg["checkpoint_path"])

    print(f"\n[Train] Starting {img_size}x{img_size} — no epoch cap | batch: {cfg['batch_size']} | patience: {PATIENCE}")
    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "lr"]}

    # ── Resume from previous session ─────────────────────────────────────
    start_epoch = 1
    resume_path = Path(cfg["resume_path"])
    if resume_path.exists() and Path(cfg["checkpoint_path"]).exists():
        try:
            _rs = torch.load(str(resume_path), map_location=device, weights_only=False)
            _ckpt = torch.load(cfg["checkpoint_path"], map_location=device, weights_only=False)
            unwrap_model(model).load_state_dict(_ckpt["state_dict"] if "state_dict" in _ckpt else _ckpt)

            # Batch size auto-detects fresh every run — if it differs from
            # the run that saved this resume file, the saved
            # optimizer/scheduler state encodes a peak LR + warmup +
            # step count computed for a DIFFERENT batch size (see
            # scaled_learning_rate()). Loading it anyway would silently
            # resume a wrong-for-this-run LR schedule — same resume-
            # correctness precedent as the v2 gate's own check. Model
            # weights, epoch, and patience state are batch-size-
            # independent and always safe to resume.
            #
            # Compared against _global_batch_size, not cfg["batch_size"]
            # (2026-07-28, per direct user follow-up — DDP support): the
            # LR was scaled from the GLOBAL effective batch size (this
            # rank's batch * world_size — see above), so a resume where
            # world_size changed but the per-rank batch size happened to
            # auto-detect the same would otherwise pass this check while
            # actually resuming state built for a different real LR.
            restore_optimizer_scheduler_state(_rs, _global_batch_size, optimizer, scheduler)
            scaler.load_state_dict(_rs["scaler_state"])
            early_stop.counter   = _rs["patience_counter"]
            early_stop.best_loss = _rs["best_val_loss"]
            history               = _rs["history"]
            start_epoch           = _rs["epoch"] + 1
            print(f"[Resume] Loaded from epoch {_rs['epoch']} "
                  f"(val_loss={_rs['best_val_loss']:.6f}, patience={_rs['patience_counter']}/{PATIENCE})")
            print(f"[Resume] Continuing from epoch {start_epoch}")
        except Exception as _e:
            print(f"[Resume] Could not load state: {_e} — starting fresh")
            start_epoch = 1
    else:
        print("[Resume] No prior checkpoint found — starting fresh")

    _run_swap_baseline_gb = round(psutil.swap_memory().used / 1024**3, 3) if HAS_PSUTIL else 0.0
    if HAS_PSUTIL and device.type == "cpu":
        print(f"[RAM] Swap/page-file baseline for this run: {_run_swap_baseline_gb:.3f} GB")

    for epoch in range(start_epoch, 10**6):
        # DDP epoch reshuffle (2026-07-28, per direct user follow-up):
        # DistributedWeightedRandomSampler's weighted draw is seeded by
        # seed + epoch (see common/distributed.py) — without telling it
        # which epoch this is, every epoch would draw the identical
        # weighted sample. No-op when not running distributed.
        if is_distributed():
            train_loader.sampler.set_epoch(epoch)
        t0 = time.time()
        try:
            train_loss, train_acc, hw, hw_raw = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, scheduler,
                epoch=epoch, img_size=img_size
            )
        except RuntimeError as _oom_e:
            # Real mid-training OOM catch (2026-08-09, per direct user
            # follow-up): before this, the only OOM handling anywhere in
            # this project was inside common/batch_sizing.py's one-time
            # startup probe — a genuine OOM during actual training (as
            # opposed to the probe) was a bare unhandled crash with no
            # checkpoint save. Same substring match the probe already
            # uses, reused here for consistency between probe-time and
            # real-training-time OOMs. epoch-1 (not epoch) because this
            # epoch never completed — history/early_stop/checkpoint state
            # all still reflect the last successful epoch.
            _oom_emsg = str(_oom_e).lower()
            if not ("out of memory" in _oom_emsg or "find was unable" in _oom_emsg or "engine" in _oom_emsg):
                raise
            print(f"  [OOM] Real CUDA OOM caught mid-epoch {epoch}: {_oom_e}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            save_resume_state(
                cfg["resume_path"],
                epoch=epoch - 1,
                batch_size=_global_batch_size,  # DDP: global, not per-rank — see the LR-scaling note above
                patience_counter=early_stop.counter,
                best_val_loss=float(early_stop.best_loss),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                scaler_state=scaler.state_dict(),
                history=history,
            )
            break
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        print(f"[{img_size}x{img_size}] Epoch {epoch:3d}  "
              f"loss: {train_loss:.6f}  acc: {train_acc:.6f}  |  "
              f"val_loss: {val_loss:.6f}  val_acc: {val_acc:.6f}  |  "
              f"lr: {current_lr:.8f}  [{elapsed:.0f}s]  |  "
              f"VRAM {hw['nvsmi_vram_used_gb_max']:.1f}/{hw['nvsmi_vram_total_gb']:.1f}GB  "
              f"CUDA {hw['cuda_util_pct_avg']:.0f}/{hw['cuda_util_pct_max']}%(avg/max)  "
              f"{hw['gpu_temp_c_avg']:.0f}/{hw['gpu_temp_c_max']}°C  {hw['gpu_power_w_avg']:.0f}W  |  "
              f"CPU {hw['cpu_pct_avg']:.0f}%  RAM {hw['ram_used_gb_avg']:.1f}/{hw['ram_total_gb']:.1f}GB  "
              f"SWAP {hw['swap_used_gb_avg']:.1f}/{hw['swap_total_gb']:.1f}GB  |  "
              f"wait {hw['data_wait_s']:.1f}s/compute {hw['compute_s']:.1f}s"
              f"{'  [THROTTLED]' if hw['gpu_throttled_any'] == 1 else ''}")

        for k, v in [("train_loss", train_loss), ("train_acc", train_acc),
                     ("val_loss", val_loss), ("val_acc", val_acc), ("lr", current_lr)]:
            history[k].append(v)
        # Every HardwareMonitor.epoch_summary() key auto-populates its own
        # history column (2026-07-28, per direct user follow-up) — the old
        # 6-field explicit list here would silently drop any new telemetry
        # field added to epoch_summary() without a matching edit here.
        # Raw per-sample-point readings, stored alongside the reduced
        # summary (2026-08-07, per direct user follow-up) so save_log()
        # can write each sample point as its own CSV row — see that
        # function's own docstring for the row layout.
        history.setdefault("_hw_raw_samples", []).append(hw_raw)
        for hw_key, hw_val in hw.items():
            history.setdefault(hw_key, []).append(hw_val)
        history.setdefault("epoch_time_s", []).append(round(elapsed, 1))

        # Write/update the CSV after every epoch (not just at the end) so it
        # exists from the first completed epoch and survives a crash or
        # manual stop. A resumed run's `history` already contains the
        # pre-resume epochs (restored from the checkpoint above), so this
        # naturally continues the same file rather than needing separate
        # append logic.
        save_log(history, cfg["log_path"])

        # Checkpoint save moved ahead of the safety-check breaks below
        # (2026-08-08, per direct user follow-up) — early_stop() is the
        # only thing that writes cfg["checkpoint_path"]; running it after
        # a safety break meant a safety-triggered stop on the very first
        # improving epoch left no checkpoint on disk at all, crashing the
        # unconditional torch.load() after this loop. Now the current
        # epoch's checkpoint is always saved (if it's a new best) before
        # any stop decision is made.
        early_stop(val_loss, model)

        _ram_stop    = check_cpu_ram_safety(device, _run_swap_baseline_gb, RAM_RESERVE_GB, epoch)
        _vram_status = check_vram_safety(
            hw['nvsmi_vram_used_gb_max'], hw['nvsmi_vram_total_gb'], epoch,
            vram_history=history['nvsmi_vram_used_gb_max'],
        )

        if _ram_stop or _vram_status == "stop":
            # Resume state now also saved on a hard safety stop (2026-08-08,
            # per direct user follow-up) — previously only a normal
            # continuing epoch saved it, leaving a safety-triggered stop
            # with no way to resume mid-schedule, only reload the best
            # checkpoint from scratch.
            save_resume_state(
                cfg["resume_path"],
                epoch=epoch,
                batch_size=_global_batch_size,  # DDP: global, not per-rank — see the LR-scaling note above
                patience_counter=early_stop.counter,
                best_val_loss=float(early_stop.best_loss),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                scaler_state=scaler.state_dict(),
                history=history,
            )
            break

        if _vram_status == "warn":
            # Reactive batch-size backoff (2026-08-08, per direct user
            # follow-up): this epoch's peak crossed into the 1GB advisory
            # buffer without threatening real exhaustion — reduce batch
            # size for subsequent epochs and keep training, rather than
            # stopping. DataLoader is rebuilt (PyTorch doesn't support
            # mutating an existing one's batch_size); the LR scheduler's
            # total_steps is rescaled to match the new steps-per-epoch
            # rate so the cosine curve still targets roughly the same
            # real-epoch horizon. NOTE: the optimizer's LR itself is NOT
            # re-scaled here (scaled_learning_rate() was only ever applied
            # once, at construction) — a known limitation flagged rather
            # than silently addressed, since re-deriving and reassigning
            # LR mid-run for a Lookahead-wrapped optimizer was a separate
            # judgment call this pass didn't cover.
            _old_steps_per_epoch = len(train_loader)
            cfg["batch_size"] = reduce_batch_size(
                cfg["batch_size"], len(train_ds), MIN_STEPS_PER_EPOCH,
                world_size=get_world_size(), device=device,
            )
            _global_batch_size = cfg["batch_size"] * get_world_size()
            train_loader = make_train_loader(train_ds, digit_train, letters_train_subset, cfg["batch_size"])
            scheduler.rescale_total_steps(_old_steps_per_epoch, len(train_loader))
            print(f"[Batch] Reduced to {cfg['batch_size']} for subsequent epochs")

        if early_stop.stop:
            break

        save_resume_state(
            cfg["resume_path"],
            epoch=epoch,
            batch_size=_global_batch_size,  # DDP: global, not per-rank — see the LR-scaling note above
            patience_counter=early_stop.counter,
            best_val_loss=float(early_stop.best_loss),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            scaler_state=scaler.state_dict(),
            history=history,
        )

    print(f"\n[Train] [{img_size}x{img_size}] Loading best checkpoint...")
    ckpt = torch.load(cfg["checkpoint_path"], map_location=device, weights_only=False)
    unwrap_model(model).load_state_dict(ckpt["state_dict"])

    print(f"\n[Eval] [{img_size}x{img_size}] Running per-class accuracy analysis on test set...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, per_class=True)
    print(f"\n{'='*40}")
    print(f"  [{img_size}x{img_size}] Router test accuracy : {test_acc:.6f}  ({test_acc*100:.4f}%)")
    print(f"  [{img_size}x{img_size}] Router test loss     : {test_loss:.6f}")
    print(f"{'='*40}")

    plot_history(history, cfg["plot_path"], img_size, title="Router Ranger")
    if is_main_process():
        # unwrap_model(): see common/distributed.py — a DDP-wrapped
        # model.state_dict() has "module."-prefixed keys, which would
        # silently fail to load into model_cpu (a plain, never-wrapped
        # model) a few lines below. Rank-0-only: several ranks writing
        # the same path at once would race.
        torch.save({
            "state_dict":  unwrap_model(model).state_dict(),
            "num_classes": NUM_CLASSES,
            "img_size":    img_size,
            "label_map":   LABEL_MAP,
        }, cfg["final_model_path"])
        size_mb = Path(cfg["final_model_path"]).stat().st_size / 1024**2
        print(f"[Save] {cfg['final_model_path']}  ({size_mb:.1f} MB)")

    if is_main_process():  # rank-0-only — see the final-model save above
        try:
            model_cpu = OCRRouterNet(NUM_CLASSES)
            model_cpu.load_state_dict(
                torch.load(cfg["final_model_path"], map_location="cpu", weights_only=False)["state_dict"]
            )
            export_onnx(model_cpu, cfg["onnx_path"], img_size)
        except Exception as e:
            print(f"[ONNX] Export failed: {e}")

    clear_resume_state(cfg["resume_path"])
    print(f"\n[Done] [{img_size}x{img_size}] All files saved to {OUTPUT_ROOT}")
    cleanup_distributed()
    return test_acc


# =============================================================================
# 6. REAL-WORLD SPOT-CHECK INFERENCE (single image)
# =============================================================================

def classify_image(model_path: str, image_path: str, img_size: int, device_str: str = "cpu",
                    unsure_floor: float = ROUTER_UNSURE_FLOOR):
    """
    Loads a trained router checkpoint (.pt, from run_training() above)
    and classifies a single real image file. Prints digit/UC/LC or [UNK]
    (per unsure_floor) with the model's confidence. Uses the same [0,1]
    grayscale-resize preprocessing the model was trained on — no
    cropping/segmentation (this model assumes an already-isolated
    character box, matching ocr_pipeline_mnist.py's normalize_char()
    contract).
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print("[Infer] Pillow/numpy are required for single-image inference.")
        return

    device = torch.device(device_str)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = OCRRouterNet(ckpt.get("num_classes", NUM_CLASSES)).to(device)
    unwrap_model(model).load_state_dict(ckpt["state_dict"])
    model.eval()

    img = Image.open(image_path).convert("L").resize((img_size, img_size))
    arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = model.predict_proba(arr)[0]
    top_idx = int(probs.argmax())
    top_conf = float(probs[top_idx])
    label = LABEL_MAP[top_idx] if top_conf >= unsure_floor else "[UNK]"
    print(f"[Infer] {image_path} → {label}  "
          f"(top={LABEL_MAP[top_idx]} conf={top_conf:.6f}, floor={unsure_floor}, "
          f"digit={probs[0]:.6f} UC={probs[1]:.6f} LC={probs[2]:.6f})")
    return label, probs


# =============================================================================
# 7. MAIN
# =============================================================================

def main(batch_override=None, gpu_id=None):
    print("\n" + "#" * 60)
    print(f"  ROUTER — {IMG_SIZE}x{IMG_SIZE} — DIGIT / UC / LC (+ [UNK] via confidence floor)")
    print(f"  Seed: GLOBAL_SEED={GLOBAL_SEED} (override with MNIST_SEED env var)")
    print("#" * 60 + "\n")

    acc = run_training(IMG_SIZE, batch_override=batch_override, gpu_id=gpu_id)

    print("\n" + "#" * 60)
    print("  TRAINING COMPLETE — ROUTER")
    print(f"  {IMG_SIZE}x{IMG_SIZE}: {acc*100:.4f}% test accuracy")
    print("#" * 60)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--batch-size', type=int, default=None,
                         help='Override auto batch detection with a fixed batch size')
    _parser.add_argument('--gpu', type=int, default=None,
                         help='Physical GPU index to train on (e.g. --gpu 1) — lets you '
                              'run different scripts on different cards at once on a '
                              'multi-GPU machine. Default: whatever CUDA already '
                              'considers the current device (GPU 0 on most single-GPU '
                              'machines).')
    _parser.add_argument('--infer', type=str, default=None,
                         help='Skip training — classify a single image file using the '
                              'trained checkpoint at v4_mnist_router_ranger_IMG_SIZE/*_final.pt')
    _parser.add_argument('--unsure-floor', type=float, default=ROUTER_UNSURE_FLOOR,
                         help=f'Confidence floor for --infer (default {ROUTER_UNSURE_FLOOR})')
    _args = _parser.parse_args()

    if _args.infer is not None:
        _model_path = str(OUTPUT_ROOT / f"v4_mnist_router_ranger_{IMG_SIZE}_final.pt")
        classify_image(_model_path, _args.infer, IMG_SIZE, unsure_floor=_args.unsure_floor)
    else:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        sys.stdout = _Tee(OUTPUT_ROOT / f"v4_mnist_router_ranger_{IMG_SIZE}_cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        main(batch_override=_args.batch_size, gpu_id=_args.gpu)
