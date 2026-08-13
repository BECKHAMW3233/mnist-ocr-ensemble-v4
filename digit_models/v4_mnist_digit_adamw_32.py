"""
v4_mnist_digit_adamw_32.py
====================
MNIST digit ensemble — OCRConvNetWide, Schedule-Free AdamW, 32x32
Pure PyTorch — Wider architecture with Squeeze-Excitation attention blocks.
Adapted from v2's mnist_adamw_32.py (32/64/128 tiers) — the
16x16 and 28x28 tiers are both new for v3, no v2 counterpart.
See v3_CHANGELOG.md for the full v3 restructure (optimizer
roster change, modularization, naming).

Architecture — OCRConvNetWide (unchanged from v2):
  Filter progression: 32→128→256→512
  Squeeze-Excitation (SE) attention after each stage
  StochasticDepth (DropPath) regularization
  Classifier head: 512→256→10
  ~9.7M parameters, ~37.1 MB (float32)

OPTIMIZER — Schedule-Free AdamW (unchanged from v2). Kept in the v3
roster alongside SOAP — see v3_CHANGELOG.md's roster-change entry for why
AdamW was kept while Lion, SGD, and AdaHessian were dropped.
  Eliminates the learning rate scheduler entirely through iterate averaging.
  Requires: pip install schedulefree
  Usage:
    - Must call optimizer.train() before each training epoch
    - Must call optimizer.eval() before validation/evaluation
  BatchNorm requirement:
    - Before each eval pass, run a 50-batch warm-up with model.train() +
      optimizer.eval() to update BatchNorm running stats at the averaged
      parameter point. Required per Schedule-Free docs for any BatchNorm model.

Digit sources at this resolution tier: see
supplementary_data.digit_sources_for_tier() — USPS is the ENTIRE
training spine at 16x16 (via load_base_usps(), not load_base_mnist() —
see load_mnist() below), since it's USPS's own native resolution and the
only source in that tier's ladder; MNIST/EMNIST Digits/SVHN/ARDIS IV
(every source except USPS) at 28x28; every source including USPS at
32/64/128. See v3_CHANGELOG.md's Resolution ladder split section for why there
are 5 resolution-tagged files per optimizer (16/28/32/64/128), not 4.

Output: ./v4_mnist_digit_adamw_32/  (created next to this script)
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================
import sys
import argparse
import itertools
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
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
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
from common.amp import amp_train_step
from common.optimizers import (
    build_adamw_sf, ADAMW_SF_LR as LEARNING_RATE, ADAMW_SF_WEIGHT_DECAY as WEIGHT_DECAY,
)

if HAS_PSUTIL:
    import psutil

try:
    from supplementary_data import (
        load_supplementary, load_base_mnist, load_base_usps, digit_sources_for_tier,
        _extract_targets,
    )
    HAS_SUPPLEMENTARY = True
except ImportError:
    HAS_SUPPLEMENTARY = False
    print("[Warning] supplementary_data.py not found — digit supplementary data unavailable")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

NUM_CLASSES      = 10
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

RAM_RESERVE_GB   = 4.0

DATA_DIR         = Path(r"E:\CSC-114\emnist-model\datasets\pytorch")
# ^ Specific to the original project machine — change this to wherever
# YOU want MNIST/EMNIST/USPS/SVHN to be downloaded/read from on your own
# system (see supplementary_data.py's own DATA_DIR for the canonical
# version of this path).
IMG_SIZE         = 32
# OUTPUT_ROOT is script-relative (Path(__file__).resolve().parent / ...) —
# safe as-is on any machine, unlike DATA_DIR above. It creates its own
# output folder next to wherever this script actually runs from, so it
# does not need to be personalized for a different system.
OUTPUT_ROOT      = Path(__file__).resolve().parent / f"v4_mnist_digit_adamw_{IMG_SIZE}"

LABEL_MAP = list("0123456789")


def build_config(img_size: int, batch_size: int) -> dict:
    prefix = f"v4_mnist_digit_adamw_{img_size}"
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
    aug_transforms = [
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=5),
        transforms.ColorJitter(contrast=0.3, brightness=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
    ] if augment else []
    base_transforms = [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
    return transforms.Compose(aug_transforms + base_transforms)


def get_class_weights(dataset) -> torch.Tensor:
    print("[Dataset] Computing class weights for balanced sampling...")
    targets = _extract_targets(dataset)
    class_counts  = torch.bincount(targets, minlength=NUM_CLASSES).float().clamp(min=1)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]
    print(f"[Dataset] Class weight range: {class_weights.min():.6f} — {class_weights.max():.6f}")
    return sample_weights


def load_mnist(data_dir: Path, img_size: int):
    """
    Loads this resolution tier's digit spine plus supplementary sources.
    At 16x16, USPS IS the spine (load_base_usps()); at every other tier,
    base MNIST is the spine (load_base_mnist()) plus whatever
    digit_sources_for_tier() adds on top.
    """
    base_loader = load_base_usps if img_size == 16 else load_base_mnist
    print("[Dataset] Loading USPS..." if img_size == 16 else "[Dataset] Loading MNIST...")
    train_ds, val_ds, test_ds = base_loader(
        data_dir=data_dir,
        train_transform=get_transforms(img_size, augment=True),
        val_transform=get_transforms(img_size, augment=False),
        test_transform=get_transforms(img_size, augment=False),
        validation_split=VALIDATION_SPLIT,
        split_seed=GLOBAL_SEED,
    )

    supp_ds = None
    if HAS_SUPPLEMENTARY:
        print("[Dataset] Loading digit supplementary data...")
        supp_ds = load_supplementary(
            transform=get_transforms(img_size, augment=True),
            train=True,
            **digit_sources_for_tier(img_size),
        )
        if supp_ds is not None:
            train_ds = ConcatDataset([train_ds, supp_ds])
            print(f"[Dataset] Combined training set: {len(train_ds):,} samples")

    return train_ds, val_ds, test_ds, supp_ds


def make_dataloader(dataset, batch_size: int, shuffle: bool = False,
                    use_weighted_sampler: bool = False,
                    num_workers_override: int = None) -> DataLoader:
    _nw = num_workers_override if num_workers_override is not None else NUM_WORKERS
    _persistent = _nw > 0
    if use_weighted_sampler:
        sample_weights = get_class_weights(dataset)
        if is_distributed():
            sampler = DistributedWeightedRandomSampler(sample_weights, len(sample_weights))
        else:
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=_nw, pin_memory=torch.cuda.is_available(),
                          persistent_workers=_persistent, drop_last=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=_nw, pin_memory=torch.cuda.is_available(),
                      persistent_workers=_persistent, drop_last=False)


# =============================================================================
# 3. MODEL ARCHITECTURE — OCRConvNetWide (unchanged from v2)
# =============================================================================

class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob


class SEResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, drop_path: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.se    = SqueezeExcitation(out_ch)
        self.drop_path = StochasticDepth(drop_path)
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        elif in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        x = self.drop_path(x)
        return F.relu(x + residual, inplace=True)


class OCRConvNetWide(nn.Module):
    """
    Wider OCR ConvNet with SE attention.
    Input:  (batch, 1, IMG_SIZE, IMG_SIZE)
    Output: (batch, 10)
    Filter progression: 32→128→256→512 — downsamples via stride=2 in each
    stage's first conv (2026-08-07 fix: previously pooled via a separate
    MaxPool2d AFTER two full-resolution convs — ~25x more compute per image
    than necessary, no accuracy gain). Shortcut/projection path uses
    avg-pool(stride=2) + stride-1 1x1 conv rather than a stride-2 1x1 conv,
    per He et al. 2018's "ResNet-D" tweak (a stride-2 1x1 conv discards 3/4
    of input pixels) — see v3_CHANGELOG.md.
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = SEResidualBlock(32,  128, stride=2)
        self.stage2 = SEResidualBlock(128, 256, stride=2)
        self.stage3 = SEResidualBlock(256, 512, stride=2)
        self.stage4 = SEResidualBlock(512, 512, stride=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)


# =============================================================================
# 4. TRAINING LOOP
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    epoch: int = None, img_size: int = None) -> tuple:
    """optimizer.train() must already have been called by the caller (see
    run_training()) — Schedule-Free AdamW needs this to switch from
    averaging mode to update mode before training."""
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
    # DistributedWeightedRandomSampler in make_dataloader()/
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
    """optimizer.eval() + the BatchNorm warm-up pass must already have run
    (see run_training()) before this is called."""
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
        worst = class_acc.argsort()[:10]
        print("\n  [Per-Class] 10 worst-performing classes:")
        for idx in worst:
            print(f"    '{LABEL_MAP[idx]}' (class {idx:2d}): "
                  f"{class_acc[idx]*100:.4f}%  ({int(class_total[idx])} samples)")

    return total_loss / total_samples, total_correct / total_samples


def _batchnorm_warmup(model, optimizer, train_loader, device, n_batches: int = 50):
    """Schedule-Free AdamW requirement: update BatchNorm running stats at
    the averaged parameter point (optimizer.eval() mode) before any real
    evaluation — see module docstring."""
    model.train()
    optimizer.eval()
    with torch.no_grad():
        for _imgs, _ in itertools.islice(train_loader, n_batches):
            model(_imgs.to(device))
    model.eval()


# =============================================================================
# 5. RUN TRAINING
# =============================================================================

def run_training(img_size: int, batch_override: int = None, gpu_id: int = None):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = setup_device(use_amp=USE_AMP,
                          gpu_id=get_local_rank() if is_distributed() else gpu_id)
    setup_distributed(device)

    if batch_override is not None:
        batch_size = batch_override
        print(f"[Batch] Override: using batch size {batch_size} (skipping auto-detect)")
    else:
        batch_size = determine_batch_size(
            build_model=lambda: OCRConvNetWide(NUM_CLASSES),
            build_optimizer=lambda params: build_adamw_sf(params, warmup_steps=5),
            criterion=nn.CrossEntropyLoss(label_smoothing=0.05),
            img_size=img_size, num_classes=NUM_CLASSES, device=device,
            use_amp=USE_AMP, grad_clip_norm=1.0,
        )
    cfg = build_config(img_size, batch_size)

    print("=" * 60)
    print(f"  MNIST Digit Ensemble — ScheduleFree-AdamW  [{img_size}x{img_size}]")
    print(f"  PyTorch {torch.__version__}  |  AMP: {USE_AMP}")
    print(f"  Output: {OUTPUT_ROOT}")
    print(f"  Resolution: {img_size}x{img_size}  |  Batch: {cfg['batch_size']} {'(override)' if batch_override else '(auto-detected)'}")
    print(f"  Optimizer: Schedule-Free AdamW")
    print(f"  Digit sources this tier: {digit_sources_for_tier(img_size)}")
    print(f"  Normalization: [0,1]")
    print("=" * 60)

    train_ds, val_ds, test_ds, supp_ds = load_mnist(DATA_DIR, img_size)

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
    train_loader = make_dataloader(train_ds, cfg["batch_size"], use_weighted_sampler=True)
    val_loader   = make_dataloader(val_ds,  cfg["batch_size"], num_workers_override=0)
    test_loader  = make_dataloader(test_ds, cfg["batch_size"], num_workers_override=0)

    model = OCRConvNetWide(NUM_CLASSES).to(device)
    model = wrap_model_ddp(model, device)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] OCRConvNetWide — {img_size}x{img_size}")
    print(f"  Parameters : {total:,}")
    print(f"  Est. size  : {total * 4 / 1024**2:.1f} MB (float32)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = build_adamw_sf(model.parameters(), warmup_steps=len(train_loader))  # 1 epoch warmup
    print(f"[Optimizer] Schedule-Free AdamW  lr={LEARNING_RATE}  wd={WEIGHT_DECAY}")
    print(f"            warmup_steps={len(train_loader)} (1 epoch)")

    scaler     = torch.amp.GradScaler('cuda', enabled=USE_AMP and device.type == "cuda")
    early_stop = EarlyStopping(patience=PATIENCE, path=cfg["checkpoint_path"])

    print(f"\n[Train] Starting {img_size}x{img_size} — no epoch cap | batch: {cfg['batch_size']} | patience: {PATIENCE}")
    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "lr"]}

    # ── Resume from previous session ─────────────────────────────────────────
    start_epoch = 1
    resume_path = Path(cfg["resume_path"])
    if resume_path.exists() and Path(cfg["checkpoint_path"]).exists():
        try:
            _rs = torch.load(str(resume_path), map_location=device, weights_only=False)
            _ckpt = torch.load(cfg["checkpoint_path"], map_location=device, weights_only=False)
            unwrap_model(model).load_state_dict(_ckpt["state_dict"] if "state_dict" in _ckpt else _ckpt)
            restore_optimizer_scheduler_state(_rs, cfg["batch_size"], optimizer)
            scaler.load_state_dict(_rs["scaler_state"])
            early_stop.counter   = _rs["patience_counter"]
            early_stop.best_loss = _rs["best_val_loss"]
            history              = _rs["history"]
            start_epoch          = _rs["epoch"] + 1
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
        print(f"[RAM] Swap/page-file baseline for this run: {_run_swap_baseline_gb:.3f} GB "
              f"— every epoch below is checked against THIS fixed value, not a rolling one.")

    for epoch in range(start_epoch, 10**6):
        # DDP epoch reshuffle (2026-07-28, per direct user follow-up):
        # DistributedWeightedRandomSampler's weighted draw is seeded by
        # seed + epoch (see common/distributed.py) — without telling it
        # which epoch this is, every epoch would draw the identical
        # weighted sample. No-op when not running distributed.
        if is_distributed():
            train_loader.sampler.set_epoch(epoch)
        t0 = time.time()

        optimizer.train()
        try:
            train_loss, train_acc, hw, hw_raw = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device,
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
                patience_counter=early_stop.counter,
                best_val_loss=float(early_stop.best_loss),
                optimizer_state=optimizer.state_dict(),
                scaler_state=scaler.state_dict(),
                history=history,
                batch_size=cfg["batch_size"],
            )
            break

        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            _batchnorm_warmup(model, optimizer, train_loader, device)
        except RuntimeError as _bn_e:
            _bn_emsg = str(_bn_e).lower()
            if not ("out of memory" in _bn_emsg or "find was unable" in _bn_emsg or "engine" in _bn_emsg or "unknown error" in _bn_emsg):
                raise
            print(f"  [OOM] CUDA error during BatchNorm warmup, epoch {epoch}: {_bn_e}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            save_resume_state(
                cfg["resume_path"],
                epoch=epoch - 1,
                patience_counter=early_stop.counter,
                best_val_loss=float(early_stop.best_loss),
                optimizer_state=optimizer.state_dict(),
                scaler_state=scaler.state_dict(),
                history=history,
                batch_size=cfg["batch_size"],
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
                patience_counter=early_stop.counter,
                best_val_loss=float(early_stop.best_loss),
                optimizer_state=optimizer.state_dict(),
                scaler_state=scaler.state_dict(),
                history=history,
                batch_size=cfg["batch_size"],
            )
            break

        if _vram_status == "warn":
            # Reactive batch-size backoff (2026-08-08, per direct user
            # follow-up): this epoch's peak crossed into the 1GB advisory
            # buffer without threatening real exhaustion — reduce batch
            # size for subsequent epochs and keep training, rather than
            # stopping. DataLoader is rebuilt (PyTorch doesn't support
            # mutating an existing one's batch_size). No LR-schedule
            # rescale needed here — Schedule-Free AdamW has no scheduler
            # object (iterate averaging replaces it entirely).
            cfg["batch_size"] = reduce_batch_size(
                cfg["batch_size"], len(train_ds), MIN_STEPS_PER_EPOCH,
                world_size=get_world_size(), device=device,
            )
            train_loader = make_dataloader(train_ds, cfg["batch_size"], use_weighted_sampler=True)
            print(f"[Batch] Reduced to {cfg['batch_size']} for subsequent epochs")

        if early_stop.stop:
            break

        save_resume_state(
            cfg["resume_path"],
            epoch=epoch,
            patience_counter=early_stop.counter,
            best_val_loss=float(early_stop.best_loss),
            optimizer_state=optimizer.state_dict(),
            scaler_state=scaler.state_dict(),
            history=history,
            batch_size=cfg["batch_size"],
        )

    print(f"\n[Train] [{img_size}x{img_size}] Loading best checkpoint...")
    ckpt = torch.load(cfg["checkpoint_path"], map_location=device, weights_only=False)
    unwrap_model(model).load_state_dict(ckpt["state_dict"])

    print(f"\n[Eval] [{img_size}x{img_size}] Running per-class accuracy analysis on test set...")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _batchnorm_warmup(model, optimizer, train_loader, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, per_class=True)
    print(f"\n{'='*40}")
    print(f"  [{img_size}x{img_size}] AdamW Test accuracy : {test_acc:.6f}  ({test_acc*100:.4f}%)")
    print(f"  [{img_size}x{img_size}] AdamW Test loss     : {test_loss:.6f}")
    print(f"{'='*40}")

    plot_history(history, cfg["plot_path"], img_size, title="MNIST Digit Ensemble ScheduleFree-AdamW")
    if is_main_process():
        # unwrap_model(): see common/distributed.py — a DDP-wrapped
        # model.state_dict() has "module."-prefixed keys, which would
        # silently fail to load into model_cpu (a plain, never-wrapped
        # model) a few lines below. Rank-0-only: several ranks writing
        # the same path at once would race.
        torch.save({"state_dict": unwrap_model(model).state_dict()}, cfg["final_model_path"])
        print(f"[Save] {cfg['final_model_path']}")

    if is_main_process():  # rank-0-only — see the final-model save above
        try:
            model_cpu = OCRConvNetWide(NUM_CLASSES)
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
# 6. MAIN
# =============================================================================

def main(batch_override=None, gpu_id=None):
    print("\n" + "#" * 60)
    print(f"  MNIST DIGIT ENSEMBLE — SCHEDULEFREE-ADAMW {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Seed: GLOBAL_SEED={GLOBAL_SEED} (override with MNIST_SEED env var)")
    print("#" * 60 + "\n")

    acc = run_training(IMG_SIZE, batch_override=batch_override, gpu_id=gpu_id)

    print("\n" + "#" * 60)
    print("  TRAINING COMPLETE — SCHEDULEFREE-ADAMW")
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
    _args = _parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(OUTPUT_ROOT / f"v4_mnist_digit_adamw_{IMG_SIZE}_cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    main(batch_override=_args.batch_size, gpu_id=_args.gpu)
