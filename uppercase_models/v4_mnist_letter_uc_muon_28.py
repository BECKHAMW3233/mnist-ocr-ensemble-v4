"""
v4_mnist_letter_uc_muon_28.py
====================
Uppercase letter-identity ensemble — OCRConvNetMuon, Muon optimizer, 28x28.
Pure PyTorch. Same architecture and hyperparameters as
v4_mnist_digit_muon_28.py (the digit ensemble), retargeted at a 26-class
uppercase A-Z letter-identity problem — see v3_CHANGELOG.md for the full
rationale (why EMNIST ByClass over EMNIST Balanced, why no digit mixing,
why no 16x16 tier).

Architecture — OCRConvNetMuon (unchanged from the digit ensemble except
the classifier head's output width, 10 -> 26):
  Filter progression: 64→128→256→512, plain residual blocks (Conv-BN-
  GELU x2 + shortcut) — no SE attention, no stochastic depth.
  Classifier head: 512→256→26

OPTIMIZER — Muon (see common/optimizers.py for the full algorithm/
implementation — centralized there 2026-08-08, reversing this file's
original "implemented directly here" design, matching v4_mnist_digit_muon_28.py's
own convention; see v3_CHANGELOG.md's 2026-07-27 and 2026-08-08 entries
for both sides of that history). Identical hyperparameters to v4_mnist_digit_muon_28.py —
see that file's own
docstring / v3_CHANGELOG.md for the full tuning rationale, unchanged here:
    Muon group   : lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.01
    AdamW fallback group: lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01
  300-step linear warmup before cosine decay (common/scheduler.py)
  PATIENCE=15
  AMP enabled + gradient clipping max_norm=1.0

Data source: EMNIST ByClass, uppercase portion only (byclass 10-35,
remapped to dense 0-25 — see supplementary_data.load_base_emnist_letters()).
No digit classes, no lowercase classes, no supplementary sources, and no
per-resolution source ladder — every letter-model resolution tier
(28/32/64/128; there is no 16x16 letter tier, that was USPS-digit-only)
loads identically.

Output: ./v4_mnist_letter_uc_muon_28/  (created next to this script)
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================
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
from torch.utils.data import DataLoader, WeightedRandomSampler
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
from common.optimizers import (
    build_muon, MUON_LR, MUON_MOMENTUM, MUON_WD, NS_STEPS,
    MUON_ADAMW_LR as ADAMW_LR, MUON_ADAMW_BETAS as ADAMW_BETAS,
    MUON_ADAMW_EPS as ADAMW_EPS, MUON_ADAMW_WD as ADAMW_WD,
)


if HAS_PSUTIL:
    import psutil

try:
    from supplementary_data import load_base_emnist_letters
    HAS_SUPPLEMENTARY = True
except ImportError:
    HAS_SUPPLEMENTARY = False
    print("[Warning] supplementary_data.py not found — this model requires it "
          "(EMNIST ByClass is its sole data source); training will fail without it.")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

LETTER_CASE   = "upper"
NUM_CLASSES   = 26
LABEL_MAP     = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
PATIENCE      = 15
NUM_WORKERS   = usable_cpu_count()  # auto-scales to real core count,
                        # leaving 25% for the OS — see common/seeding.py.
MIN_STEPS_PER_EPOCH = 15  # floor on real gradient-update steps per epoch — see
                          # cap_batch_size_for_min_steps() in common/batch_sizing.py
USE_AMP       = True


WARMUP_STEPS  = 300
LABEL_SMOOTH  = 0.05
SCHEDULE_EPOCH_ESTIMATE = 150  # shapes the cosine LR curve only, not a hard epoch cap —
                                # PATIENCE-based early stopping is what actually ends training.

RAM_RESERVE_GB = 4.0

DATA_DIR      = Path(r"E:\CSC-114\emnist-model\datasets\pytorch")
# ^ Specific to the original project machine — change this to wherever
# YOU want EMNIST to be downloaded/read from on your own system (see
# supplementary_data.py's own DATA_DIR for the canonical version of this
# path).
IMG_SIZE      = 28
# OUTPUT_ROOT is script-relative (Path(__file__).resolve().parent / ...) —
# safe as-is on any machine, unlike DATA_DIR above. It creates its own
# output folder next to wherever this script actually runs from, so it
# does not need to be personalized for a different system.
OUTPUT_ROOT   = Path(__file__).resolve().parent / f"v4_mnist_letter_uc_muon_{IMG_SIZE}"


def build_config(img_size: int, batch_size: int) -> dict:
    prefix = f"v4_mnist_letter_uc_muon_{img_size}"
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


def load_letters(img_size: int):
    """
    Loads the uppercase 26-class letter split from EMNIST ByClass — this
    model's sole data source. No supplementary sources, no per-resolution
    source ladder — every letter-model resolution tier calls this
    identically.
    """
    train_transform = get_transforms(img_size, augment=True)
    test_transform  = get_transforms(img_size, augment=False)

    train_ds, val_ds, test_ds, train_targets = load_base_emnist_letters(
        case=LETTER_CASE,
        train_transform=train_transform,
        val_transform=test_transform,
        test_transform=test_transform,
        validation_split=0.15,
        split_seed=GLOBAL_SEED,
        return_train_targets=True,
    )
    return train_ds, val_ds, test_ds, train_targets


def make_dataloader(train_ds, train_targets, batch_size: int):
    class_counts   = torch.bincount(train_targets, minlength=NUM_CLASSES).float().clamp(min=1)
    class_weights  = 1.0 / class_counts
    sample_weights = class_weights[train_targets]

    if is_distributed():
        sampler = DistributedWeightedRandomSampler(sample_weights, len(sample_weights))
    else:
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                      num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0)


def make_eval_dataloader(dataset, batch_size: int):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=True, persistent_workers=False)


# =============================================================================
# 3. ARCHITECTURE — OCRConvNetMuon
# =============================================================================

class MuonResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + self.shortcut(x))


class OCRConvNetMuon(nn.Module):
    """
    Input:  (batch, 1, IMG_SIZE, IMG_SIZE)
    Output: (batch, 26)
    Filter progression: 64→128→256→512, plain residual blocks.
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.stage1 = MuonResidualBlock(64, 128, stride=2)
        self.stage2 = MuonResidualBlock(128, 256, stride=2)
        self.stage3 = MuonResidualBlock(256, 512, stride=2)
        self.stage4 = MuonResidualBlock(512, 512, stride=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)


# =============================================================================
# 2b. MUON OPTIMIZER — see common/optimizers.py (centralized 2026-08-08;
#     was previously implemented directly in this file, matching v4_mnist_digit_muon_28.py
#     — see v3_CHANGELOG.md's 2026-07-27 and 2026-08-08 entries for why).
# =============================================================================

# =============================================================================
# 4. TRAINING / EVALUATION
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
        worst = class_acc.argsort()[:10]
        print("\n  [Per-Class] 10 worst-performing classes:")
        for idx in worst:
            print(f"    '{LABEL_MAP[idx]}' (class {idx:2d}): "
                  f"{class_acc[idx]*100:.4f}%  ({int(class_total[idx])} samples)")

    return total_loss / total_samples, total_correct / total_samples


# =============================================================================
# 5. TRAINING PROCEDURE
# =============================================================================

def run_training(img_size: int, batch_override: int = None, gpu_id: int = None):
    if not HAS_SUPPLEMENTARY:
        print("[Error] supplementary_data.py is required (EMNIST ByClass is "
              "this model's sole data source) and was not found — cannot train.")
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
            build_model=lambda: OCRConvNetMuon(NUM_CLASSES),
            build_optimizer=build_muon,
            criterion=nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH),
            img_size=img_size, num_classes=NUM_CLASSES, device=device,
            use_amp=USE_AMP, grad_clip_norm=1.0,
        )
    cfg = build_config(img_size, batch_size)

    print("=" * 60)
    print(f"  Letter Identity Ensemble — Uppercase — Muon  [{img_size}x{img_size}]")
    print(f"  PyTorch {torch.__version__}  |  AMP: {USE_AMP}")
    print(f"  Output: {OUTPUT_ROOT}")
    print(f"  Resolution: {img_size}x{img_size}  |  Batch: {cfg['batch_size']} {'(override)' if batch_override else '(auto-detected)'}")
    print(f"  Optimizer: Muon (Newton-Schulz orthogonalized momentum) + AdamW fallback")
    print(f"  Letter source: EMNIST ByClass, case={LETTER_CASE} (26 classes, no digits)")
    print(f"  Normalization: [0,1]")
    print("=" * 60)

    train_ds, val_ds, test_ds, train_targets = load_letters(img_size)

    _min_steps_batch = cap_batch_size_for_min_steps(
        cfg["batch_size"], len(train_ds), MIN_STEPS_PER_EPOCH,
        world_size=get_world_size(),
    )
    if _min_steps_batch < cfg["batch_size"]:
        print(f"[Batch] Capping batch size {cfg['batch_size']} -> {_min_steps_batch} "
              f"to guarantee >= {MIN_STEPS_PER_EPOCH} steps/epoch "
              f"({len(train_ds):,} train samples)")
        cfg["batch_size"] = _min_steps_batch
    train_loader = make_dataloader(train_ds, train_targets, cfg["batch_size"])
    val_loader   = make_eval_dataloader(val_ds,  cfg["batch_size"])
    test_loader  = make_eval_dataloader(test_ds, cfg["batch_size"])

    model = OCRConvNetMuon(NUM_CLASSES).to(device)
    model = wrap_model_ddp(model, device)
    total = sum(p.numel() for p in model.parameters())
    n_muon  = sum(p.numel() for p in model.parameters() if p.ndim >= 2)
    n_other = total - n_muon
    print(f"\n[Model] OCRConvNetMuon — {img_size}x{img_size}")
    print(f"  Parameters : {total:,}  (Muon group: {n_muon:,}  |  AdamW-fallback group: {n_other:,})")
    print(f"  Est. size  : {total * 4 / 1024**2:.1f} MB (float32)")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    optimizer = build_muon(model.parameters())
    total_steps = SCHEDULE_EPOCH_ESTIMATE * len(train_loader)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_STEPS, total_steps)
    print(f"[Optimizer] Muon  lr={MUON_LR}  momentum={MUON_MOMENTUM}  wd={MUON_WD}  ns_steps={NS_STEPS}")
    print(f"            AdamW fallback  lr={ADAMW_LR}  betas={ADAMW_BETAS}  wd={ADAMW_WD}")
    print(f"[Schedule]  warmup_steps={WARMUP_STEPS}  total_steps={total_steps:,}")

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
            restore_optimizer_scheduler_state(_rs, cfg["batch_size"], optimizer, scheduler)
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
        if is_distributed():
            train_loader.sampler.set_epoch(epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
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
                patience_counter=early_stop.counter,
                best_val_loss=float(early_stop.best_loss),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                scaler_state=scaler.state_dict(),
                history=history,
                batch_size=cfg["batch_size"],
            )
            break
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        current_lr = scheduler.get_lr()[0]
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
                scheduler_state=scheduler.state_dict(),
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
            # mutating an existing one's batch_size); the LR scheduler's
            # total_steps is rescaled to match the new steps-per-epoch
            # rate so the cosine curve still targets roughly the same
            # real-epoch horizon.
            _old_steps_per_epoch = len(train_loader)
            cfg["batch_size"] = reduce_batch_size(
                cfg["batch_size"], len(train_ds), MIN_STEPS_PER_EPOCH,
                world_size=get_world_size(), device=device,
            )
            train_loader = make_dataloader(train_ds, train_targets, cfg["batch_size"])
            scheduler.rescale_total_steps(_old_steps_per_epoch, len(train_loader))
            print(f"[Batch] Reduced to {cfg['batch_size']} for subsequent epochs")

        if early_stop.stop:
            break

        save_resume_state(
            cfg["resume_path"],
            epoch=epoch,
            patience_counter=early_stop.counter,
            best_val_loss=float(early_stop.best_loss),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            scaler_state=scaler.state_dict(),
            history=history,
            batch_size=cfg["batch_size"],
        )

    print(f"\n[Train] [{img_size}x{img_size}] Loading best checkpoint...")
    ckpt = torch.load(cfg["checkpoint_path"], map_location=device, weights_only=False)
    unwrap_model(model).load_state_dict(ckpt["state_dict"])

    print(f"\n[Eval] [{img_size}x{img_size}] Running per-class accuracy analysis on test set...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, per_class=True)
    print(f"\n{'='*40}")
    print(f"  [{img_size}x{img_size}] Muon Test accuracy : {test_acc:.6f}  ({test_acc*100:.4f}%)")
    print(f"  [{img_size}x{img_size}] Muon Test loss     : {test_loss:.6f}")
    print(f"{'='*40}")

    plot_history(history, cfg["plot_path"], img_size, title="Letter Identity Ensemble (Uppercase) Muon")
    if is_main_process():
        torch.save({"state_dict": unwrap_model(model).state_dict()}, cfg["final_model_path"])
        print(f"[Save] {cfg['final_model_path']}")

    if is_main_process():
        try:
            model_cpu = OCRConvNetMuon(NUM_CLASSES)
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
    print(f"  LETTER IDENTITY ENSEMBLE — UPPERCASE — MUON {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Seed: GLOBAL_SEED={GLOBAL_SEED} (override with MNIST_SEED env var)")
    print("#" * 60 + "\n")

    acc = run_training(IMG_SIZE, batch_override=batch_override, gpu_id=gpu_id)

    print("\n" + "#" * 60)
    print("  TRAINING COMPLETE — MUON (UPPERCASE)")
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
    sys.stdout = _Tee(OUTPUT_ROOT / f"v4_mnist_letter_uc_muon_{IMG_SIZE}_cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    main(batch_override=_args.batch_size, gpu_id=_args.gpu)
