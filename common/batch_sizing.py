"""
common/batch_sizing.py
=======================
Auto batch-size detection: bottom-up coarse pass through the CANDIDATES
ladder below, stopping at the first failure, then binary-search
refinement rounded to the nearest multiple of 8, landing within 8 of the
true VRAM/RAM ceiling.

Candidate ladder: future-proofed for hardware beyond the current RTX
4080 16GB, up to a 300GB-VRAM-per-device ceiling. Doubles 64 -> 2048 (6
candidates), then steps by a constant +1536 up to 306176, capped with
one final point at 307200 (= 300 x 1024, mirroring the same "round GB
boundary" convention used at 98304 = 96 x 1024 earlier in the ladder) —
205 candidates total. This is a no-op ceiling on today's 16GB card (the
coarse pass below stops at the FIRST candidate that fails, so a long
tail of never-reached candidates costs nothing) — the same file needs no
further edits if/when bigger hardware becomes available; it'll just
naturally probe further up this same ladder. Whether the ladder's top is
ever reached in practice still depends heavily on image resolution —
per-sample activation memory grows with img_size far more than with
available VRAM, so the smaller resolution tiers (e.g. 16x16) are far
more likely to approach the top of this ladder than 64x64/128x128 will
on any single device. Note this probes PER-DEVICE VRAM
(torch.cuda.get_device_properties(gpu_idx).total_memory on whichever
single GPU the process is bound to) — a cluster's aggregate memory
across many GPUs doesn't raise this ceiling unless a single device
itself has more VRAM.

The AMP-vs-not and gradient-clipping-vs-not branches below are selected
by a parameter (SOAP: no AMP, no clipping; every other optimizer,
including the router: AMP + clip max_norm=1.0). The AMP branch delegates
to common/amp.py's amp_train_step() rather than repeating the autocast/
GradScaler/clip/step mechanics a second time in this file — the
batch-size probe and the real training loop share the exact same
mixed-precision step function, not just a close copy of it.

Callers supply:
  build_model()         -> a fresh, untrained nn.Module (not yet .to(device))
  build_optimizer(params) -> an optimizer instance for those params
  criterion              -> a loss module (e.g. nn.CrossEntropyLoss)
everything else (candidate ladder, OOM detection, refinement) is shared.
"""
import subprocess

import torch
import torch.nn as nn

from .telemetry import HAS_PSUTIL
from .amp import amp_train_step, build_grad_scaler
from .distributed import all_reduce_min

if HAS_PSUTIL:
    import psutil

RAM_RESERVE_GB = 4.0  # reserved for the OS on CPU-only runs — higher than
                      # the GPU path's VRAM_RESERVE_GB below since idle
                      # system RAM usage is meaningfully higher than idle
                      # VRAM usage on a typical desktop OS.

VRAM_RESERVE_GB = 1.0  # adamw_*.py/router_ranger_*.py never call
                       # torch.cuda.reset_peak_memory_stats() per epoch
                       # (unlike soap_*.py/muon_*.py, which do), so their
                       # console VRAM telemetry is a cumulative peak since
                       # the batch-size probe ran, not a clean per-epoch
                       # reading.


def _probe_batch_size(bs: int, build_model, build_optimizer, criterion,
                       img_size: int, num_classes: int, device: torch.device,
                       probe_steps: int, use_amp: bool,
                       grad_clip_norm) -> bool:
    """
    Runs probe_steps real training steps (forward, backward,
    optimizer.step) at a single candidate batch size with a real
    optimizer instance. Returns True if it fits, False if it OOMs or
    spills to shared VRAM/system RAM.
    """
    _on_cuda = device.type == "cuda"
    # Multi-GPU support: uses the actual `device` object already passed
    # into this function (device.index) rather than querying
    # torch.cuda.current_device() fresh, since `device` is the
    # authoritative source of which GPU this probe is actually meant to
    # run on — get_device_properties(0) always means the PHYSICAL first
    # GPU, unlike memory_reserved()/empty_cache()/etc. below, which
    # correctly operate on whatever CUDA's "current device" is.
    _gpu_idx = (device.index if device.index is not None else torch.cuda.current_device()) if _on_cuda else 0
    _candidate_swap_baseline_gb = round(psutil.swap_memory().used / 1024**3, 3) if HAS_PSUTIL else 0.0

    try:
        if _on_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _free_vram = (torch.cuda.get_device_properties(_gpu_idx).total_memory
                         - torch.cuda.memory_reserved()) / 1024**3
            # separate, cheap pre-flight check (not tied to VRAM_RESERVE_GB
            # below) — just decides whether this candidate is even worth
            # attempting
            if _free_vram < 1.5:
                print(f'[Batch] Batch size {bs} — insufficient free VRAM ({_free_vram:.1f}GB free), skipping')
                return False
        elif HAS_PSUTIL:
            _free_ram = psutil.virtual_memory().available / 1024**3
            if _free_ram < RAM_RESERVE_GB:
                print(f'[Batch] Batch size {bs} — insufficient free RAM ({_free_ram:.1f}GB free, {RAM_RESERVE_GB}GB reserved), skipping')
                return False

        model_test = build_model().to(device)
        probe_optimizer = build_optimizer(model_test.parameters())
        if hasattr(probe_optimizer, "train"):
            probe_optimizer.train()  # Schedule-Free-style optimizers (e.g. AdamWScheduleFree)
                                     # require this before step(); absent on standard
                                     # torch.optim.Optimizer subclasses (SOAP, Muon, Ranger)
        probe_scaler = build_grad_scaler(device, use_amp) if use_amp else None

        dummy_x = torch.rand(bs, 1, img_size, img_size, device=device)
        dummy_y = torch.randint(0, num_classes, (bs,), device=device)

        for _step in range(probe_steps):
            probe_optimizer.zero_grad()
            if use_amp:
                loss, out, _stepped = amp_train_step(
                    model_test, dummy_x, dummy_y, criterion, probe_optimizer, probe_scaler,
                    device, use_amp=use_amp, grad_clip_norm=grad_clip_norm,
                )
            else:
                out  = model_test(dummy_x)
                loss = criterion(out, dummy_y)
                loss.backward()
                if grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model_test.parameters(), max_norm=grad_clip_norm)
                probe_optimizer.step()

            if _on_cuda:
                try:
                    _smi = subprocess.check_output(
                        ['nvidia-smi', f'-i={_gpu_idx}', '--query-gpu=memory.used',
                         '--format=csv,noheader,nounits'],
                        timeout=3
                    ).decode().strip()
                    _used_mb = float(_smi.split('\n')[0].strip())
                    _total_vram_mb = torch.cuda.get_device_properties(_gpu_idx).total_memory / 1024**2
                    _vram_ceiling_mb = _total_vram_mb - (VRAM_RESERVE_GB * 1024)
                    if _used_mb > _vram_ceiling_mb:
                        raise RuntimeError('out of memory')
                except subprocess.SubprocessError:
                    _total_vram_gb = torch.cuda.get_device_properties(_gpu_idx).total_memory / 1024**3
                    if torch.cuda.max_memory_reserved() / 1024**3 > (_total_vram_gb - VRAM_RESERVE_GB):
                        raise RuntimeError('out of memory')
            elif HAS_PSUTIL:
                _free_ram = psutil.virtual_memory().available / 1024**3
                if _free_ram < RAM_RESERVE_GB:
                    raise RuntimeError('out of memory')
                if round(psutil.swap_memory().used / 1024**3, 3) > _candidate_swap_baseline_gb:
                    raise RuntimeError("page file usage grew beyond this candidate's own baseline — treated as OOM")

        del model_test, probe_optimizer, probe_scaler, dummy_x, dummy_y, out, loss
        if _on_cuda:
            torch.cuda.empty_cache()
            _reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
            _total_mb = torch.cuda.get_device_properties(_gpu_idx).total_memory / 1024**2
            if _reserved_mb > _total_mb - (VRAM_RESERVE_GB * 1024):
                torch.cuda.empty_cache()
                print(f'[Batch] Batch size {bs} — spilling to shared VRAM, stepping down')
                return False
        elif HAS_PSUTIL:
            _free_ram = psutil.virtual_memory().available / 1024**3
            if _free_ram < RAM_RESERVE_GB:
                print(f'[Batch] Batch size {bs} — RAM dropped below {RAM_RESERVE_GB}GB reserve after probe, stepping down')
                return False
            _swap_now = round(psutil.swap_memory().used / 1024**3, 3)
            if _swap_now > _candidate_swap_baseline_gb:
                print(f'[Batch] Batch size {bs} — page file usage grew beyond this '
                      f'candidate\'s own {_candidate_swap_baseline_gb:.3f} GB baseline '
                      f'({_swap_now:.3f} GB now) after probe, stepping down')
                return False
        print(f"[Batch] Batch size {bs} — OK ({probe_steps} real steps completed)")
        return True
    except RuntimeError as e:
        _emsg = str(e).lower()
        if "out of memory" in _emsg or "find was unable" in _emsg or "engine" in _emsg:
            if _on_cuda:
                torch.cuda.empty_cache()
            print(f"[Batch] Batch size {bs} — failed ({type(e).__name__}), trying next")
            return False
        else:
            raise


def determine_batch_size(build_model, build_optimizer, criterion, img_size: int,
                          num_classes: int, device: torch.device,
                          candidates=(64, 128, 256, 512, 1024, 2048, 3584, 5120, 6656, 8192,
                                      9728, 11264, 12800, 14336, 15872, 17408, 18944, 20480,
                                      22016, 23552, 25088, 26624, 28160, 29696, 31232, 32768,
                                      34304, 35840, 37376, 38912, 40448, 41984, 43520, 45056,
                                      46592, 48128, 49664, 51200, 52736, 54272, 55808, 57344,
                                      58880, 60416, 61952, 63488, 65024, 66560, 68096, 69632,
                                      71168, 72704, 74240, 75776, 77312, 78848, 80384, 81920,
                                      83456, 84992, 86528, 88064, 89600, 91136, 92672, 94208,
                                      95744, 97280, 98816, 100352, 101888, 103424, 104960,
                                      106496, 108032, 109568, 111104, 112640, 114176, 115712,
                                      117248, 118784, 120320, 121856, 123392, 124928, 126464,
                                      128000, 129536, 131072, 132608, 134144, 135680, 137216,
                                      138752, 140288, 141824, 143360, 144896, 146432, 147968,
                                      149504, 151040, 152576, 154112, 155648, 157184, 158720,
                                      160256, 161792, 163328, 164864, 166400, 167936, 169472,
                                      171008, 172544, 174080, 175616, 177152, 178688, 180224,
                                      181760, 183296, 184832, 186368, 187904, 189440, 190976,
                                      192512, 194048, 195584, 197120, 198656, 200192, 201728,
                                      203264, 204800, 206336, 207872, 209408, 210944, 212480,
                                      214016, 215552, 217088, 218624, 220160, 221696, 223232,
                                      224768, 226304, 227840, 229376, 230912, 232448, 233984,
                                      235520, 237056, 238592, 240128, 241664, 243200, 244736,
                                      246272, 247808, 249344, 250880, 252416, 253952, 255488,
                                      257024, 258560, 260096, 261632, 263168, 264704, 266240,
                                      267776, 269312, 270848, 272384, 273920, 275456, 276992,
                                      278528, 280064, 281600, 283136, 284672, 286208, 287744,
                                      289280, 290816, 292352, 293888, 295424, 296960, 298496,
                                      300032, 301568, 303104, 304640, 306176, 307200),
                          probe_steps: int = 5, use_amp: bool = True,
                          grad_clip_norm=1.0) -> int:
    """
    Bottom-up coarse pass + binary-search refinement. See module
    docstring for the full rationale.

    DDP note (see common/distributed.py, including its "NOT YET
    VALIDATED against real multi-GPU hardware" caveat, which applies
    here too): on a distributed run, EVERY rank runs the full probe
    independently against its own device, and the final result is the
    MINIMUM across all ranks (via all_reduce_min()) — a shared/contended
    compute node can leave one rank's GPU with meaningfully less free
    VRAM than another's even on identical hardware, so trusting one
    rank's probe alone would be unsafe. Costs every rank its own probe's
    time and VRAM churn.
    """
    print(f"[Batch] Auto-detecting batch size for {img_size}x{img_size} "
          f"({probe_steps} real training steps per candidate, bottom-up "
          f"with binary-search refinement)...")

    last_working  = None
    first_failing = None
    for bs in candidates:
        if _probe_batch_size(bs, build_model, build_optimizer, criterion, img_size,
                              num_classes, device, probe_steps, use_amp, grad_clip_norm):
            last_working = bs
        else:
            first_failing = bs
            break

    if last_working is None:
        print(f"[Batch] Even the smallest candidate ({candidates[0]}) failed — "
              f"using it anyway, training may still OOM")
        return all_reduce_min(candidates[0], device)

    if first_failing is None:
        print(f"[Batch] All candidates up to {last_working} fit — "
              f"using the largest, no refinement needed")
        return all_reduce_min(last_working, device)

    lo, hi = last_working, first_failing
    print(f"[Batch] Bracket found: {lo} works, {hi} fails — refining...")
    while hi - lo > 8:
        mid = round(((lo + hi) // 2) / 8) * 8
        if mid <= lo or mid >= hi:
            break
        if _probe_batch_size(mid, build_model, build_optimizer, criterion, img_size,
                              num_classes, device, probe_steps, use_amp, grad_clip_norm):
            lo = mid
        else:
            hi = mid

    print(f"[Batch] Refined to batch size {lo} (largest confirmed-working, "
          f"within 8 of the true VRAM ceiling)")
    return all_reduce_min(lo, device)


def cap_batch_size_for_min_steps(batch_size: int, dataset_size: int, min_steps: int,
                                  world_size: int = 1) -> int:
    """
    Caps an already-determined batch size down so a training epoch gets
    at least min_steps real gradient-update steps, given the real
    training-set size.

    Why this can't live inside determine_batch_size() above: the VRAM/RAM
    probe runs against dummy random tensors, before the real dataset is
    loaded (every caller determines batch size first, then loads its
    dataset — see each script's run_training()) — it has no way to know
    the real dataset size at the point it picks a batch size. This
    function is meant to be called by the caller once the real training
    set is loaded, right before building the train DataLoader, so a tiny
    dataset (or a model small enough that the VRAM probe lands at the top
    of its candidate ladder — e.g. the router at 16x16, ~62.6K params)
    can't end up with too few batches per epoch to train on meaningfully.

    Only ever lowers batch_size, never raises it — if the VRAM-probed
    size already yields >= min_steps for this dataset, it's returned
    unchanged.

    world_size (DDP, see common/distributed.py): under distributed
    training, `batch_size` here is the PER-RANK batch size, but every
    rank steps together in lockstep each iteration — the real number of
    optimizer.step() calls per epoch is dataset_size / (batch_size *
    world_size), not dataset_size / batch_size. Defaults to 1 (unchanged
    formula) for every non-distributed call site.
    """
    max_batch_for_min_steps = max(1, dataset_size // (min_steps * world_size))
    return min(batch_size, max_batch_for_min_steps)


def reduce_batch_size(current_batch_size: int, dataset_size: int, min_steps: int,
                       backoff_pct: float = 0.10, world_size: int = 1,
                       device: torch.device = None) -> int:
    """
    Steps a batch size down by backoff_pct (rounded to a multiple of 8)
    for reactive mid-run adjustment after a completed epoch's peak VRAM
    crosses into check_vram_safety()'s warn_reserve_gb buffer — floored
    by the same min-steps-per-epoch guarantee cap_batch_size_for_min_steps()
    already enforces at initial auto-detect time, so repeated reductions
    can't collapse batch size below what the dataset needs for a
    meaningful epoch.

    device: if given and CUDA, calls torch.cuda.empty_cache() before
    returning. nvidia-smi's memory.used reports the caching allocator's
    reserved pool, not just live tensors, and "the unused memory managed
    by the allocator will still show as if used in nvidia-smi" until
    empty_cache() releases it (per PyTorch's own CUDA memory-management
    docs) — without this call, a reduced batch size never actually
    lowers the nvidia-smi reading check_vram_safety() watches, only the
    batch's live tensor footprint. Optional/defaulted to None so every
    pre-existing call site keeps working unchanged until it's updated to
    pass its own device.
    """
    reduced = max(8, round((current_batch_size * (1 - backoff_pct)) / 8) * 8)
    result = cap_batch_size_for_min_steps(reduced, dataset_size, min_steps, world_size)
    if device is not None and device.type == "cuda":
        torch.cuda.empty_cache()
    return result
