"""
common/seeding.py
==================
Deterministic seeding, shared across every v3 training script. Extracted
verbatim from the set_all_seeds() block that was duplicated identically
across all 16 v2 training scripts (12 optimizer scripts + the v2 digit
gate) — same GLOBAL_SEED/MNIST_SEED environment-variable convention, same
CUBLAS_WORKSPACE_CONFIG requirement, same cudnn.deterministic/benchmark
settings. Behavior is unchanged from v2; only the code's location moved.

IMPORTANT — import order, preserved from every v2 script:
    apply_cublas_workspace_config()   # before `import torch`
    import torch
    reserve_cpu_threads()             # immediately after `import torch`,
                                       # before any other torch call
    set_all_seeds(get_global_seed())  # before importing torch.nn / torchvision
Getting this order wrong silently reintroduces nondeterminism (cuBLAS) or
a CPU thread count that doesn't reliably take effect — both documented,
confirmed issues in the v2 codebase this ordering fixes.
"""
import os


def apply_cublas_workspace_config() -> None:
    """Must be called before `import torch`. Required for deterministic
    torch.mm/mv/bmm (used internally by every nn.Linear layer in every
    architecture in this project) on CUDA 10.2+."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def get_global_seed() -> int:
    """Reads MNIST_SEED (default 42, matching the v2 split-seed convention).
    Read via env var rather than a --seed CLI flag because set_all_seeds()
    must run at import time, before argparse ever gets a chance to parse."""
    return int(os.environ.get("MNIST_SEED", 42))


def set_all_seeds(seed: int) -> None:
    """
    Seeds Python's random, NumPy, PyTorch CPU/CUDA RNG, and forces
    deterministic cuDNN (trading some speed for reproducibility). Must be
    called after `import torch` (and after apply_cublas_workspace_config(),
    which must run before that import).

    Only prints in the main process: on Windows, persistent_workers=True
    with num_workers>0 uses multiprocessing's 'spawn' start method, which
    re-imports the calling script fresh in every worker — re-running this
    call. The reseeding itself still needs to happen in every worker
    (that's what gives each worker correctly-seeded augmentation RNG);
    only the console print is redundant past the first call.
    """
    import random as _py_random
    import numpy as np
    import torch

    _py_random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # trades some speed for determinism

    import multiprocessing as _mp
    if _mp.current_process().name == "MainProcess":
        print(f"[Seed] Using GLOBAL_SEED={seed} for weight init, dropout, "
              f"augmentation RNG, and CUDA RNG (override with the MNIST_SEED "
              f"environment variable)")


def usable_cpu_count(cpu_reserve_pct: int = 25) -> int:
    """
    Returns the number of logical cores usable after reserving
    cpu_reserve_pct% for the OS — the same reservation math
    reserve_cpu_threads() uses for torch.set_num_threads(), exposed
    standalone here (2026-07-28, per direct user follow-up) so DataLoader
    worker counts (NUM_WORKERS in every training script) can auto-scale
    to this machine's real core count instead of a hardcoded literal.

    Deliberately NOT gated on CUDA availability the way
    reserve_cpu_threads() is below — torch.set_num_threads() only matters
    for CPU-bound tensor ops, which are irrelevant once CUDA is doing the
    compute, but DataLoader worker processes (which feed the GPU via
    prefetching) compete for CPU regardless of which device model compute
    runs on.
    """
    total_cores = os.cpu_count() or 1
    reserved = max(1, round(total_cores * cpu_reserve_pct / 100))
    return max(1, total_cores - reserved)


def reserve_cpu_threads(cpu_reserve_pct: int = 25) -> None:
    """
    Reserves cpu_reserve_pct% of logical cores for the OS on CPU-only
    runs. Must be called immediately after `import torch`, before any
    other PyTorch eager/JIT/autograd call — torch.set_num_threads() only
    reliably takes effect that early, per PyTorch's own docs. No-op when
    CUDA is available (torch.cuda.is_available() itself is safe to call
    this early).
    """
    import torch
    if not torch.cuda.is_available():
        torch.set_num_threads(usable_cpu_count(cpu_reserve_pct))
