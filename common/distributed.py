"""
common/distributed.py
======================
Optional multi-GPU DistributedDataParallel (DDP) support (2026-07-28, per
direct user follow-up) — splits ONE model's training across multiple
GPUs on the same machine, launched via `torchrun`. This is the second,
much larger multi-GPU mode; see setup_device()'s gpu_id parameter
(common/telemetry.py) for the simpler "run different independent models
on different cards at once" mode, which needs none of this module.

*** NOT YET VALIDATED AGAINST REAL MULTI-GPU HARDWARE ***
Every real run this whole v3 codebase has ever been checked against —
every batch size, every VRAM number, every telemetry field, every real
accuracy figure cited anywhere in v3_CHANGELOG.md — came from a single
RTX 4080. This module was written from PyTorch's own documented
`torch.distributed`/DDP API and standard, widely-used patterns for
combining weighted sampling with distributed sharding, reasoned through
carefully, but has NEVER been run against actual multi-GPU hardware,
because none is available in the environment this was built in. DDP bugs
are specifically the class of bug that "imports cleanly" and "reads
correctly" cannot catch — mismatched collective calls across ranks
deadlock instead of raising, and a checkpoint race from an unguarded
write is silent until the file is corrupted. Treat every training script
run under `torchrun` as unverified until confirmed on real hardware; the
single-GPU / --gpu-flag path is completely unaffected by anything in
this file and remains exactly as validated as before.

How it's triggered: `torchrun` (PyTorch's own launcher) sets RANK,
WORLD_SIZE, and LOCAL_RANK environment variables automatically, one
process per GPU — nothing in this module sets those itself. Every
function below is a no-op / passthrough when those variables aren't
present (i.e. every existing single-process invocation, with or without
--gpu, is completely unaffected).

Usage (one process per GPU, launched by torchrun, not by hand):
    torchrun --standalone --nproc_per_node=2 v3_mnist_digit_soap_64.py
Do NOT also pass --gpu under torchrun — LOCAL_RANK (set by the launcher)
already picks each process's device; see setup_device()'s docstring.
"""
import os

import torch


def is_distributed() -> bool:
    """True only when torchrun (or an equivalent launcher) has set
    WORLD_SIZE > 1 — i.e. this process is one of several cooperating
    ranks. False for every plain `python script.py` invocation, with or
    without --gpu."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    """The GPU index THIS process should use — set by torchrun per
    process on the local machine. Distinct from get_rank() on a
    multi-machine job (not a configuration this project's scripts are
    set up for — single-machine, --nproc_per_node only); identical to it
    for the single-machine case these scripts actually support."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    """True on rank 0, and (by construction) always True when not
    running distributed at all — every existing single-process script
    invocation is rank 0 of a world size of 1."""
    return get_rank() == 0


def setup_distributed(device: torch.device) -> None:
    """
    Initializes the default process group. Must be called exactly once,
    early — after setup_device() has resolved the per-rank device (so
    the correct GPU is already selected via LOCAL_RANK before this
    touches anything collective), before the model is constructed or any
    DataLoader with a distributed-aware sampler is built. No-op if
    is_distributed() is False.

    Backend: 'nccl' for CUDA (the standard, GPU-direct, documented choice
    for multi-GPU training — NOT validated here, no multi-GPU hardware
    available to confirm NCCL is correctly installed/working in this
    environment). 'gloo' for CPU (NCCL doesn't support CPU tensors) —
    included for completeness; CPU-only multi-process training is not a
    scenario this project has any real use case for, but the branch
    costs nothing to have correct.
    """
    if not is_distributed():
        return
    import torch.distributed as dist
    backend = "nccl" if device.type == "cuda" else "gloo"
    dist.init_process_group(backend=backend)


def cleanup_distributed() -> None:
    """Destroys the process group if one was initialized. Safe to call
    even when is_distributed() is False (no-op)."""
    if not is_distributed():
        return
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """
    Wraps model in DistributedDataParallel when running distributed,
    returns model UNCHANGED otherwise — every non-distributed call site
    in this codebase keeps working exactly as before, no conditional
    needed at the call site itself beyond calling this function.

    IMPORTANT — checkpoint compatibility: a DDP-wrapped model's
    state_dict() keys are prefixed with 'module.' (DDP stores the real
    model as a .module attribute and proxies everything else through to
    it). Every place this codebase saves a checkpoint or reconstructs a
    model for ONNX export must call unwrap_model(model).state_dict() /
    unwrap_model(model), NOT model.state_dict() directly, or a checkpoint
    saved from a distributed run silently becomes incompatible with the
    normal single-GPU load path (a real, well-documented DDP gotcha, not
    a hypothetical one) — see unwrap_model() below.
    """
    if not is_distributed():
        return model
    device_ids = [device.index] if device.type == "cuda" else None
    return torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Returns the real underlying model, whether or not it's currently
    DDP-wrapped — see wrap_model_ddp()'s docstring for why every
    checkpoint save / ONNX-export model reconstruction must go through
    this rather than assuming model.state_dict() is always the raw,
    unprefixed state dict."""
    return model.module if hasattr(model, "module") else model


def all_reduce_sum(value: float, device: torch.device) -> float:
    """
    Sums `value` across every rank and returns the same total to all of
    them — used to aggregate each rank's LOCAL total_loss/total_correct/
    total_samples accumulators (train_one_epoch()/evaluate() in every
    training script) into the correct GLOBAL epoch metric before
    computing loss/accuracy ratios. Without this, a distributed run's
    printed/logged per-epoch loss and accuracy would silently only
    reflect whatever slice of the data that ONE rank happened to see,
    not the real epoch-wide numbers a single-GPU run would report —
    quietly wrong, not crashed, exactly the kind of DDP bug that doesn't
    announce itself. No-op passthrough (returns value unchanged) when
    not running distributed.
    """
    if not is_distributed():
        return value
    import torch.distributed as dist
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def all_reduce_min(value, device: torch.device):
    """
    Takes the minimum of `value` across every rank and returns the same
    minimum to all of them — used for the auto-detected batch size
    (2026-08-02, per direct user follow-up): rather than trusting rank
    0's probe alone (which silently assumed every rank's GPU had
    equivalent free VRAM — see broadcast_int()'s own docstring for the
    original reasoning, and why that assumption doesn't hold on a
    shared/contended compute node, where another tenant's job can leave
    one rank's GPU with meaningfully less free VRAM than another's even
    on identical hardware), every rank now probes its own device
    independently (see determine_batch_size() in common/batch_sizing.py)
    and this takes the minimum across all of them, so every rank trains
    at a batch size that's safe for the rank with the LEAST available
    VRAM, not just whatever rank 0 happened to have free. No-op
    passthrough (returns value unchanged) when not running distributed.
    """
    if not is_distributed():
        return value
    import torch.distributed as dist
    tensor = torch.tensor(value, dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return int(tensor.item())


def broadcast_int(value: int, device: torch.device, src: int = 0) -> int:
    """
    Broadcasts an integer FROM rank `src` (default rank 0) to every
    other rank, returning the same value everywhere. Used specifically
    for the auto-detected batch size (2026-07-28): rather than assuming
    every rank's VRAM probe would independently land on the identical
    batch size (true only if every GPU in the job is genuinely
    identical, which this codebase has no way to verify), the probe now
    runs ONLY on rank 0 and every other rank receives that exact same
    number via this call — guarantees every rank trains with the same
    batch size even on a mixed-GPU job, and halves the batch-size-probe
    VRAM churn on every rank but one. No-op passthrough (returns value
    unchanged) when not running distributed.
    """
    if not is_distributed():
        return value
    import torch.distributed as dist
    tensor = torch.tensor(value, dtype=torch.int64, device=device)
    dist.broadcast(tensor, src=src)
    return int(tensor.item())


class DistributedWeightedRandomSampler(torch.utils.data.Sampler):
    """
    Combines weighted (with-replacement) class-balancing sampling — what
    every training script's WeightedRandomSampler already does — with
    DDP's requirement that each rank see a DIFFERENT shard of the epoch's
    samples, not the same ones redundantly. Vanilla PyTorch has no
    built-in that does both at once (DistributedSampler shards a
    Dataset's indices directly with uniform probability; it has no
    concept of per-sample weights).

    Standard, widely-used pattern (not a novel invention): every rank
    deterministically generates the SAME full weighted-with-replacement
    index list for the epoch (torch.multinomial seeded identically via
    seed + epoch on every rank — no inter-process communication needed
    for this part, since the same seed always produces the same
    sequence), then each rank takes every world_size-th index starting
    at its own rank offset (`indices[rank::world_size]`). This guarantees
    every rank does disjoint work (no duplicated samples across ranks in
    the same epoch) while preserving the original per-sample class-
    balancing weights in aggregate across the whole distributed batch.

    set_epoch(epoch) MUST be called by the training loop before each
    epoch's DataLoader iteration begins (the standard DDP sampler
    contract, same as torch's own DistributedSampler) — without it,
    every epoch draws the identical weighted sample, defeating the
    point of re-sampling each epoch. NOT unit-testable against real
    multi-rank behavior in this sandbox (no second process/GPU to
    confirm disjoint sharding against); the single-rank case
    (world_size=1, rank=0) degenerates to exactly a plain
    WeightedRandomSampler and was verified against that directly.
    """

    def __init__(self, weights, num_samples: int, rank: int = None,
                 world_size: int = None, seed: int = 0):
        self.weights     = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.rank        = rank if rank is not None else get_rank()
        self.world_size  = world_size if world_size is not None else get_world_size()
        self.seed        = seed
        self.epoch       = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.num_samples, replacement=True, generator=g)
        return iter(indices[self.rank::self.world_size].tolist())

    def __len__(self) -> int:
        return (self.num_samples - self.rank + self.world_size - 1) // self.world_size
