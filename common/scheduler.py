"""
common/scheduler.py
====================
Linear-warmup -> cosine-decay LR scheduler, stepped once per optimizer
step (not once per epoch). Extracted from the WarmupCosineScheduler class
duplicated across every v2 script that used per-step scheduling
(mnist_soap_*.py, mnist_adahessian_*.py, mnist_digit_gate.py) — same
shape-target-not-a-cap semantics (PATIENCE-based early stopping is what
actually ends training; total_steps only shapes the cosine curve).

Works unchanged over any number of param_groups: each group's own base
LR (captured at construction) is scaled by the same warmup/decay curve,
so a caller with multiple param_groups at different base LRs — e.g.
Muon's own group plus its AdamW-fallback group, each at a different LR —
gets consistent relative scaling automatically with no extra code.
"""
import math


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, eta_min: float = 1e-6):
        self.optimizer    = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps  = total_steps
        # eta_min here is a FRACTION of each group's own base LR, not an
        # absolute LR floor like PyTorch's CosineAnnealingLR(eta_min=...) —
        # the real floor is base_lr * eta_min (e.g. eta_min=1e-6 with
        # base_lr=1e-3 bottoms out at 1e-9, not 1e-6).
        self.eta_min      = eta_min
        self.base_lrs     = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count   = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            scale = self.step_count / self.warmup_steps
        else:
            progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            scale = self.eta_min + 0.5 * (1 - self.eta_min) * (1 + math.cos(math.pi * progress))
        for pg, base in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base * scale

    def rescale_total_steps(self, steps_per_epoch_old: int, steps_per_epoch_new: int) -> None:
        """
        Proportionally rescales the remaining step-count horizon when the
        caller's batch size changes mid-run (2026-08-08, per direct user
        follow-up — reactive batch-size backoff after a VRAM-reserve
        crossing): preserves how many real EPOCHS are left until the
        cosine curve bottoms out, re-expressed in the new batch size's
        steps-per-epoch rate, rather than leaving total_steps fixed at
        the step-count the OLD batch size would have taken to get there.
        """
        remaining_steps = self.total_steps - self.step_count
        remaining_epochs_equiv = remaining_steps / steps_per_epoch_old
        self.total_steps = self.step_count + round(remaining_epochs_equiv * steps_per_epoch_new)

    def get_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def get_last_lr(self):
        return self.get_lr()

    def state_dict(self):
        return {
            "step_count":   self.step_count,
            "base_lrs":     self.base_lrs,
            "warmup_steps": self.warmup_steps,
            "total_steps":  self.total_steps,
            "eta_min":      self.eta_min,
        }

    def load_state_dict(self, state):
        self.step_count   = state["step_count"]
        self.base_lrs     = state["base_lrs"]
        self.warmup_steps = state["warmup_steps"]
        self.total_steps  = state["total_steps"]
        self.eta_min      = state["eta_min"]
