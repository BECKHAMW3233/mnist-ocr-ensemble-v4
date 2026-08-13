"""
common/optimizers.py
=====================
Optimizer construction, centralized (2026-08-08, per direct user follow-up,
knowingly overriding the 2026-07-27 decision documented in
v3_CHANGELOG.md's "Naming and modularization-scope correction" — that
entry reverted an earlier attempt at this exact centralization, reasoning
that optimizer algorithms were never part of Part 2's modularization
scope. That history was shown directly and William chose to proceed
anyway) from build_soap()/build_adamw_sf()/build_muon()/
build_ranger_optimizer() and their supporting classes (Muon, RAdam,
Lookahead) and hyperparameter constants — verified byte-for-byte
identical (module-level constants and function/class bodies) across all
44 training scripts; none of it varies by resolution or model category.

Each build_*() function keeps its original default hyperparameter values.
Scripts that need a non-default value can still override via keyword
arguments — none currently do.
"""
import math
import sys

import torch
from torch.optim import Optimizer


# =============================================================================
# SOAP
# =============================================================================

SOAP_LR             = 1e-3
SOAP_BETAS          = (0.95, 0.95)
SOAP_WEIGHT_DECAY   = 5e-4
SOAP_PRECOND_FREQ   = 100


def build_soap(params, lr: float = SOAP_LR, betas=SOAP_BETAS,
                weight_decay: float = SOAP_WEIGHT_DECAY,
                precondition_frequency: int = SOAP_PRECOND_FREQ):
    try:
        from pytorch_optimizer import SOAP
    except ImportError:
        print("ERROR: pytorch_optimizer not installed. Run: pip install pytorch_optimizer")
        sys.exit(1)
    return SOAP(params, lr=lr, betas=betas, weight_decay=weight_decay,
                precondition_frequency=precondition_frequency)


# =============================================================================
# AdamW (Schedule-Free)
# Requires: pip install schedulefree. Must call optimizer.train() before
# each training epoch and optimizer.eval() before validation/evaluation —
# each calling script's own training loop is responsible for that, this
# module only constructs the optimizer.
# =============================================================================

ADAMW_SF_LR            = 1e-3
ADAMW_SF_WEIGHT_DECAY  = 1e-4


def build_adamw_sf(params, warmup_steps: int, lr: float = ADAMW_SF_LR,
                    weight_decay: float = ADAMW_SF_WEIGHT_DECAY):
    try:
        import schedulefree
    except ImportError:
        print("ERROR: schedulefree not installed. This script requires Schedule-Free "
              "AdamW — run: pip install schedulefree")
        sys.exit(1)
    return schedulefree.AdamWScheduleFree(
        params, lr=lr, weight_decay=weight_decay, warmup_steps=warmup_steps,
    )


# =============================================================================
# Muon (+ AdamW fallback group) — inlined implementation, not from an
# external package.
# =============================================================================

MUON_LR          = 0.02
MUON_MOMENTUM    = 0.95
MUON_WD          = 0.01
NS_STEPS         = 5
MUON_ADAMW_LR    = 3e-4
MUON_ADAMW_BETAS = (0.9, 0.95)
MUON_ADAMW_EPS   = 1e-8
MUON_ADAMW_WD    = 0.01


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Quintic Newton-Schulz iteration approximating G's orthogonalized
    "UV^T" term without computing an actual SVD. The coefficients
    (3.4445, -4.7750, 2.0315) are the published quintic coefficients
    (Jordan et al.) chosen so the iteration converges in ~5 steps from a
    spectrally-normalized start. Runs in float32 throughout (not
    bfloat16, unlike some reference implementations) so behavior is
    identical whether this project falls back to CPU or not.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(Optimizer):
    """
    One Optimizer over two kinds of param_groups, distinguished by the
    `use_muon` key each group dict must carry:

      use_muon=True  — Muon update: SGD-style Nesterov momentum, the
        momentum-combined gradient orthogonalized via Newton-Schulz
        before being applied, scaled by sqrt(max(1, rows/cols)) to keep
        the update's RMS roughly shape-independent (matches the
        published recipe). A 4D Conv2d weight (out_ch, in_ch, kh, kw) is
        reshaped to 2D (out_ch, in_ch*kh*kw) for orthogonalization and
        reshaped back before the update is applied.
      use_muon=False — standard decoupled-weight-decay AdamW update, for
        the parameters Muon isn't designed for.

    Both branches maintain per-parameter state via the base Optimizer's
    own `self.state` dict, so state_dict()/load_state_dict() (needed for
    this project's checkpoint/resume) work with no extra code.
    """
    def __init__(self, param_groups):
        for g in param_groups:
            g.setdefault("lr", 0.02 if g.get("use_muon") else 3e-4)
            g.setdefault("momentum", 0.95)
            g.setdefault("nesterov", True)
            g.setdefault("ns_steps", 5)
            g.setdefault("weight_decay", 0.0)
            g.setdefault("betas", (0.9, 0.95))
            g.setdefault("eps", 1e-8)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon(group)
            else:
                self._step_adamw(group)
        return loss

    def _step_muon(self, group):
        momentum = group["momentum"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(g)
            update = g.add(buf, alpha=momentum) if group["nesterov"] else buf

            orig_shape = update.shape
            mat = update.reshape(orig_shape[0], -1) if update.ndim > 2 else update
            rows, cols = mat.shape
            ortho = zeropower_via_newtonschulz5(mat, steps=group["ns_steps"])
            ortho = ortho.reshape(orig_shape)

            scale = max(1.0, rows / cols) ** 0.5
            if group["weight_decay"] != 0.0:
                p.mul_(1 - group["lr"] * group["weight_decay"])
            p.add_(ortho, alpha=-group["lr"] * scale)

    def _step_adamw(self, group):
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
            bias_c1 = 1 - beta1 ** state["step"]
            bias_c2 = 1 - beta2 ** state["step"]
            denom = (exp_avg_sq.sqrt() / (bias_c2 ** 0.5)).add_(group["eps"])
            step_size = group["lr"] / bias_c1
            if group["weight_decay"] != 0.0:
                p.mul_(1 - group["lr"] * group["weight_decay"])
            p.addcdiv_(exp_avg, denom, value=-step_size)


def build_muon(params) -> Muon:
    """
    Splits a flat parameter iterable (not a model — this is the contract
    common/batch_sizing.py's determine_batch_size() calls build_optimizer
    with) into the Muon group (ndim >= 2) and the AdamW-fallback group
    (ndim < 2). Used both by the batch-size probe (via this function) and
    the real, non-probe optimizer construction in each script's
    run_training().
    """
    params = list(params)
    muon_params  = [p for p in params if p.requires_grad and p.ndim >= 2]
    other_params = [p for p in params if p.requires_grad and p.ndim < 2]
    param_groups = [
        dict(params=muon_params, use_muon=True, lr=MUON_LR, momentum=MUON_MOMENTUM,
             nesterov=True, ns_steps=NS_STEPS, weight_decay=MUON_WD),
        dict(params=other_params, use_muon=False, lr=MUON_ADAMW_LR, betas=MUON_ADAMW_BETAS,
             eps=MUON_ADAMW_EPS, weight_decay=MUON_ADAMW_WD),
    ]
    return Muon(param_groups)


# =============================================================================
# Ranger (RAdam + Lookahead) — inlined implementation, not from an external
# package. Ranger = RAdam (Liu et al., 2019, "On the Variance of the
# Adaptive Learning Rate and Beyond") + Lookahead (Zhang, Lucas, Hinton &
# Ba, 2019, "Lookahead Optimizer: k steps forward, 1 step back"), combined
# per Wright (2019)'s "Ranger" recipe.
# =============================================================================

class RAdam(Optimizer):
    """Rectified Adam. Decoupled weight decay (applied multiplicatively
    before the adaptive update, AdamW-style)."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                if wd != 0.0:
                    p.mul_(1 - lr * wd)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_c1 = 1 - beta1 ** t
                m_hat = exp_avg / bias_c1

                rho_inf = 2.0 / (1.0 - beta2) - 1.0
                beta2_t = beta2 ** t
                rho_t = rho_inf - 2.0 * t * beta2_t / (1.0 - beta2_t)

                if rho_t > 4.0:
                    bias_c2 = 1 - beta2_t
                    v_hat = (exp_avg_sq / bias_c2).sqrt()
                    r_t = math.sqrt(
                        ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf) /
                        ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
                    )
                    p.addcdiv_(m_hat, v_hat.add_(eps), value=-lr * r_t)
                else:
                    p.add_(m_hat, alpha=-lr)
        return loss


class Lookahead(Optimizer):
    """
    Wraps a base optimizer (RAdam here). Shares its `state` dict with the
    base optimizer so the "slow" weight buffer this class adds lives
    alongside RAdam's own per-parameter state — PyTorch's default
    Optimizer.state_dict() serializes by ordinal position within
    param_groups, not by object identity, so this one shared dict is all
    that's needed for both optimizers' state to round-trip correctly
    through this project's resume/checkpoint mechanism.
    """
    def __init__(self, base_optimizer: Optimizer, k: int = 6, alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if k < 1:
            raise ValueError(f"Invalid k: {k}")
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = base_optimizer.param_groups
        self.state = base_optimizer.state
        self.defaults = base_optimizer.defaults
        for group in self.param_groups:
            group.setdefault("la_step_counter", 0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group["la_step_counter"] += 1
            if group["la_step_counter"] % self.k == 0:
                for fast_p in group["params"]:
                    if fast_p.grad is None:
                        continue
                    state = self.state[fast_p]
                    if "slow_buffer" not in state:
                        state["slow_buffer"] = torch.clone(fast_p.data).detach()
                    slow = state["slow_buffer"]
                    slow.add_(fast_p.data - slow, alpha=self.alpha)
                    fast_p.data.copy_(slow)
        return loss

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group):
        self.base_optimizer.add_param_group(param_group)
        self.param_groups = self.base_optimizer.param_groups


def build_ranger_optimizer(params, lr: float = 1e-3, betas=(0.9, 0.999),
                            eps: float = 1e-8, weight_decay: float = 0.0,
                            k: int = 6, alpha: float = 0.5) -> Lookahead:
    """Builds a Lookahead-wrapped RAdam instance — this project's Ranger."""
    base = RAdam(list(params), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return Lookahead(base, k=k, alpha=alpha)
