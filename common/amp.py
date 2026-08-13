"""
common/amp.py
==============
Mixed-precision training step: autocast forward/loss + GradScaler-managed
backward/step/clip. Extracted from the identical block duplicated across
every v2 script that used AMP (all except SOAP, which requires float32
throughout for its Kronecker eigendecomposition and never enables AMP —
that script correctly does not import this module) and every v3 script
in the same position (AdamW, Muon, the router). Also used by
common/batch_sizing.py's probe, so the exact same mechanics that decide
a batch size are the ones actually used during real training — not a
close copy that could quietly drift from it.
"""
import torch
import torch.nn as nn


def build_grad_scaler(device: torch.device, use_amp: bool = True) -> torch.amp.GradScaler:
    return torch.amp.GradScaler('cuda', enabled=use_amp and device.type == "cuda")


def amp_train_step(model, images, labels, criterion, optimizer, scaler,
                    device: torch.device, use_amp: bool = True, grad_clip_norm=1.0):
    """
    Runs one forward + backward + optimizer step under AMP. Returns
    (loss, logits, stepped) — `stepped` is False on a batch where
    GradScaler skipped the underlying optimizer.step() (an inf/nan
    gradient, most common while the loss scale is still calibrating in
    the first several batches). Callers with a per-step LR scheduler
    should only call scheduler.step() when stepped is True — advancing
    the schedule on a skipped step would run it ahead of real training
    progress. Callers without a scheduler (Schedule-Free AdamW) can
    ignore the third return value.
    """
    with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                        enabled=use_amp and device.type == "cuda"):
        logits = model(images)
        loss = criterion(logits, labels)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    if grad_clip_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    stepped = scaler.get_scale() >= scale_before

    return loss, logits, stepped
