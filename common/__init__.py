"""
common/
=======
Shared modules imported by every training script (the 15 digit models,
the 24 uppercase/lowercase letter models, and the 5 router models):
auto batch-sizing, checkpoint/resume, telemetry, seeding, CLI logging,
ONNX export.

Every function here is a direct extraction of logic previously
duplicated across every training script, with only the truly per-script
bits (model/optimizer construction, resolution, output paths) left as
caller-supplied parameters.
"""
import os

# Must be set before `import torch` first touches CUDA — every script's
# very first import is `from common.seeding import ...`, which runs this
# package's __init__.py first, ahead of `import torch`. setdefault(), not
# a hard set, so an explicit value already set in the caller's own shell
# isn't silently overridden. expandable_segments:True is currently a
# confirmed no-op on this Windows PyTorch build, kept for portability to
# other platforms; garbage_collection_threshold:0.8 is the part actually
# active here.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.8")
