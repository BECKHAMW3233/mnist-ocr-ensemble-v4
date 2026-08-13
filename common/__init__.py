"""
common/
=======
Shared modules imported by every v3 training script (the 15 digit models
and the 5 router models). Pulled out of the v2 project's per-script
duplication (auto batch-sizing, checkpoint/resume, telemetry, seeding,
CLI logging, ONNX export) into one place each, per the v3 modularization
pass — see v3_CHANGELOG.md for the reasoning and file-by-file diff summary.

This is a structural refactor only: every function here is a direct
extraction of logic that was previously copy-pasted near-verbatim across
12-16 v2 scripts, with only the truly per-script bits (model/optimizer
construction, resolution, output paths) left as caller-supplied
parameters. No behavior changes were made while extracting.

The PYTORCH_ALLOC_CONF setting below is a later addition, not part of
that original extraction — it IS a real behavior change (a CUDA
allocator hotfix), see v3_CHANGELOG.md.
"""
import os

# Must be set before `import torch` first touches CUDA — every script's
# very first import is `from common.seeding import ...`, which runs this
# package's __init__.py first, ahead of `import torch`. setdefault(), not
# a hard set, so an explicit value already set in the caller's own shell
# isn't silently overridden.
# Note: expandable_segments:True is currently a confirmed no-op on this
# Windows PyTorch build (see v3_CHANGELOG.md) — kept for portability/
# future platforms; garbage_collection_threshold:0.8 is the part
# actually active here.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.8")
