# EMNIST Ensemble v4

v4 of a multi-optimizer PyTorch OCR ensemble for handwritten digit and
letter recognition. Optimizers: SOAP, AdamW, Muon. Includes a
case-classifying router (digit/uppercase/lowercase/unknown) ahead of
planned letter-reading models. No post-processing — raw output only.

This is v4 of the project, renamed from v3 — see `v4_CHANGELOG.md` for
the full rename history and rationale.

## Current structure

    emnist_ensemble_v4/
    |-- common/                  # Shared modules (batch-sizing, checkpointing, telemetry, seeding, etc.)
    |   |-- __init__.py
    |   |-- amp.py
    |   |-- batch_sizing.py
    |   |-- checkpointing.py
    |   |-- cli_logging.py
    |   |-- distributed.py
    |   |-- onnx_export.py
    |   |-- optimizers.py
    |   |-- scheduler.py
    |   `-- telemetry.py
    |-- digit_models/            # 15 digit training scripts (SOAP/AdamW/Muon x 5 resolutions)
    |-- lowercase_models/        # 12 lowercase letter training scripts (SOAP/AdamW/Muon x 4 resolutions)
    |-- uppercase_models/        # 12 uppercase letter training scripts (SOAP/AdamW/Muon x 4 resolutions)
    |-- router_models/           # 5 router training scripts (Ranger x 5 resolutions)
    |-- _precomputed_cache/      # Precomputed dataset cache (.npz files)
    |-- ocr_pipeline_mnist.py    # Inference pipeline
    |-- supplementary_data.py    # Shared dataset loading
    |-- build_dataset_cache.py   # One-time dataset cache builder
    |-- setup_packages.py        # One-shot package installer
    |-- requirements.txt
    |-- run_all_training.ps1     # Runs all 44 training scripts in tier order
    |-- v4_CHANGELOG.md
    `-- README.md

Model outputs (checkpoints, ONNX exports, logs, training curves) are
created alongside each training script when it runs — none exist yet,
training hasn't started under the v4 naming.

## Usage

See each training script's own module docstring for its specific CLI
usage. `run_all_training.ps1` runs all 44 in resolution-tier order —
see its header comment for skip/resume behavior. `ocr_pipeline_mnist.py
--help` for pipeline usage.

## Requirements

See `requirements.txt`; `setup_packages.py` is a one-shot installer for
everything listed there plus the optimizer packages each training
script needs.
