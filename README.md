# EMNIST Ensemble v4

v4 of a multi-optimizer PyTorch OCR ensemble for handwritten digit and
letter recognition. Optimizers: SOAP, AdamW, Muon. Includes a
case-classifying router (digit/uppercase/lowercase/unknown) ahead of
separate uppercase and lowercase letter-reading models. No
post-processing — raw output only.

This is v4 of the project, renamed from v3 — see `v4_CHANGELOG.md` for
the full rename history and rationale.

## Current structure

Every file below is one that is (or will be, once added) tracked in
git — `.pt` checkpoint weights and `_precomputed_cache/`'s `.npz`
files are gitignored and never appear here. Output files follow a
fixed pattern per trained config: `.onnx` is the exported inference
model, `_curves.png` is the loss/accuracy plot, `_log.csv` is the
per-epoch + hardware telemetry log, and `_cli_<timestamp>.txt` is the
full console transcript for that training run (some configs have more
than one transcript because they were run more than once).

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
    |   |-- seeding.py
    |   `-- telemetry.py
    |-- digit_models/            # 15 digit training scripts (SOAP/AdamW/Muon x 5 resolutions)
    |   |-- v4_mnist_digit_adamw_16.py
    |   |-- v4_mnist_digit_adamw_16/
    |   |   |-- v4_mnist_digit_adamw_16.onnx
    |   |   |-- v4_mnist_digit_adamw_16_cli_20260813_223018.txt
    |   |   |-- v4_mnist_digit_adamw_16_curves.png
    |   |   `-- v4_mnist_digit_adamw_16_log.csv
    |   |-- v4_mnist_digit_adamw_28.py
    |   |-- v4_mnist_digit_adamw_28/
    |   |   |-- v4_mnist_digit_adamw_28.onnx
    |   |   |-- v4_mnist_digit_adamw_28_cli_20260814_010715.txt
    |   |   |-- v4_mnist_digit_adamw_28_curves.png
    |   |   `-- v4_mnist_digit_adamw_28_log.csv
    |   |-- v4_mnist_digit_adamw_32.py
    |   |-- v4_mnist_digit_adamw_32/
    |   |   |-- v4_mnist_digit_adamw_32.onnx
    |   |   |-- v4_mnist_digit_adamw_32_cli_20260814_130924.txt
    |   |   |-- v4_mnist_digit_adamw_32_cli_20260814_233449.txt
    |   |   |-- v4_mnist_digit_adamw_32_curves.png
    |   |   `-- v4_mnist_digit_adamw_32_log.csv
    |   |-- v4_mnist_digit_adamw_64.py
    |   |-- v4_mnist_digit_adamw_64/
    |   |   |-- v4_mnist_digit_adamw_64.onnx
    |   |   |-- v4_mnist_digit_adamw_64_cli_20260816_000215.txt
    |   |   |-- v4_mnist_digit_adamw_64_curves.png
    |   |   `-- v4_mnist_digit_adamw_64_log.csv
    |   |-- v4_mnist_digit_adamw_128.py
    |   |-- v4_mnist_digit_muon_16.py
    |   |-- v4_mnist_digit_muon_16/
    |   |   |-- v4_mnist_digit_muon_16.onnx
    |   |   |-- v4_mnist_digit_muon_16_cli_20260813_223156.txt
    |   |   |-- v4_mnist_digit_muon_16_curves.png
    |   |   `-- v4_mnist_digit_muon_16_log.csv
    |   |-- v4_mnist_digit_muon_28.py
    |   |-- v4_mnist_digit_muon_28/
    |   |   |-- v4_mnist_digit_muon_28.onnx
    |   |   |-- v4_mnist_digit_muon_28_cli_20260814_045142.txt
    |   |   |-- v4_mnist_digit_muon_28_curves.png
    |   |   `-- v4_mnist_digit_muon_28_log.csv
    |   |-- v4_mnist_digit_muon_32.py
    |   |-- v4_mnist_digit_muon_32/
    |   |   |-- v4_mnist_digit_muon_32.onnx
    |   |   |-- v4_mnist_digit_muon_32_cli_20260814_234617.txt
    |   |   |-- v4_mnist_digit_muon_32_curves.png
    |   |   `-- v4_mnist_digit_muon_32_log.csv
    |   |-- v4_mnist_digit_muon_64.py
    |   |-- v4_mnist_digit_muon_64/
    |   |   |-- v4_mnist_digit_muon_64.onnx
    |   |   |-- v4_mnist_digit_muon_64_cli_20260816_030750.txt
    |   |   |-- v4_mnist_digit_muon_64_curves.png
    |   |   `-- v4_mnist_digit_muon_64_log.csv
    |   |-- v4_mnist_digit_muon_128.py
    |   |-- v4_mnist_digit_soap_16.py
    |   |-- v4_mnist_digit_soap_16/
    |   |   |-- v4_mnist_digit_soap_16.onnx
    |   |   |-- v4_mnist_digit_soap_16_cli_20260813_215204.txt
    |   |   |-- v4_mnist_digit_soap_16_cli_20260813_222706.txt
    |   |   |-- v4_mnist_digit_soap_16_curves.png
    |   |   `-- v4_mnist_digit_soap_16_log.csv
    |   |-- v4_mnist_digit_soap_28.py
    |   |-- v4_mnist_digit_soap_28/
    |   |   |-- v4_mnist_digit_soap_28.onnx
    |   |   |-- v4_mnist_digit_soap_28_cli_20260813_225730.txt
    |   |   |-- v4_mnist_digit_soap_28_curves.png
    |   |   `-- v4_mnist_digit_soap_28_log.csv
    |   |-- v4_mnist_digit_soap_32.py
    |   |-- v4_mnist_digit_soap_32/
    |   |   |-- v4_mnist_digit_soap_32.onnx
    |   |   |-- v4_mnist_digit_soap_32_cli_20260814_072820.txt
    |   |   |-- v4_mnist_digit_soap_32_cli_20260814_125141.txt
    |   |   |-- v4_mnist_digit_soap_32_curves.png
    |   |   `-- v4_mnist_digit_soap_32_log.csv
    |   |-- v4_mnist_digit_soap_64.py
    |   |-- v4_mnist_digit_soap_64/
    |   |   |-- v4_mnist_digit_soap_64.onnx
    |   |   |-- v4_mnist_digit_soap_64_cli_20260815_033624.txt
    |   |   |-- v4_mnist_digit_soap_64_cli_20260815_203439.txt
    |   |   |-- v4_mnist_digit_soap_64_cli_20260815_231040.txt
    |   |   |-- v4_mnist_digit_soap_64_curves.png
    |   |   `-- v4_mnist_digit_soap_64_log.csv
    |   `-- v4_mnist_digit_soap_128.py
    |-- lowercase_models/        # 12 lowercase letter training scripts (SOAP/AdamW/Muon x 4 resolutions)
    |   |-- v4_mnist_letter_lc_adamw_28.py
    |   |-- v4_mnist_letter_lc_adamw_28/
    |   |   |-- v4_mnist_letter_lc_adamw_28.onnx
    |   |   |-- v4_mnist_letter_lc_adamw_28_cli_20260814_063230.txt
    |   |   |-- v4_mnist_letter_lc_adamw_28_curves.png
    |   |   `-- v4_mnist_letter_lc_adamw_28_log.csv
    |   |-- v4_mnist_letter_lc_adamw_32.py
    |   |-- v4_mnist_letter_lc_adamw_32/
    |   |   |-- v4_mnist_letter_lc_adamw_32.onnx
    |   |   |-- v4_mnist_letter_lc_adamw_32_cli_20260815_021208.txt
    |   |   |-- v4_mnist_letter_lc_adamw_32_curves.png
    |   |   `-- v4_mnist_letter_lc_adamw_32_log.csv
    |   |-- v4_mnist_letter_lc_adamw_64.py
    |   |-- v4_mnist_letter_lc_adamw_64/   # run in progress -- no onnx/curves.png yet
    |   |   |-- v4_mnist_letter_lc_adamw_64_cli_20260816_103926.txt
    |   |   `-- v4_mnist_letter_lc_adamw_64_log.csv
    |   |-- v4_mnist_letter_lc_adamw_128.py
    |   |-- v4_mnist_letter_lc_muon_28.py
    |   |-- v4_mnist_letter_lc_muon_28/
    |   |   |-- v4_mnist_letter_lc_muon_28.onnx
    |   |   |-- v4_mnist_letter_lc_muon_28_cli_20260814_065006.txt
    |   |   |-- v4_mnist_letter_lc_muon_28_curves.png
    |   |   `-- v4_mnist_letter_lc_muon_28_log.csv
    |   |-- v4_mnist_letter_lc_muon_32.py
    |   |-- v4_mnist_letter_lc_muon_32/
    |   |   |-- v4_mnist_letter_lc_muon_32.onnx
    |   |   |-- v4_mnist_letter_lc_muon_32_cli_20260815_025231.txt
    |   |   |-- v4_mnist_letter_lc_muon_32_curves.png
    |   |   `-- v4_mnist_letter_lc_muon_32_log.csv
    |   |-- v4_mnist_letter_lc_muon_64.py
    |   |-- v4_mnist_letter_lc_muon_128.py
    |   |-- v4_mnist_letter_lc_soap_28.py
    |   |-- v4_mnist_letter_lc_soap_28/
    |   |   |-- v4_mnist_letter_lc_soap_28.onnx
    |   |   |-- v4_mnist_letter_lc_soap_28_cli_20260814_062021.txt
    |   |   |-- v4_mnist_letter_lc_soap_28_curves.png
    |   |   `-- v4_mnist_letter_lc_soap_28_log.csv
    |   |-- v4_mnist_letter_lc_soap_32.py
    |   |-- v4_mnist_letter_lc_soap_32/
    |   |   |-- v4_mnist_letter_lc_soap_32.onnx
    |   |   |-- v4_mnist_letter_lc_soap_32_cli_20260815_015729.txt
    |   |   |-- v4_mnist_letter_lc_soap_32_curves.png
    |   |   `-- v4_mnist_letter_lc_soap_32_log.csv
    |   |-- v4_mnist_letter_lc_soap_64.py
    |   |-- v4_mnist_letter_lc_soap_64/
    |   |   |-- v4_mnist_letter_lc_soap_64.onnx
    |   |   |-- v4_mnist_letter_lc_soap_64_cli_20260816_095535.txt
    |   |   |-- v4_mnist_letter_lc_soap_64_curves.png
    |   |   `-- v4_mnist_letter_lc_soap_64_log.csv
    |   `-- v4_mnist_letter_lc_soap_128.py
    |-- uppercase_models/        # 12 uppercase letter training scripts (SOAP/AdamW/Muon x 4 resolutions)
    |   |-- v4_mnist_letter_uc_adamw_28.py
    |   |-- v4_mnist_letter_uc_adamw_28/
    |   |   |-- v4_mnist_letter_uc_adamw_28.onnx
    |   |   |-- v4_mnist_letter_uc_adamw_28_cli_20260814_053632.txt
    |   |   |-- v4_mnist_letter_uc_adamw_28_curves.png
    |   |   `-- v4_mnist_letter_uc_adamw_28_log.csv
    |   |-- v4_mnist_letter_uc_adamw_32.py
    |   |-- v4_mnist_letter_uc_adamw_32/
    |   |   |-- v4_mnist_letter_uc_adamw_32.onnx
    |   |   |-- v4_mnist_letter_uc_adamw_32_cli_20260815_005350.txt
    |   |   |-- v4_mnist_letter_uc_adamw_32_curves.png
    |   |   `-- v4_mnist_letter_uc_adamw_32_log.csv
    |   |-- v4_mnist_letter_uc_adamw_64.py
    |   |-- v4_mnist_letter_uc_adamw_64/
    |   |   |-- v4_mnist_letter_uc_adamw_64.onnx
    |   |   |-- v4_mnist_letter_uc_adamw_64_cli_20260816_080955.txt
    |   |   |-- v4_mnist_letter_uc_adamw_64_curves.png
    |   |   `-- v4_mnist_letter_uc_adamw_64_log.csv
    |   |-- v4_mnist_letter_uc_adamw_128.py
    |   |-- v4_mnist_letter_uc_muon_28.py
    |   |-- v4_mnist_letter_uc_muon_28/
    |   |   |-- v4_mnist_letter_uc_muon_28.onnx
    |   |   |-- v4_mnist_letter_uc_muon_28_cli_20260814_060531.txt
    |   |   |-- v4_mnist_letter_uc_muon_28_curves.png
    |   |   `-- v4_mnist_letter_uc_muon_28_log.csv
    |   |-- v4_mnist_letter_uc_muon_32.py
    |   |-- v4_mnist_letter_uc_muon_32/
    |   |   |-- v4_mnist_letter_uc_muon_32.onnx
    |   |   |-- v4_mnist_letter_uc_muon_32_cli_20260815_014049.txt
    |   |   |-- v4_mnist_letter_uc_muon_32_curves.png
    |   |   `-- v4_mnist_letter_uc_muon_32_log.csv
    |   |-- v4_mnist_letter_uc_muon_64.py
    |   |-- v4_mnist_letter_uc_muon_64/
    |   |   |-- v4_mnist_letter_uc_muon_64.onnx
    |   |   |-- v4_mnist_letter_uc_muon_64_cli_20260816_091734.txt
    |   |   |-- v4_mnist_letter_uc_muon_64_curves.png
    |   |   `-- v4_mnist_letter_uc_muon_64_log.csv
    |   |-- v4_mnist_letter_uc_muon_128.py
    |   |-- v4_mnist_letter_uc_soap_28.py
    |   |-- v4_mnist_letter_uc_soap_28/
    |   |   |-- v4_mnist_letter_uc_soap_28.onnx
    |   |   |-- v4_mnist_letter_uc_soap_28_cli_20260814_050958.txt
    |   |   |-- v4_mnist_letter_uc_soap_28_curves.png
    |   |   `-- v4_mnist_letter_uc_soap_28_log.csv
    |   |-- v4_mnist_letter_uc_soap_32.py
    |   |-- v4_mnist_letter_uc_soap_32/
    |   |   |-- v4_mnist_letter_uc_soap_32.onnx
    |   |   |-- v4_mnist_letter_uc_soap_32_cli_20260815_003007.txt
    |   |   |-- v4_mnist_letter_uc_soap_32_curves.png
    |   |   `-- v4_mnist_letter_uc_soap_32_log.csv
    |   |-- v4_mnist_letter_uc_soap_64.py
    |   |-- v4_mnist_letter_uc_soap_64/
    |   |   |-- v4_mnist_letter_uc_soap_64.onnx
    |   |   |-- v4_mnist_letter_uc_soap_64_cli_20260816_070651.txt
    |   |   |-- v4_mnist_letter_uc_soap_64_curves.png
    |   |   `-- v4_mnist_letter_uc_soap_64_log.csv
    |   `-- v4_mnist_letter_uc_soap_128.py
    |-- router_models/           # 5 router training scripts (Ranger x 5 resolutions)
    |   |-- v4_mnist_router_ranger_16.py
    |   |-- v4_mnist_router_ranger_16/
    |   |   |-- v4_mnist_router_ranger_16.onnx
    |   |   |-- v4_mnist_router_ranger_16_cli_20260813_214251.txt
    |   |   |-- v4_mnist_router_ranger_16_curves.png
    |   |   `-- v4_mnist_router_ranger_16_log.csv
    |   |-- v4_mnist_router_ranger_28.py
    |   |-- v4_mnist_router_ranger_28/
    |   |   |-- v4_mnist_router_ranger_28.onnx
    |   |   |-- v4_mnist_router_ranger_28_cli_20260813_223358.txt
    |   |   |-- v4_mnist_router_ranger_28_curves.png
    |   |   `-- v4_mnist_router_ranger_28_log.csv
    |   |-- v4_mnist_router_ranger_32.py
    |   |-- v4_mnist_router_ranger_32/
    |   |   |-- v4_mnist_router_ranger_32.onnx
    |   |   |-- v4_mnist_router_ranger_32_cli_20260814_070238.txt
    |   |   |-- v4_mnist_router_ranger_32_curves.png
    |   |   `-- v4_mnist_router_ranger_32_log.csv
    |   |-- v4_mnist_router_ranger_64.py
    |   |-- v4_mnist_router_ranger_64/
    |   |   |-- v4_mnist_router_ranger_64.onnx
    |   |   |-- v4_mnist_router_ranger_64_cli_20260815_030439.txt
    |   |   |-- v4_mnist_router_ranger_64_curves.png
    |   |   `-- v4_mnist_router_ranger_64_log.csv
    |   `-- v4_mnist_router_ranger_128.py
    |-- ocr_pipeline_mnist.py    # Inference pipeline
    |-- supplementary_data.py    # Shared dataset loading
    |-- build_dataset_cache.py   # One-time dataset cache builder
    |-- setup_packages.py        # One-shot package installer
    |-- requirements.txt
    |-- run_all_training.ps1     # Runs all 44 training scripts in tier order
    |-- v4_CHANGELOG.md
    |-- CLAUDE.md
    |-- README.md
    `-- .gitignore

32 of the 44 configs are fully trained (`.onnx` present).
`v4_mnist_letter_lc_adamw_64.py` is currently mid-run (no `.onnx`/
`curves.png` yet — those only get written once training finishes).
The remaining 11 haven't been started and have no output folder at
all: the `_128` tier across all four model families, plus lowercase
`_muon_64`. `.pt` checkpoint weight files (including the in-progress
run's `_resume.pt`) and `_precomputed_cache/`'s `.npz` files exist
locally but are gitignored and never reach GitHub, so none of those
appear above.

## Usage

See each training script's own module docstring for its specific CLI
usage. `run_all_training.ps1` runs all 44 in resolution-tier order —
see its header comment for skip/resume behavior. `ocr_pipeline_mnist.py
--help` for pipeline usage.

## Telemetry

Every training script collects per-epoch hardware telemetry via
`common/telemetry.py`'s `HardwareMonitor` — GPU utilization/temperature
(including thermal margin to throttle)/power/clocks/VRAM (including a
system-wide-vs-this-process breakdown), per-core CPU, system RAM/swap,
disk I/O, and PCIe link state/throughput — logged to each run's own
`_log.csv`: one row per hardware sample point plus a per-epoch summary
row (min/avg/max). See `common/cli_logging.py`'s `HW_TELEMETRY_FIELDS`/
`_POINT_FIELDS` for the exact columns, and `v4_CHANGELOG.md` for what
was added and why.

## Requirements

See `requirements.txt`; `setup_packages.py` is a one-shot installer for
everything listed there plus the optimizer packages each training
script needs.
