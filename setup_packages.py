"""
setup_packages.py
===================
One-shot Python package installer for mnist-ocr-ensemble-v3.

Installs everything needed to run:
  - The 15 digit training scripts (v3_mnist_digit_{soap,adamw,muon}_{16,28,32,64,128}.py)
  - The 5 router training scripts (v3_mnist_router_ranger_{16,28,32,64,128}.py)
  - The shared supplementary_data.py dataset loader and common/ package
  - The ONNX inference pipeline (ocr_pipeline_mnist.py)

Muon and Ranger need NO separate package — both are implemented inline
directly inside their respective training scripts, not imported from a
third-party optimizer library (see v3_CHANGELOG.md's modularization-scope
note). Nothing here installs anything for them.

PyTorch only — no TensorFlow, no Keras, nothing outside the PyTorch
ecosystem is installed by this script.

Confirmed against this project's actual working environment (see
v3_CHANGELOG.md's CUDA-allocator entry and run_all_training.ps1's
python.exe path):
    Python 3.14, torch==2.13.0+cu130, torchvision==0.28.0

Scope note: this script installs Python packages via pip only. It does
NOT touch the NVIDIA driver, CUDA, or the Python interpreter itself —
make sure those are already working (torch.cuda.is_available() == True)
before relying on GPU training.

Usage:
    python setup_packages.py                 # installs everything
    python setup_packages.py --check          # only checks, installs nothing
    python setup_packages.py --skip-torch     # skip torch/torchvision (already installed)
    python setup_packages.py --cuda cu121     # use a different CUDA wheel index
    python setup_packages.py --skip-onnx      # skip onnxruntime-gpu/onnx/opencv — see the
                                               # warning printed by this flag before using it

Every install THIS SCRIPT makes is run through sys.executable -m pip,
not a bare `pip` command — avoids installing into the wrong Python
environment if more than one is present on the machine. Launch this
script itself with the interpreter you actually intend to use.
"""

import argparse
import subprocess
import sys
import importlib


# -----------------------------------------------------------------------
# Package groups
# -----------------------------------------------------------------------

# (import_name, pip_package_name, why_its_needed)
# import_name is what `import X` uses; pip_package_name is what you'd
# actually pip install — these differ for a few of these (e.g. cv2 vs
# opencv-python, PIL vs Pillow).

CORE_TRAINING_PACKAGES = [
    ("numpy",       "numpy",       "array ops, used throughout"),
    ("matplotlib",  "matplotlib",  "training curve plots"),
    ("PIL",         "Pillow",      "image loading — supplementary_data.py, router --infer"),
    ("psutil",      "psutil",      "CPU%/RAM telemetry — common/telemetry.py, common/batch_sizing.py"),
]

OPTIMIZER_PACKAGES = [
    ("schedulefree",      "schedulefree",      "Schedule-Free AdamW — v3_mnist_digit_adamw_*.py"),
    ("pytorch_optimizer", "pytorch_optimizer", "SOAP optimizer — v3_mnist_digit_soap_*.py"),
    # Muon and Ranger are NOT here — both are inlined directly into
    # v3_mnist_digit_muon_*.py / v3_mnist_router_ranger_*.py, no
    # third-party optimizer package needed for either.
]

ONNX_INFERENCE_PACKAGES = [
    ("onnxruntime",  "onnxruntime-gpu",  "GPU-accelerated ONNX inference — used by "
                                          "ocr_pipeline_mnist.py for actual inference "
                                          "(NOT plain 'onnxruntime' — that installs CPU-only and "
                                          "CUDAExecutionProvider silently won't be available)"),
    ("onnx",         "onnx",             "ONNX export validation — imported and actually used "
                                          "(onnx.checker.check_model()/onnx.load()) only by the 5 "
                                          "v3_mnist_digit_soap_*.py scripts' validate=True export "
                                          "path (common/onnx_export.py). Not needed by adamw/muon "
                                          "digit scripts or the router scripts, which export with "
                                          "validate=False by default."),
    ("cv2",          "opencv-python",    "image preprocessing — used only by ocr_pipeline_mnist.py, "
                                          "not needed for training"),
]

# Cross-platform, optional. Only used as a fallback ARDIS-archive
# extraction backend in supplementary_data.py, and only if none of
# 7-Zip/unrar/unar are already found on PATH — training and inference
# both work without it if ARDIS is already extracted or one of those
# external tools is present.
OPTIONAL_PACKAGES = [
    ("rarfile", "rarfile", "ARDIS auto-download fallback extractor — supplementary_data.py; "
                            "still needs 7-Zip/unrar/unar as its own extraction backend"),
]


def run_pip(args: list, description: str) -> bool:
    """Runs `<this interpreter> -m pip <args>`, streaming output live."""
    cmd = [sys.executable, "-m", "pip"] + args
    print(f"\n[Install] {description}")
    print(f"          $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[Warning] Command exited with code {result.returncode} — see output above.")
        return False
    return True


def is_installed(import_name: str) -> bool:
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def check_packages(packages: list, label: str):
    print(f"\n[Check] {label}")
    all_ok = True
    for import_name, pip_name, why in packages:
        ok = is_installed(import_name)
        status = "OK" if ok else "MISSING"
        print(f"  [{status:^7}] {pip_name:20s} (import {import_name:20s}) — {why}")
        if not ok:
            all_ok = False
    return all_ok


def install_torch(cuda_tag: str):
    """
    Installs torch + torchvision from PyTorch's own wheel index for the
    given CUDA build tag (e.g. 'cu130', matching this project's confirmed
    working version — see v3_CHANGELOG.md and run_all_training.ps1's
    python.exe path). Uses --index-url as primary, per
    PyTorch's own documented install instructions; if that fails due to a
    missing dependency mirror (a known, documented issue for some CUDA
    tags), retry with --extra-index-url instead, which several real users
    report as the working fallback.
    """
    index_url = f"https://download.pytorch.org/whl/{cuda_tag}"
    print(f"\n[Torch] Installing torch + torchvision for {cuda_tag} from {index_url}")

    ok = run_pip(
        ["install", "torch", "torchvision", "--index-url", index_url],
        f"torch + torchvision ({cuda_tag})",
    )
    if not ok:
        print("[Torch] --index-url install failed — retrying with --extra-index-url "
              "(known fallback for some CUDA wheel indexes missing a dependency mirror)")
        run_pip(
            ["install", "torch", "torchvision", "--extra-index-url", index_url],
            f"torch + torchvision ({cuda_tag}, fallback)",
        )


def install_group(packages: list, label: str):
    print(f"\n[Install] {label}")
    for import_name, pip_name, why in packages:
        if is_installed(import_name):
            print(f"  [SKIP] {pip_name} already installed")
            continue
        run_pip(["install", pip_name], f"{pip_name} — {why}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Only check what's installed, don't install anything")
    parser.add_argument("--skip-torch", action="store_true",
                        help="Skip torch/torchvision install (use if already installed)")
    parser.add_argument("--cuda", default="cu130",
                        help="CUDA wheel tag for torch (default: cu130, matching this "
                             "project's confirmed working environment — see "
                             "v3_CHANGELOG.md and run_all_training.ps1's python.exe "
                             "path). Use 'cpu' for a CPU-only install.")
    parser.add_argument("--skip-onnx", action="store_true",
                        help="Skip onnxruntime-gpu/onnx/opencv. WARNING: onnx is required by "
                             "the 5 SOAP digit scripts' export-validation path, and "
                             "onnxruntime-gpu is required by ocr_pipeline_mnist.py — only "
                             "opencv-python (cv2) is genuinely inference-only. Safe to use "
                             "this flag only if you won't run SOAP training or "
                             "ocr_pipeline_mnist.py.")
    args = parser.parse_args()

    print("=" * 70)
    print("  mnist-ocr-ensemble-v3 — package setup")
    print(f"  Interpreter: {sys.executable}")
    print(f"  Python:      {sys.version.split()[0]}")
    print("=" * 70)

    if args.check:
        ok1 = check_packages(CORE_TRAINING_PACKAGES, "Core training packages")
        ok2 = check_packages(OPTIMIZER_PACKAGES, "Optimizer packages (Muon/Ranger need none — inlined)")
        ok3 = check_packages(ONNX_INFERENCE_PACKAGES, "ONNX inference / SOAP-export-validation packages")
        check_packages(OPTIONAL_PACKAGES, "Optional (ARDIS auto-download fallback)")
        torch_ok = is_installed("torch")
        print(f"\n[Check] torch: {'OK' if torch_ok else 'MISSING'}")
        if torch_ok:
            import torch
            print(f"         version: {torch.__version__}")
            print(f"         CUDA available: {torch.cuda.is_available()}")
        print()
        if ok1 and ok2 and ok3 and torch_ok:
            print("[Done] Everything needed is installed.")
        else:
            print("[Done] Some packages are missing — run without --check to install them.")
        return

    if not args.skip_torch:
        install_torch(args.cuda)
    else:
        print("\n[Torch] --skip-torch passed, not touching torch/torchvision")

    install_group(CORE_TRAINING_PACKAGES, "Core training packages")
    install_group(OPTIMIZER_PACKAGES, "Optimizer packages (schedulefree, pytorch_optimizer)")

    if not args.skip_onnx:
        install_group(ONNX_INFERENCE_PACKAGES, "ONNX inference / SOAP-export-validation packages")
    else:
        print("\n[ONNX] --skip-onnx passed, skipping onnxruntime-gpu/onnx/opencv")
        print("       [Warning] onnx is required by the 5 v3_mnist_digit_soap_*.py scripts'")
        print("       export-validation path and will raise ImportError there if missing.")
        print("       onnxruntime-gpu is required by ocr_pipeline_mnist.py. Only opencv-python")
        print("       (cv2) is genuinely inference-only. Re-run without --skip-onnx if you plan")
        print("       to train SOAP models or run the inference pipeline.")

    install_group(OPTIONAL_PACKAGES, "Optional packages")

    print("\n" + "=" * 70)
    print("[Done] Setup complete. Run with --check to verify everything installed correctly.")
    print("=" * 70)

    if is_installed("torch"):
        import torch
        print(f"\ntorch {torch.__version__} — CUDA available: {torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            print("[Warning] CUDA is NOT available in this torch install. If you have an "
                  "NVIDIA GPU, this usually means the wrong wheel was installed (CPU-only "
                  "instead of a CUDA build) — re-run with the correct --cuda tag for your "
                  "driver, e.g. --cuda cu130.")


if __name__ == "__main__":
    main()
