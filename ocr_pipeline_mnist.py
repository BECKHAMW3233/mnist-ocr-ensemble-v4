"""
ocr_pipeline_mnist.py
=====================
MNIST OCR Pipeline — Single model, ensemble, or any combination of ONNX
digit models, with an optional router pre-filter. No post-processing. No
remapping. Raw inference only.

Usage:
  Single model:
    python ocr_pipeline_mnist.py --models v3_mnist_digit_soap_64/v3_mnist_digit_soap_64.onnx image.jpg

  Multiple models (ensemble vote):
    python ocr_pipeline_mnist.py --models v3_mnist_digit_soap_64/v3_mnist_digit_soap_64.onnx v3_mnist_digit_adamw_64/v3_mnist_digit_adamw_64.onnx image.jpg

  Scan a directory for all .onnx files (recursive):
    python ocr_pipeline_mnist.py --model-dir . digit.jpg

  Glob patterns for images:
    python ocr_pipeline_mnist.py --models v3_mnist_digit_soap_64/*.onnx test*.jpg

  Multiple images:
    python ocr_pipeline_mnist.py --models v3_mnist_digit_soap_64/v3_mnist_digit_soap_64.onnx a.jpg b.jpg c.jpg

  With the router pre-filter (see v3_mnist_router_ranger_*.py) — each
  detected box is classified digit/UC/LC/[UNK] BEFORE the digit ensemble
  runs; only a "digit" verdict reaches the (expensive, 10-class) ensemble
  at all — UC/LC boxes are output directly as terminal labels, and
  [UNK] boxes are skipped entirely:
    python ocr_pipeline_mnist.py --router-model v3_mnist_router_ranger_64/v3_mnist_router_ranger_64.onnx --models v3_mnist_digit_soap_64/v3_mnist_digit_soap_64.onnx image.jpg

  --models and --model-dir are mutually exclusive — use one or the other.
  --router-model is independent of both — optional either way; omitting
  it reproduces the pipeline's original router-free behavior exactly
  (every detected box goes straight to the digit ensemble, as if every
  box were a confident "digit" verdict).

Output markers:
  ??            A character position was detected but the digit ensemble
                could not agree on a digit (models split with no
                majority/weighted winner). This is now the ONLY thing ??
                means in v3 — v2's second meaning ("the gate confidently
                said not-digit") no longer applies, since the router
                fully replaces the standalone binary gate rather than
                running alongside it (see v3_CHANGELOG.md's router-wiring
                entry for the full reasoning) — a router "not a digit"
                call now produces UC, LC, or [UNK] instead, never ??.
                Distinct from [NON-DIGIT?] below, which means every
                DIGIT model's own top confidence was too low to trust
                any answer at all — a symptom of ensemble uncertainty,
                not the router's own verdict.
  [NON-DIGIT?]  Best confidence across all digit models fell below
                NON_DIGIT_CONF_FLOOR — likely not a digit at all, not
                just a hard-to-read one. Only ever produced by the digit
                ensemble itself; unrelated to --router-model (a box the
                router called "digit" still goes through this same check
                — the router only decides whether the ensemble runs at
                all, not how it scores its own confidence once it does).
  UC / LC       --router-model classified this box as an uppercase or
                lowercase letter (confidence at or above
                --router-unsure-floor). Terminal labels — no letter-
                identity or case-specific reading model exists yet (see
                v3_mnist_router_ranger_*.py's own docstring). The digit
                ensemble never runs on a UC/LC box.
  [UNK]         --router-model's own top-class confidence for this box
                fell below --router-unsure-floor (default
                ROUTER_UNSURE_FLOOR below) — the router itself could not
                classify the box as digit, UC, or LC with enough
                confidence to commit. The digit ensemble never runs on
                an [UNK] box (mirroring how a confident not-digit call
                skipped the ensemble under v2's gate). Excluded from
                accuracy scoring, with its own warn line, its own count
                in PER-MODEL SUMMARY ([R?]), and its own flag in
                CHARACTER DETAIL (✗ router) — see CHARACTER DETAIL /
                PER-MODEL SUMMARY / ACCURACY below.

Note: get_boxes() has no concept of a grid — it clusters detected contours into
"lines" purely by proximity and vertical position, not fixed rows/columns. Real
handwritten digits are rarely uniform: they vary in size, spacing, and vertical
alignment even within the same line, and get_boxes() has no ground truth to
compare against. A digit that fails the width filter (e.g. a stray connecting
stroke fuses onto it, making its box wider than a normal character) is now
recovered by a rescue pass if doing so fills an unusually large gap in its
line — see get_boxes()'s own docstring for exactly what is and isn't covered.
This does not catch every case: a digit rejected for height or aspect ratio,
or one whose vertical center lands outside its line's clustering window, can
still be silently absent from the output with no ?? or other marker, because
no character-slot was ever created for it. ?? only covers characters that WERE
detected and handed to the models but couldn't be agreed on afterward.

Letters and box detection: get_boxes()'s size windows and aspect-ratio
bounds were originally tuned against digit test images only. Checked for
v3 (not assumed fine) against realistic letter bounding-box shapes before
routing became part of this pipeline — see get_boxes()'s own docstring
for what was verified and the one change made (the aspect-ratio floor).

Requires: pip install onnxruntime-gpu opencv-python numpy
"""

import cv2
import numpy as np

# Preload CUDA/cuDNN DLLs BEFORE creating any onnxruntime session. Without
# this, onnxruntime-gpu can register CUDAExecutionProvider as "available"
# (it shows up in get_available_providers()) but silently fall back to CPU
# per-session with no exception and no error message — the DLLs it needs
# at runtime were never actually loaded into the process. Two ways to fix
# it, tried in order:
#   1. If PyTorch with CUDA support is installed (common on a machine that
#      also trains models), importing it loads the same CUDA/cuDNN DLLs
#      onnxruntime-gpu needs — no separate CUDA toolkit/cuDNN install
#      required. This is the easiest fix when torch+CUDA is already known
#      to work (e.g. this project's own training scripts use it).
#   2. onnxruntime-gpu >= 1.21.0 ships its own onnxruntime.preload_dlls()
#      that does the same thing without needing PyTorch at all.
# If neither succeeds, CUDA sessions will still silently fall back to CPU
# — the per-model load diagnostic later in this script explains that case
# and points at checking CUDA/cuDNN versions directly.
try:
    import torch  # noqa: F401  (import side-effect: preloads CUDA/cuDNN DLLs)
except ImportError:
    pass
import onnxruntime as ort
try:
    ort.preload_dlls()
except (AttributeError, Exception):
    # AttributeError: onnxruntime-gpu < 1.21.0, function doesn't exist yet.
    # Other exceptions: preload itself failed — not fatal, CPU fallback
    # still works, just without the fix this preload attempts to provide.
    pass

from collections import Counter
import sys
import glob
import os
import argparse
import datetime
import re


# ── Logging ──────────────────────────────────────────────────────────────────

class _Tee:
    """Mirrors stdout to a timestamped log file beside this script."""
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(script_dir, "pipeline_logs")
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"ocr_mnist_{ts}.log")
        self._log = open(log_path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = self
        print(f"  Log: {log_path}")

    def write(self, data):
        self._stdout.write(data)
        self._log.write(data)

    def flush(self):
        self._stdout.flush()
        self._log.flush()

    def close(self):
        sys.stdout = self._stdout
        self._log.close()


# ── Constants ─────────────────────────────────────────────────────────────────

LABELS = [str(i) for i in range(10)]

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# If the best confidence across ALL digit models for a character is below this
# threshold, the character is flagged as a likely non-digit rather than
# silently output as a low-confidence digit guess. Tune this value based on
# real-world testing. Unrelated to the router — see ROUTER_UNSURE_FLOOR below.
NON_DIGIT_CONF_FLOOR = 0.40

# Router (see v3_mnist_router_ranger_*.py) confidence floor. The router's
# output head is 3-class (digit / UC / LC) — there is no trained "unknown"
# class. If the router's own top-class softmax probability for a box falls
# below this floor, the pipeline renders [UNK] instead of the top class,
# and the digit ensemble never runs on that box. Must match the value each
# router model was itself validated against if you want inference-time
# behavior to match training-time expectations — override per run with
# --router-unsure-floor rather than editing this constant, so it can be
# tuned across runs against real test images without touching the script.
ROUTER_UNSURE_FLOOR = 0.6

# Must match v3_mnist_router_ranger_*.py's own LABEL_MAP order exactly —
# index 0 = digit, 1 = UC (uppercase), 2 = LC (lowercase). This is a
# cross-file contract, not re-derived from the ONNX file itself (the
# router's output is just 3 unlabeled logits on the wire).
ROUTER_LABELS = ["digit", "UC", "LC"]

# Letter-identity ensembles (see v3_mnist_letter_{uc,lc}_*.py) — used by
# predict_char_topn()'s labels= param when a router-classified UC/LC box
# is handed to the matching letter ensemble instead of staying a bare
# terminal label. Index order must match each letter script's own
# LABEL_MAP (dense 0-25, A=0..Z=25 / a=0..z=25) — a cross-file contract,
# same as ROUTER_LABELS above.
UC_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LC_LABELS = list("abcdefghijklmnopqrstuvwxyz")


# ── Image utilities ───────────────────────────────────────────────────────────

def get_model_input_size(session) -> int:
    return int(session.get_inputs()[0].shape[2])


def short_model_name(path: str) -> str:
    """Extract a readable short name from the ONNX filename.

    v3_mnist_digit_soap_64.onnx      -> digit_soap_64
    v3_mnist_router_ranger_64.onnx   -> router_ranger_64
    v3_mnist_letter_uc_soap_64.onnx  -> letter_uc_soap_64
    v3_mnist_letter_lc_adamw_28.onnx -> letter_lc_adamw_28
    v1_lion_64_64.onnx             -> lion_64   (v2 naming, kept for
                                                  backward compatibility)
    adahessian_64.onnx             -> adahessian_64  (older v1/v2 naming)
    anything_else.onnx             -> filename stem (no extension)
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    # v3_mnist_digit_{optimizer}_{res} pattern
    m = re.match(r"v3_mnist_digit_([a-z0-9]+)_(\d+)$", stem)
    if m:
        return f"digit_{m.group(1)}_{m.group(2)}"
    # v3_mnist_router_{optimizer}_{res} pattern
    m = re.match(r"v3_mnist_router_([a-z0-9]+)_(\d+)$", stem)
    if m:
        return f"router_{m.group(1)}_{m.group(2)}"
    # v3_mnist_letter_{uc,lc}_{optimizer}_{res} pattern
    m = re.match(r"v3_mnist_letter_(uc|lc)_([a-z0-9]+)_(\d+)$", stem)
    if m:
        return f"letter_{m.group(1)}_{m.group(2)}_{m.group(3)}"
    # v1_{optimizer}_{res}_{res} pattern (v2 naming)
    m = re.match(r"v1_([a-z0-9]+)_(\d+)_\d+$", stem)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    # {optimizer}_{res} pattern (v2/v1 naming, e.g. adahessian_64, soap_128)
    m = re.match(r"([a-z0-9]+)_(\d+)$", stem)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return stem


def normalize_char(char_gray, img_size: int):
    """Binarizes (Otsu) a grayscale character crop, tightly crops to its
    ink bounding box, centers it on a padded square canvas, dilates
    slightly, and resizes to img_size x img_size. Returns None if the
    crop has no ink (cv2.findNonZero finds nothing). Bridges a raw scan
    crop into the clean, centered image domain the digit/router models
    were trained on.
    """
    _, binary = cv2.threshold(char_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary[y:y+h, x:x+w]
    size = max(w, h)
    pad = int(size * 0.2)
    canvas = np.zeros((size + pad*2, size + pad*2), dtype=np.uint8)
    x_off = pad + (size - w) // 2
    y_off = pad + (size - h) // 2
    canvas[y_off:y_off+h, x_off:x_off+w] = cropped
    kernel_size = max(2, size // 20)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    canvas = cv2.dilate(canvas, kernel, iterations=2)
    return cv2.resize(canvas, (img_size, img_size), interpolation=cv2.INTER_AREA)


def predict_char_topn(session, char_gray, img_size: int, n: int = 3, labels=None):
    """Runs one ensemble-member ONNX session on one character crop (via
    normalize_char()) and returns its top-n (label, softmax probability)
    predictions, highest confidence first. Returns [("?", 0.0)] * n if
    normalize_char() finds no ink to classify. labels defaults to the
    digit ensemble's own LABELS (0-9); pass UC_LABELS/LC_LABELS to reuse
    this same function for the letter-identity ensembles instead — the
    only thing that differs between a digit and a letter model at
    inference time is which label list its output index maps to.
    """
    labels = LABELS if labels is None else labels
    normalized = normalize_char(char_gray, img_size)
    if normalized is None:
        return [("?", 0.0)] * n
    arr = normalized.astype(np.float32) / 255.0
    arr = arr.reshape(1, 1, img_size, img_size)
    logits = session.run(["logits"], {"image": arr})[0][0]
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    top_indices = np.argsort(probs)[::-1][:n]
    return [(labels[idx], float(probs[idx])) for idx in top_indices]


def predict_router(session, char_gray, img_size: int, unsure_floor: float = ROUTER_UNSURE_FLOOR):
    """
    Runs the router ONNX model (see v3_mnist_router_ranger_*.py) on one
    already-detected character box. Returns (verdict, top_label, prob):
      verdict   — "digit", "UC", "LC", or "unknown"
      top_label — the router's raw top class ("digit"/"UC"/"LC")
                  regardless of the floor, even when verdict ends up
                  "unknown" — useful for display/tuning
      prob      — the router's own top-class softmax probability

    Same normalize_char() preprocessing as predict_char_topn() above — the
    router was trained on the same clean/centered image domain the digit
    specialists were (see v3_mnist_router_ranger_*.py's get_transforms()),
    so it needs the same real-scan-to-training-domain bridge
    normalize_char() already provides for them. No separate preprocessing
    path for the router — this is a deliberate v3 requirement, not an
    oversight.

    Unlike the v2 gate's predict_gate() (a symmetric confidence BAND
    around a single P(digit) output), this is a single confidence FLOOR
    applied to the router's own top-class probability across 3 classes —
    the router's output head has no natural "0.5 midpoint" to be
    symmetric around, since it's not a binary decision.
    """
    normalized = normalize_char(char_gray, img_size)
    if normalized is None:
        return "unknown", "digit", 0.0
    arr = normalized.astype(np.float32) / 255.0
    arr = arr.reshape(1, 1, img_size, img_size)
    logits = session.run(["logits"], {"image": arr})[0][0]
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    top_idx = int(np.argmax(probs))
    top_label = ROUTER_LABELS[top_idx]
    top_conf = float(probs[top_idx])
    if top_conf < unsure_floor:
        return "unknown", top_label, top_conf
    return top_label, top_label, top_conf


# ── Box detection ─────────────────────────────────────────────────────────────

def merge_nearby_boxes(boxes, gap_x=15, gap_y=35):
    """Iteratively merges (x, y, w, h) boxes whose horizontal edges are
    within gap_x of each other AND whose vertical centers are within
    gap_y of each other into their bounding union, repeating until no
    more pairs qualify. Used to fuse fragments (e.g. a broken stroke)
    that contour detection split into separate boxes for one character.
    """
    if not boxes:
        return boxes
    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        new_boxes = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            ax, ay, aw, ah = boxes[i]
            acy = ay + ah // 2
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                bx, by, bw, bh = boxes[j]
                bcy = by + bh // 2
                x_close = ax < bx + bw + gap_x and bx < ax + aw + gap_x
                y_close = abs(acy - bcy) < gap_y
                if x_close and y_close:
                    nx = min(ax, bx)
                    ny = min(ay, by)
                    nw = max(ax + aw, bx + bw) - nx
                    nh = max(ay + ah, by + bh) - ny
                    boxes[i] = (nx, ny, nw, nh)
                    ax, ay, aw, ah = nx, ny, nw, nh
                    acy = ay + ah // 2
                    used[j] = True
                    merged = True
            new_boxes.append(boxes[i])
            used[i] = True
        boxes = new_boxes
    return boxes


def get_boxes(image_path):
    """Detect character bounding boxes and group them into lines.

    Returns (gray_image, lines), where lines is a list of rows and each row
    is a list of (x, y, w, h) boxes sorted left-to-right.

    This is proximity-based clustering, not a fixed grid: each contour that
    survives the height/width/aspect filter below becomes a candidate box,
    and boxes are grouped into a "line" purely by how close their vertical
    centers are to each other (line_thresh, computed from that line's own
    box heights). There is no assumption that digits are evenly spaced,
    uniformly sized, or aligned to rows/columns — real handwriting isn't.

    Letters (v3): this project's box detection was originally tuned only
    against digit test images (v1/v2). Before wiring in the router (which
    needs realistic letter boxes to reach it at all — see
    v3_mnist_router_ranger_*.py), the size windows and aspect-ratio bounds
    below were checked against realistic letter shapes rather than assumed
    fine:
      - Ascender/descender letters (b, d, h, k, l; g, j, p, q, y): their
        bounding-box HEIGHT is comparable to or taller than a digit's in
        the same handwriting (digits already span roughly cap-height to
        baseline) — SIZE_WINDOWS' height percentages needed no change.
      - Narrow letters (i, l — and to a lesser extent digit "1", which the
        original windows already had to admit): a single thin pen stroke
        has a roughly fixed pixel width regardless of the character's
        design height, so its w/h aspect ratio can run lower than a
        typical digit's. The original floor (aspect > 0.1) risked
        rejecting a genuinely narrow single-stroke letter as noise.
        MIN_ASPECT_RATIO below was lowered from 0.1 to 0.06 to admit this
        case — still well above what a pure noise speck or scan artifact
        would realistically produce, based on this project's own
        observed digit boxes.
      - The split-boxes pass (_split_oversized_boxes below, for touching/
        merged characters) operates on generic ink-column geometry with
        no digit-specific assumption — checked and left unchanged.
      - MIN_ABS_HEIGHT_PX/MIN_ABS_WIDTH_PX and the width-ceiling rescue
        pass: unaffected by the letter case specifically (both were
        already about very small vs. very wide boxes, not narrow ones).
    This is a documented judgment call based on reasoning about letter
    shapes, not a real-world letter-image validation run — see
    v3_CHANGELOG.md's "what was and wasn't verified" note for this entry.

    Multi-scale detection: the same contour set is filtered through three
    overlapping height/width acceptance windows (small/medium/large,
    see SIZE_WINDOWS below) rather than one fixed range, so a single image
    containing both small, cramped digits and large ones doesn't force a
    single compromise threshold that's wrong for one end of that range. A
    contour is accepted if it fits ANY window; results are deduplicated
    (contours from overlapping windows collapse to one box, not three)
    before merging and line-clustering proceed exactly as before. This
    still isn't true multi-resolution image-pyramid detection (the contours
    themselves are extracted once, at one image scale) — it's the size
    filter that now runs at multiple scales, not the underlying edge/
    contour detection itself. A digit whose true size falls entirely
    outside all three windows is still rejected, same as before, just with
    a much wider combined range than the original single window covered.

    Rescue pass: a contour that fails ONLY the width ceiling of the window
    it would otherwise fit (correct height and aspect ratio, just too wide
    — the usual cause is a stray connecting stroke or underline fusing onto
    an otherwise normal digit) is not discarded outright. Each line is
    checked for an unusually large gap relative to its own typical spacing
    — between two existing characters, or after the last one, or before the
    first one (a trailing/leading dropped digit, like "1 8 4" missing a
    final "5", produces no gap between existing boxes at all, so the
    last/first-box edges have to be checked too, not just the gaps strictly
    between pairs). If an oversized contour's position and height fit
    inside that gap, it's added back as a real character instead of
    silently dropped. This does not touch contours rejected for height,
    aspect ratio, or being more than 2x over a window's width ceiling
    (those stay rejected — they're far more likely to be scan noise or a
    genuine full-width artifact than a real digit).

    On the underlying cause (a stray stroke fusing onto a digit's contour,
    inflating its bounding box): this is a known failure mode in OCR
    preprocessing, usually called "underline/rule-line removal." A
    standard OpenCV/computer-vision preprocessing technique for it (see
    OpenCV's documentation on morphological operations) detects long thin
    strokes with a wide horizontal morphological opening
    (cv2.getStructuringElement with a kernel like (25, 1) applied via
    cv2.MORPH_OPEN) and erases them before contour extraction, rather than
    rescuing an oversized box after the fact. That approach was reasoned
    through against this project's own images, not actually tested (no
    test images/dataset were available in this environment — see
    v3_CHANGELOG.md's Verification note), and NOT adopted here: as an
    illustrative estimate rather than a measured value, a stray stroke on
    a real "5" in this dataset would run only ~45px long (~8% of the
    working image width) — a kernel wide enough to reliably reject
    full-width scan artifacts elsewhere in the same image (which run
    ~560px, near the full width) would be too wide to catch that short a
    stroke, and a kernel narrow enough to catch it would start eroding
    real digit strokes elsewhere. The width-ceiling rescue approach
    implemented below sidesteps that tuning problem entirely by working in
    box-space after contour detection rather than pixel-space before it,
    at the cost of only fixing width-based rejections specifically (see
    the remaining failure modes below for what it still misses). This is
    a documented judgment call based on reasoning about typical
    stroke/artifact proportions, not a real-world image-measurement
    validation run — see v3_CHANGELOG.md's "what was and wasn't verified"
    note for this entry.

    Remaining known failure modes (still neither raise an error nor flag
    anything — the character in question is just absent from the returned
    lines):
      - A character rejected for HEIGHT or ASPECT RATIO in every size
        window (not width) is never reconsidered by the rescue pass above.
      - A character smaller or larger than all three SIZE_WINDOWS combined
        is still rejected outright — multi-scale widened the covered
        range, it didn't remove the range's edges entirely.
      - A line with only one detected character has no internal gap to
        measure, so the rescue pass has nothing to compare against and is
        skipped for that line entirely.
      - A character whose vertical center falls outside the current line's
        line_thresh window (e.g. it's written higher/lower than the rest of
        that row — plausible for a lowercase letter with no ascender next
        to an uppercase or digit-heavy line) can still be split into its
        own line or merged into the wrong one.
    Callers should still not assume the number of boxes returned equals the
    number of characters actually on the page — the rescue pass narrows one
    specific gap, it doesn't close all of them.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cv2.imread returned None: {image_path}")
    h_img, w_img = img.shape[:2]
    scale = min(1000 / w_img, 1000 / h_img, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w_img * scale), int(h_img * scale)))
        h_img, w_img = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Multi-scale size windows — run the same contour set through several
    # height/width acceptance ranges instead of one fixed range, so a single
    # image containing both small, cramped digits and large ones (a big
    # "100" next to a tiny "5", for example) doesn't force a single
    # compromise threshold that's wrong for one end of that range.
    # Windows overlap deliberately (SMALL's ceiling extends past MEDIUM's
    # floor, etc.) so a digit sitting near a boundary isn't missed by
    # landing in a gap between two non-overlapping ranges.
    SIZE_WINDOWS = [
        ("small",  0.015, 0.12, 0.005, 0.10),   # h_min%, h_max%, w_min%, w_max%
        ("medium", 0.03,  0.35, 0.01,  0.25),   # original single-pass range
        ("large",  0.20,  0.80, 0.08,  0.55),   # ceiling raised from 0.60 to 0.80 —
                                                  # reasoning-based estimate, not a
                                                  # measured case (see v3_CHANGELOG.md's
                                                  # Verification note): on a very short
                                                  # hand-cropped line image, a normal
                                                  # single digit could plausibly occupy
                                                  # 70%+ of the image height (e.g. an
                                                  # illustrative h=108 on a 156px-tall
                                                  # crop), which the old 60% ceiling
                                                  # would silently reject with no other
                                                  # window covering it either
    ]

    # Aspect-ratio (w/h) bounds — see get_boxes()'s own docstring's
    # "Letters (v3)" section for why MIN_ASPECT_RATIO was lowered from the
    # v1/v2 value (0.1) to admit narrow single-stroke letters (i, l) that
    # digit-only tuning didn't need to consider. MAX_ASPECT_RATIO (wide
    # merged/touching-character blobs, or scan artifacts) is unrelated to
    # the letter case and unchanged.
    MIN_ASPECT_RATIO = 0.06
    MAX_ASPECT_RATIO = 10.0

    # Absolute pixel floor, independent of the percentage windows above.
    # On a very short/wide cropped image (e.g. a single hand-cropped line
    # only ~140px tall after the 1000px-max scale-down), a percentage-based
    # floor can compute to just a few pixels — small enough that a stray
    # scan artifact or noise speck passes the "small" window's height check
    # even though no real character stroke is anywhere near that short. This
    # floor rejects anything under MIN_ABS_HEIGHT_PX regardless of what
    # percentage of image height that represents.
    MIN_ABS_HEIGHT_PX = 12
    MIN_ABS_WIDTH_PX  = 4

    def _touches_frame_edge(x, y, w, h, margin=2):
        # A contour whose bounding box starts at (or within `margin` px of)
        # the very top/bottom/left/right of the image is very likely a crop
        # or scan-boundary artifact, not a real character — genuine
        # characters in a hand-cropped line image are essentially never
        # drawn flush against the crop edge itself.
        return (y <= margin or (y + h) >= h_img - margin
                or x <= margin or (x + w) >= w_img - margin)

    def _dedup_key(box):
        # Round to a coarse grid so near-identical boxes from overlapping
        # windows collapse to the same key without needing exact pixel
        # equality (adjacent passes can differ by a pixel or two).
        x, y, w, h = box
        return (x // 5, y // 5, w // 5, h // 5)

    boxes = []
    wide_rejects = []  # contours rejected ONLY for exceeding the width ceiling —
                        # these are the ones worth a second look, since a digit
                        # with a stray connecting stroke (see rescue pass below)
                        # is the most common real-world cause, not noise.
    seen_box_keys = set()
    seen_reject_keys = set()
    for window_name, h_min_pct, h_max_pct, w_min_pct, w_max_pct in SIZE_WINDOWS:
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h < MIN_ABS_HEIGHT_PX or w < MIN_ABS_WIDTH_PX:
                continue
            if _touches_frame_edge(x, y, w, h):
                continue
            aspect = w / h if h > 0 else 0
            h_ok = h_img*h_min_pct < h < h_img*h_max_pct
            aspect_ok = MIN_ASPECT_RATIO < aspect < MAX_ASPECT_RATIO
            w_ok = w_img*w_min_pct < w < w_img*w_max_pct
            if h_ok and w_ok and aspect_ok:
                key = _dedup_key((x, y, w, h))
                if key not in seen_box_keys:
                    seen_box_keys.add(key)
                    boxes.append((x, y, w, h))
            elif h_ok and aspect_ok and w >= w_img*w_max_pct:
                # Failed on width alone within this window — height and
                # aspect both look like a real character. Cap how far over
                # the ceiling we'll ever consider (2x this window's ceiling)
                # so an actual full-width scan artifact is never rescued,
                # only a moderately oversized single character.
                if w < w_img*w_max_pct*2:
                    key = _dedup_key((x, y, w, h))
                    if key not in seen_reject_keys:
                        seen_reject_keys.add(key)
                        wide_rejects.append((x, y, w, h))

    if not boxes:
        return gray, []

    boxes = merge_nearby_boxes(boxes, gap_x=15, gap_y=35)
    boxes.sort(key=lambda b: b[1] + b[3] // 2)
    lines = []
    current_line = [boxes[0]]
    line_thresh = max(b[3] for b in boxes) * 0.5
    for box in boxes[1:]:
        cy_new = box[1] + box[3] // 2
        cy_ref = int(sum(b[1] + b[3] // 2 for b in current_line) / len(current_line))
        if abs(cy_new - cy_ref) < line_thresh:
            current_line.append(box)
        else:
            lines.append(sorted(current_line, key=lambda b: b[0]))
            current_line = [box]
    lines.append(sorted(current_line, key=lambda b: b[0]))

    # Rescue pass — a wide-reject contour whose position and height fit
    # plausibly into a line (right after the last box, right before the
    # first box, or in an unusually large gap between two boxes — relative
    # to that line's OWN normal spacing) is a strong sign a character was
    # dropped by the width filter above, not that the writer left a
    # deliberate blank space (deliberate spaces are handled separately
    # downstream, at print time, by comparing each gap against that
    # character's own width — this is a coarser, earlier check that runs
    # before any character has been classified).
    if len(lines) > 0 and wide_rejects:
        for line in lines:
            avg_h = sum(b[3] for b in line) / len(line)
            cy_ref = sum(b[1] + b[3] // 2 for b in line) / len(line)
            # Typical spacing for this line: median gap between existing
            # boxes if there are at least two, otherwise fall back to a
            # spacing estimate based on character height (roughly how wide
            # a MNIST-style digit tends to be relative to its own height).
            if len(line) >= 2:
                gaps = [line[i+1][0] - (line[i][0] + line[i][2]) for i in range(len(line) - 1)]
                median_gap = sorted(gaps)[len(gaps) // 2]
            else:
                median_gap = avg_h * 0.3
            gap_floor = max(median_gap * 3, avg_h * 0.8)

            def _try_rescue(zone_left, zone_right):
                for wr in wide_rejects:
                    wx, wy, ww, wh = wr
                    wcy = wy + wh // 2
                    if (zone_left - 10 <= wx and wx + ww <= zone_right + 10
                            and abs(wcy - cy_ref) < line_thresh
                            and abs(wh - avg_h) < avg_h):
                        return wr
                return None

            # Gaps strictly between two existing boxes.
            if len(line) >= 2:
                gaps = [line[i+1][0] - (line[i][0] + line[i][2]) for i in range(len(line) - 1)]
                for i, gap in enumerate(gaps):
                    if gap > gap_floor:
                        found = _try_rescue(line[i][0] + line[i][2], line[i+1][0])
                        if found:
                            line.append(found)
                            wide_rejects.remove(found)

            # After the last box in the line — the case a purely
            # between-boxes check can never catch (e.g. a dropped trailing
            # character, as when a line ends "1 8 4" but should be "1 8 4 5").
            last = max(line, key=lambda b: b[0])
            found = _try_rescue(last[0] + last[2], last[0] + last[2] + gap_floor + avg_h)
            if found:
                line.append(found)
                wide_rejects.remove(found)

            # Before the first box in the line, symmetric with the above.
            first = min(line, key=lambda b: b[0])
            found = _try_rescue(max(0, first[0] - gap_floor - avg_h), first[0])
            if found:
                line.append(found)
                wide_rejects.remove(found)
        lines = [sorted(line, key=lambda b: b[0]) for line in lines]

    # Split pass — a box whose aspect ratio is far wider than what a single
    # character in its own line normally looks like is almost certainly
    # several touching/closely-spaced characters that the dilate+contour
    # step fused into one connected region (a tightly packed run like
    # "9623701" never gets a gap between characters wide enough for
    # cv2.findContours to see them as separate blobs in the first place —
    # this is not something the rescue pass above can fix, since there was
    # never a "rejected" contour for a missing character to rescue; the
    # whole run was one contour from the start).
    #
    # Fix: for any box that's an outlier vs. its own line's typical aspect
    # ratio, look at the vertical ink-column projection (sum of foreground
    # pixels in each column) inside that box's crop of the thresholded
    # image. Real gaps between separate characters show up as runs of
    # columns with ~zero ink — cutting at the midpoint of each such run
    # splits the merged box into one sub-box per character. This only fires
    # on boxes that are already anomalous for their line; a normal
    # single-character box is left untouched. Generic ink-column geometry —
    # no digit-specific assumption, checked and unchanged for letters.
    def _split_oversized_boxes(line, thresh_img):
        if len(line) < 1:
            return line
        aspects = [w / h for (x, y, w, h) in line if h > 0]
        if not aspects:
            return line
        # ABSOLUTE_SPLIT_CEILING is a hard, context-independent floor for
        # the split threshold: no isolated real character is ever this much
        # wider than it is tall, so any box at or above it is treated as a
        # split candidate regardless of what else is in its line. This
        # matters specifically for lines with only one box (or several
        # boxes that are all already oversized) — the median of a single
        # wide box is that box's own aspect ratio, which can never exceed
        # a threshold derived purely from itself, so a line-relative-only
        # rule would silently never fire in exactly the cases (single
        # merged blob per line) this pass exists to catch.
        ABSOLUTE_SPLIT_CEILING = 2.2
        if len(aspects) >= 3:
            # With enough boxes for a trustworthy median, let a genuinely
            # narrow-character line raise the bar higher than the absolute
            # floor (so a line of consistently wide characters isn't
            # flagged just for being wide) — but never lower than the
            # absolute floor, since that would risk splitting normal
            # single characters in a line that happens to run narrow.
            #
            # The multiplier is capped (not a flat median*2.5) because an
            # uncapped multiplier breaks down exactly when a line is mostly
            # ALREADY multi-character groups (e.g. "36 42 380 670..." where
            # most "characters" are actually 2-3 digit numbers) — the
            # median itself is inflated by the merged boxes this pass is
            # supposed to catch, so median*2.5 raises the floor high enough
            # that nothing in the line ever looks anomalous relative to its
            # already-merged neighbors. Capping the floor keeps real
            # multi-character merges catchable even when they're the
            # majority of the line, while still letting a handful of
            # genuinely wide single characters raise the bar somewhat.
            median_aspect = sorted(aspects)[len(aspects) // 2]
            SPLIT_ASPECT_FLOOR = min(
                max(ABSOLUTE_SPLIT_CEILING, median_aspect * 2.5),
                ABSOLUTE_SPLIT_CEILING * 1.5,
            )
        else:
            SPLIT_ASPECT_FLOOR = ABSOLUTE_SPLIT_CEILING
        new_line = []
        for (x, y, w, h) in line:
            aspect = w / h if h > 0 else 0
            if aspect < SPLIT_ASPECT_FLOOR or w < 60:
                new_line.append((x, y, w, h))
                continue

            crop = thresh_img[y:y+h, x:x+w]
            col_ink = crop.sum(axis=0) / 255.0  # ink pixels per column

            # Find runs of near-zero-ink columns — candidate gaps.
            ink_floor = max(1.0, col_ink.max() * 0.04)
            is_gap = col_ink <= ink_floor
            gap_runs = []
            run_start = None
            for i, g in enumerate(is_gap):
                if g and run_start is None:
                    run_start = i
                elif not g and run_start is not None:
                    gap_runs.append((run_start, i))
                    run_start = None
            if run_start is not None:
                gap_runs.append((run_start, len(is_gap)))

            # Only trust gaps wide enough to be a real inter-character
            # space, not a thin stroke crossing (e.g. inside a "4" or "8").
            # A minimum gap width scaled to box height avoids splitting on
            # noise while still catching genuine character boundaries.
            min_gap_px = max(3, int(h * 0.15))
            split_points = [
                (a + b) // 2 for (a, b) in gap_runs
                if (b - a) >= min_gap_px and 0 < a and b < w
            ]

            if not split_points:
                # No true zero-ink gap — try a softer fallback: a local
                # minimum in ink density can still mark a real boundary
                # between two lightly-touching characters, even though it
                # never drops to zero. This only fires when the width
                # strongly implies more than one character is present
                # (aspect still over the split floor) and a genuine local
                # dip exists — it does not touch boxes that already found a
                # true gap above, and it still respects min_gap_px so a
                # single character's internal stroke crossings (the middle
                # of an "8" or "4") aren't mistaken for a boundary.
                window = max(3, min_gap_px)
                candidates = []
                for i in range(window, w - window):
                    seg = col_ink[i - window:i + window + 1]
                    center = col_ink[i]
                    # A local minimum: this column is the lowest point in
                    # its neighborhood AND meaningfully lower than the
                    # segment's own average, so a flat, uniformly-inked
                    # stretch (the middle of a single wide character)
                    # doesn't get treated as a dip just for being average.
                    if center == seg.min() and center < seg.mean() * 0.6:
                        candidates.append(i)
                # Collapse adjacent candidate columns into single points
                # (a real dip is usually more than one column wide).
                fallback_points = []
                for c in candidates:
                    if not fallback_points or c - fallback_points[-1] > window:
                        fallback_points.append(c)
                # Only accept fallback splits that would produce believable
                # character-width pieces — reject if any resulting piece
                # would be narrower than the box's own height * 0.15 (too
                # thin to be a real character) since a bad fallback split
                # is worse than leaving the box merged.
                if fallback_points:
                    cuts_try = [0] + fallback_points + [w]
                    piece_widths = [cuts_try[i+1] - cuts_try[i] for i in range(len(cuts_try) - 1)]
                    if all(pw >= max(6, int(h * 0.15)) for pw in piece_widths):
                        split_points = fallback_points

            if not split_points:
                # Neither the true-gap search nor the ink-density fallback
                # found a boundary — this is the genuinely hard case: two
                # characters whose strokes actually touch, with no column
                # anywhere that's meaningfully lower-ink than its neighbors
                # (a pure column-darkness measure can't see this, since the
                # touching point can be just as dark as the strokes on
                # either side of it). Distance transform measures something
                # different: at each foreground pixel, how far is it to the
                # nearest background pixel. Where two round/thick strokes
                # merely touch, that "waist" is geometrically thinner than
                # the bodies of either character even if it's not lighter
                # in raw ink count — distance-transform-based separation of
                # touching blobs is a standard technique in computer vision
                # (see OpenCV's documentation on distanceTransform() and
                # watershed segmentation), applied here as a 1D column
                # profile since these are roughly side-by-side characters,
                # not arbitrarily-shaped overlapping regions.
                dist = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
                col_dist = dist.max(axis=0)
                if col_dist.max() > 0:
                    dist_window = max(3, min_gap_px)
                    dist_candidates = []
                    for i in range(dist_window, w - dist_window):
                        seg = col_dist[i - dist_window:i + dist_window + 1]
                        center = col_dist[i]
                        if center > 0 and center == seg.min() and center < seg.mean() * 0.7:
                            dist_candidates.append(i)
                    dist_points = []
                    for c in dist_candidates:
                        if not dist_points or c - dist_points[-1] > dist_window:
                            dist_points.append(c)
                    if dist_points:
                        cuts_try = [0] + dist_points + [w]
                        piece_widths = [cuts_try[i+1] - cuts_try[i] for i in range(len(cuts_try) - 1)]
                        if all(pw >= max(6, int(h * 0.15)) for pw in piece_widths):
                            split_points = dist_points

            if not split_points:
                new_line.append((x, y, w, h))
                continue

            # Cut the box at each split point into sub-boxes, using each
            # sub-crop's own tight ink bounding box (not the full sub-slice)
            # so the split boxes hug each character rather than inheriting
            # the full original box's height/padding.
            cuts = [0] + split_points + [w]
            sub_boxes = []
            for i in range(len(cuts) - 1):
                sx0, sx1 = cuts[i], cuts[i+1]
                if sx1 - sx0 < min_gap_px:
                    continue
                sub_crop = crop[:, sx0:sx1]
                coords = cv2.findNonZero(sub_crop)
                if coords is None:
                    continue
                sxx, syy, sww, shh = cv2.boundingRect(coords)
                sub_boxes.append((x + sx0 + sxx, y + syy, sww, shh))

            if len(sub_boxes) >= 2:
                new_line.extend(sub_boxes)
            else:
                # Splitting didn't actually produce multiple usable
                # sub-boxes (e.g. ink is one continuous unbroken stroke) —
                # keep the original box rather than discard real content.
                new_line.append((x, y, w, h))
        return sorted(new_line, key=lambda b: b[0])

    lines = [_split_oversized_boxes(line, thresh) for line in lines]

    return gray, lines


# ── Voting ────────────────────────────────────────────────────────────────────

def vote_topn(all_top3, conf_threshold=0.20):
    """Majority vote across models for one already-detected character box.

    Returns (label, agreement, top1_list).
    label is a digit '0'-'9', or '??' if the models genuinely split with no
    majority, no weighted winner, and no single dominant-confidence model —
    i.e. this character WAS detected and classified by every model, but the
    ensemble could not settle on one answer. agreement is 'ALL', 'MAJORITY',
    'WEIGHTED', or 'SPLIT' (SPLIT always pairs with the '??' label).
    """
    top1_labels = [t[0][0] for t in all_top3 if t[0][1] >= conf_threshold]
    if not top1_labels:
        top1_labels = [t[0][0] for t in all_top3]

    count = Counter(top1_labels)
    if not count:
        return "??", "SPLIT", top1_labels

    top_label, top_count = count.most_common(1)[0]
    n = len(all_top3)

    if top_count == n:
        return top_label, "ALL", top1_labels
    elif top_count > 1:
        return top_label, "MAJORITY", top1_labels

    # Weighted confidence tiebreak
    scores = {}
    for top3 in all_top3:
        for rank, (lbl, conf) in enumerate(top3):
            weight = conf * (1.0 / (rank + 1))
            scores[lbl] = scores.get(lbl, 0.0) + weight

    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1 and sorted_scores[0] > sorted_scores[1] * 1.3:
        return max(scores, key=scores.get), "WEIGHTED", top1_labels

    # Single-model dominant confidence rescue
    all_top1_confs = [(t[0][0], t[0][1]) for t in all_top3]
    best_conf_label, best_conf = max(all_top1_confs, key=lambda x: x[1])
    if best_conf > 0.22:
        other_confs = [c for l, c in all_top1_confs if l != best_conf_label]
        if not other_confs or best_conf > max(other_confs) * 1.2:
            return best_conf_label, "WEIGHTED", top1_labels

    return "??", "SPLIT", top1_labels


# ── Path resolution ───────────────────────────────────────────────────────────

def resolve_image_paths(args):
    """Expands each CLI image argument (glob pattern or literal path) into
    a sorted list of existing file paths, then filters that list down to
    SUPPORTED_EXTS. Prints a [warn] line and skips anything that doesn't
    resolve to a file or has an unsupported extension.
    """
    if not args:
        return []
    paths = []
    for arg in args:
        expanded = glob.glob(arg)
        if expanded:
            paths.extend(sorted(expanded))
        elif os.path.isfile(arg):
            paths.append(arg)
        else:
            print(f"  [warn] Not found, skipping: {arg}")
    filtered = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext in SUPPORTED_EXTS:
            filtered.append(p)
        else:
            print(f"  [warn] Unsupported extension '{ext}', skipping: {p}")
    return filtered


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(image_path, sessions, img_sizes, model_names, ground_truth=None,
                  router_session=None, router_img_size=None,
                  router_unsure_floor=ROUTER_UNSURE_FLOOR,
                  uc_sessions=None, uc_img_sizes=None, uc_model_names=None,
                  lc_sessions=None, lc_img_sizes=None, lc_model_names=None):
    """
    image_path: path to a single image file to run the pipeline on (the
    __main__ entry point loops over multiple resolved images, calling
    this once per image).

    sessions: list of loaded onnxruntime InferenceSession objects — the
    digit ensemble models (one per --models/--model-dir entry that loaded
    successfully). len(sessions) decides whether this run prints as
    "Single Model" or an "N-Model Ensemble".

    img_sizes: list of int, parallel to sessions — each model's expected
    square input resolution (e.g. 64 for a 64x64 model), read from the
    ONNX file itself via get_model_input_size().

    model_names: list of str, parallel to sessions — each model's short
    display name (see short_model_name()), used in the INDIVIDUAL MODEL
    PREDICTIONS, PER-MODEL SUMMARY, and PER-MODEL ACCURACY sections.

    ground_truth (optional): known-correct answer string to score the
    final read against, character by character (digits only, no spaces,
    e.g. "5038" — see --ground-truth). When provided, prints the
    PER-MODEL ACCURACY and ACCURACY sections; when None (the default),
    both sections are skipped entirely.

    router_session (optional): a loaded router ONNX session (see
    v3_mnist_router_ranger_*.py / predict_router() above). If provided,
    every detected box is classified digit/UC/LC before the digit
    ensemble runs. Only a "digit" verdict (confidence >= router_unsure_floor)
    proceeds to the full ensemble; a verdict below router_unsure_floor is
    rendered [UNK] (no ensemble run). UC/LC verdicts are handed to the
    matching letter-identity ensemble below, if one was loaded — otherwise
    they're output directly as terminal labels, exactly as before letter
    models existed. Passing None (the default) reproduces the pipeline's
    original behavior exactly, with zero router involvement — every box
    goes straight to the digit ensemble.

    uc_sessions/uc_img_sizes/uc_model_names, lc_sessions/lc_img_sizes/
    lc_model_names (all optional): loaded uppercase/lowercase letter-
    identity ONNX sessions (see v3_mnist_letter_{uc,lc}_*.py), parallel
    lists in the same shape as sessions/img_sizes/model_names above. A
    router-classified UC (or LC) box is handed to the matching ensemble
    here — single-model shortcut or vote_topn(), same as the digit
    ensemble — if that list is non-empty; if empty/None (the default),
    that box keeps the original bare-'UC'/'LC'-label behavior. Only takes
    effect together with router_session.
    """
    n_models = len(sessions)
    ensemble_label = f"{n_models}-Model Ensemble" if n_models > 1 else "Single Model"

    print(f"\n{'='*60}")
    print(f"  MNIST OCR Pipeline — {ensemble_label}")
    print(f"  Image: {os.path.basename(image_path)}")
    print(f"{'='*60}")

    try:
        gray, lines = get_boxes(image_path)
    except ValueError as e:
        print(f"  ERROR: {e}")
        return

    total_chars = sum(len(l) for l in lines)
    print(f"  Detected: {total_chars} characters across {len(lines)} line(s)")
    print(f"  (Detection is proximity-based, not a grid — an unusually large,")
    print(f"   small, or disconnected character may not produce a box at all and")
    print(f"   would be missing from this count without any warning.)\n")

    if not lines:
        print("  No characters detected.\n")
        return

    # Run inference
    all_model_lines = [[] for _ in range(n_models)]
    ensemble_lines  = []
    router_counts = {"digit": 0, "UC": 0, "LC": 0, "unknown": 0}

    for line in lines:
        per_model_line = [[] for _ in range(n_models)]
        ensemble_line  = []
        prev_x_end = None

        for (x, y, w, h) in line:
            space = prev_x_end is not None and (x - prev_x_end) > w * 1.2

            router_verdict, router_top_label, router_prob = None, None, None
            if router_session is not None:
                router_verdict, router_top_label, router_prob = predict_router(
                    router_session, gray[y:y+h, x:x+w], router_img_size, router_unsure_floor,
                )
                router_counts[router_verdict] += 1

            letter_detail = None
            if router_verdict == "unknown":
                # Router couldn't classify this box with enough confidence
                # to commit to digit/UC/LC — skip the (expensive, 10-class)
                # digit ensemble entirely for this box. Placeholder
                # per-model values match predict_char_topn()'s own "?"
                # convention for a None/unavailable session, so downstream
                # per-model line/summary rendering doesn't need a special case.
                winner    = "UNK"
                agreement = "ROUTER-UNKNOWN"
                raw_top1s = ["?"] * n_models
                all_top3  = [[("?", 0.0), ("?", 0.0), ("?", 0.0)] for _ in range(n_models)]
            elif router_verdict in ("UC", "LC"):
                # Confident letter classification — hand off to the
                # matching letter-identity ensemble (see
                # v3_mnist_letter_{uc,lc}_*.py) if one was loaded via
                # --letter-uc-models/--letter-lc-models. raw_top1s/all_top3
                # here MUST stay digit-ensemble-shaped (length n_models) —
                # they feed the digit-indexed INDIVIDUAL MODEL PREDICTIONS/
                # PER-MODEL SUMMARY sections below — so the letter
                # ensemble's own per-model detail is carried separately in
                # letter_detail (used only by CHARACTER DETAIL) instead of
                # overwriting them. agreement deliberately stays the bare
                # "ROUTER-UC"/"ROUTER-LC" tag regardless of the letter
                # ensemble's vote outcome — every downstream section
                # (rendering, CHARACTER DETAIL flags, PER-MODEL SUMMARY,
                # and the --ground-truth digit-accuracy exclusion list)
                # branches on that exact string, and letter boxes must
                # keep being excluded from digit-ground-truth accuracy
                # scoring regardless of whether an ensemble backed them.
                if router_verdict == "UC":
                    letter_sessions, letter_img_sizes, letter_model_names, letter_labels = (
                        uc_sessions, uc_img_sizes, uc_model_names, UC_LABELS)
                else:
                    letter_sessions, letter_img_sizes, letter_model_names, letter_labels = (
                        lc_sessions, lc_img_sizes, lc_model_names, LC_LABELS)

                if letter_sessions:
                    letter_top3 = [
                        predict_char_topn(s, gray[y:y+h, x:x+w], letter_img_sizes[i], n=3, labels=letter_labels)
                        for i, s in enumerate(letter_sessions)
                    ]
                    if len(letter_sessions) == 1:
                        winner = letter_top3[0][0][0]
                    else:
                        winner, _, _ = vote_topn(letter_top3)
                    letter_detail = (letter_model_names, letter_top3)
                else:
                    # No letter model loaded for this case — fall back to
                    # the bare terminal label (original, router-only
                    # behavior, before any letter-identity model existed).
                    winner = router_verdict

                agreement = f"ROUTER-{router_verdict}"
                raw_top1s = ["?"] * n_models
                all_top3  = [[("?", 0.0), ("?", 0.0), ("?", 0.0)] for _ in range(n_models)]
            else:
                # router_verdict is "digit", or router_session is None
                # entirely (original router-free behavior) — proceed to
                # the full digit ensemble exactly as before.
                all_top3 = []
                for i, session in enumerate(sessions):
                    if session is None:
                        all_top3.append([("?", 0.0), ("?", 0.0), ("?", 0.0)])
                    else:
                        top3 = predict_char_topn(session, gray[y:y+h, x:x+w], img_sizes[i], n=3)
                        all_top3.append(top3)

                raw_top1s = [t[0][0] if t[0][1] >= 0.20 else "?" for t in all_top3]

                # Non-digit detection — if best top-1 confidence across all models
                # is below the floor, flag as likely non-digit regardless of vote.
                # This runs the SAME as before regardless of the router's own
                # "digit" call — the router only decides whether the ensemble
                # runs at all, not how it scores its own confidence once it does.
                best_conf_any = max(top3[0][1] for top3 in all_top3)
                non_digit = best_conf_any < NON_DIGIT_CONF_FLOOR

                if non_digit:
                    winner    = "NON-DIGIT?"
                    agreement = "NON-DIGIT"
                elif n_models == 1:
                    winner    = all_top3[0][0][0]
                    agreement = "SINGLE"
                else:
                    winner, agreement, _ = vote_topn(all_top3)

            for i in range(n_models):
                if space:
                    per_model_line[i].append(" ")
                per_model_line[i].append(raw_top1s[i])

            if space:
                ensemble_line.append((" ", "ALL", [" "] * n_models, [], None, None, None))
            ensemble_line.append((winner, agreement, raw_top1s, all_top3, router_verdict, router_prob, letter_detail))
            prev_x_end = x + w

        for i in range(n_models):
            all_model_lines[i].append("".join(per_model_line[i]))
        ensemble_lines.append(ensemble_line)

    if router_session is not None:
        print(f"  Router: {router_counts['digit']} digit (-> ensemble), "
              f"{router_counts['UC']} UC, {router_counts['LC']} LC, "
              f"{router_counts['unknown']} unknown (rendered [UNK])\n")

    # Individual model output (only shown if >1 model)
    if n_models > 1:
        print(f"  {'─'*50}")
        print(f"  INDIVIDUAL MODEL PREDICTIONS")
        print(f"  {'─'*50}")
        for i, name in enumerate(model_names):
            print(f"  {name}  [{img_sizes[i]}x{img_sizes[i]}]:")
            for ln, text in enumerate(all_model_lines[i], 1):
                print(f"    Line {ln}: {text}")
        print()

    # Result
    print(f"  {'─'*50}")
    if n_models > 1:
        print(f"  ENSEMBLE RESULT  (plain=all agree  [x]=majority/weighted  "
              f"??=models split, no answer  [NON-DIGIT?]=likely not a digit  "
              f"UC/LC=router letter call  [UNK]=router unclassifiable)")
    else:
        print(f"  RESULT")
    print(f"  {'─'*50}")

    agree_count = 0
    letter_count = 0
    total_count = 0
    unknown_count = 0
    for ln, line in enumerate(ensemble_lines, 1):
        text = ""
        for entry in line:
            label, agreement = entry[0], entry[1]
            if label == " ":
                text += " "
                continue
            total_count += 1
            if agreement == "NON-DIGIT":
                text += "[NON-DIGIT?]"
                unknown_count += 1
            elif agreement == "ROUTER-UNKNOWN":
                text += "[UNK]"
                unknown_count += 1
            elif agreement in ("ROUTER-UC", "ROUTER-LC"):
                text += label  # bare "UC"/"LC" — a confident classification,
                                # not an ensemble tally, so no brackets
                letter_count += 1
            elif agreement in ("ALL", "SINGLE"):
                text += label
                agree_count += 1
            elif agreement in ("MAJORITY", "WEIGHTED"):
                text += f"[{label}]"
            else:
                text += "??"
                unknown_count += 1
        print(f"  Line {ln}: {text}")

    if n_models > 1:
        pct = 100 * agree_count / total_count if total_count else 0
        print(f"\n  Consensus: {agree_count}/{total_count} chars ({pct:.1f}% full agreement)")
    if letter_count:
        print(f"  Letters: {letter_count} character(s) classified UC/LC by the router "
              f"— see CHARACTER DETAIL below.")
    if unknown_count:
        print(f"  Unidentified: {unknown_count} character(s) marked ?? or [NON-DIGIT?] or [UNK] above "
              f"— see CHARACTER DETAIL below for per-model votes on each.")

    # Page layout — same characters as ENSEMBLE RESULT above, but positioned
    # left-to-right and top-to-bottom to roughly match their actual placement
    # on the page, instead of every line being flush-left with even spacing.
    img_h, img_w = gray.shape[:2]
    layout_width = 60  # character columns to map the page width onto
    print(f"\n  {'─'*50}")
    print(f"  PAGE LAYOUT  (approximate — horizontal position and line spacing")
    print(f"  scaled from the image; not a precise ruler)")
    print(f"  {'─'*50}")

    prev_line_bottom = None
    for line_boxes, ens_line in zip(lines, ensemble_lines):
        # Blank line(s) when the gap above this line is unusually large —
        # mirrors a paragraph break / skipped row on the physical page.
        line_top = min(b[1] for b in line_boxes)
        if prev_line_bottom is not None:
            gap = line_top - prev_line_bottom
            avg_h = sum(b[3] for b in line_boxes) / len(line_boxes)
            if gap > avg_h * 1.5:
                print()
        prev_line_bottom = max(b[1] + b[3] for b in line_boxes)

        # Build the visible characters for this line (skip the synthetic
        # space entries already inserted for in-line gaps — those are about
        # spacing within a line, not the line's position on the page).
        chars = [(e[0], e[1]) for e in ens_line if e[0] != " "]
        left_x = min(b[0] for b in line_boxes)
        indent = int((left_x / img_w) * layout_width)

        rendered = ""
        for label, agreement in chars:
            if agreement == "NON-DIGIT":
                rendered += "[NON-DIGIT?]"
            elif agreement == "ROUTER-UNKNOWN":
                rendered += "[UNK]"
            elif agreement in ("MAJORITY", "WEIGHTED"):
                rendered += f"[{label}]"
            else:
                rendered += label  # digit labels (ALL/SINGLE) and bare "UC"/"LC"
        print(f"  {' ' * indent}{rendered}")

    # Character detail — full top-3 per model per character
    print(f"\n  {'─'*50}")
    print(f"  CHARACTER DETAIL — full top-3 per model")
    print(f"  {'─'*50}")

    for ln, (line_boxes, ens_line) in enumerate(zip(lines, ensemble_lines), 1):
        print(f"\n  Line {ln}:")
        char_idx = 0
        for box_idx, (x, y, w, h) in enumerate(line_boxes):
            while char_idx < len(ens_line) and ens_line[char_idx][0] == " ":
                char_idx += 1
            if char_idx >= len(ens_line):
                break
            entry = ens_line[char_idx]
            final_label, agreement, raw_top1s, all_top3, router_verdict, router_prob, letter_detail = entry

            flag = ("✓ all"      if agreement == "ALL"            else
                    "single"     if agreement == "SINGLE"          else
                    "~ maj"      if agreement == "MAJORITY"        else
                    "~ wgt"      if agreement == "WEIGHTED"        else
                    "! NON-DIG"  if agreement == "NON-DIGIT"       else
                    "✗ router"   if agreement == "ROUTER-UNKNOWN"  else
                    "○ UC"       if agreement == "ROUTER-UC"       else
                    "○ LC"       if agreement == "ROUTER-LC"       else
                    "✗ split")

            aspect = w / h if h > 0 else 0
            nd_warn = ("  *** LIKELY NON-DIGIT — check image ***" if agreement == "NON-DIGIT" else
                       "  *** ROUTER FLAGGED UNKNOWN — ensemble skipped ***" if agreement == "ROUTER-UNKNOWN" else
                       "")
            router_note = (f"  router={router_top_label}({router_prob:.1%})"
                           if router_verdict is not None else "")
            print(f"\n    Char {box_idx+1:>2}  [x:{x} y:{y} w:{w} h:{h} asp:{aspect:.2f}]  vote={flag}  final={final_label}{router_note}{nd_warn}")
            if agreement == "ROUTER-UNKNOWN":
                print(f"      (digit ensemble skipped — router classified this box as unclassifiable)")
            elif agreement in ("ROUTER-UC", "ROUTER-LC") and letter_detail is None:
                _case = "uc" if agreement == "ROUTER-UC" else "lc"
                print(f"      (digit ensemble skipped — router classified this box as {final_label}; "
                      f"no --letter-{_case}-models loaded, so no letter-identity ensemble ran)")
            else:
                _model_names, _top3s = letter_detail if letter_detail is not None else (model_names, all_top3)
                for mi, (mname, top3) in enumerate(zip(_model_names, _top3s)):
                    top1_lbl, top1_conf = top3[0]
                    top2_lbl, top2_conf = top3[1] if len(top3) > 1 else ("?", 0.0)
                    top3_lbl, top3_conf = top3[2] if len(top3) > 2 else ("?", 0.0)
                    marker = " <-- SPLIT" if agreement == "SPLIT" and top1_lbl != final_label else ""
                    print(
                        f"      [{mi+1:>2}] {mname:<20s}  "
                        f"#1:{top1_lbl}({top1_conf:5.1%})  "
                        f"#2:{top2_lbl}({top2_conf:5.1%})  "
                        f"#3:{top3_lbl}({top3_conf:5.1%})"
                        f"{marker}"
                    )

            char_idx += 1

    # Per-model summary — individual read per model across all lines
    print(f"\n  {'─'*50}")
    print(f"  PER-MODEL SUMMARY")
    print(f"  {'─'*50}")
    for i, name in enumerate(model_names):
        read = "".join(
            " "           if entry[0] == " "               else
            "[ND?]"       if entry[1] == "NON-DIGIT"        else
            "[R?]"        if entry[1] == "ROUTER-UNKNOWN"   else
            "[UC]"        if entry[1] == "ROUTER-UC"        else
            "[LC]"        if entry[1] == "ROUTER-LC"        else
            entry[2][i]
            for line in ensemble_lines
            for entry in line
        )
        print(f"  [{i+1:>2}] {name:<20s}  read: {read}")

    if ground_truth is not None:
        gt = ground_truth.strip()
        print(f"\n  {'─'*50}")
        print(f"  PER-MODEL ACCURACY  (ground truth: {gt})")
        print(f"  {'─'*50}")
        _excluded = ("NON-DIGIT", "ROUTER-UNKNOWN", "ROUTER-UC", "ROUTER-LC")
        for i, name in enumerate(model_names):
            pred_chars = [
                entry[2][i]
                for line in ensemble_lines
                for entry in line
                if entry[0] != " " and entry[1] not in _excluded
            ]
            nd_count = sum(
                1 for line in ensemble_lines
                for entry in line
                if entry[1] == "NON-DIGIT"
            )
            router_excl_count = sum(
                1 for line in ensemble_lines
                for entry in line
                if entry[1] in ("ROUTER-UNKNOWN", "ROUTER-UC", "ROUTER-LC")
            )
            n_gt  = len(gt)
            correct = sum(1 for p, g in zip(pred_chars, gt) if p == g)
            pct = 100 * correct / n_gt if n_gt else 0.0
            nd_note = f"  [{nd_count} non-digit skipped]" if nd_count else ""
            router_note = f"  [{router_excl_count} router-excluded]" if router_excl_count else ""
            print(f"  [{i+1:>2}] {name:<20s}  {correct:>3}/{n_gt}  ({pct:5.1f}%)"
                  f"  pred: {''.join(pred_chars)}{nd_note}{router_note}")

    # Accuracy scoring
    if ground_truth is not None:
        gt = ground_truth.strip()
        _excluded = ("NON-DIGIT", "ROUTER-UNKNOWN", "ROUTER-UC", "ROUTER-LC")
        final_chars = [
            entry[0]
            for line in ensemble_lines
            for entry in line
            if entry[0] != " " and entry[1] not in _excluded
        ]
        nd_total = sum(
            1 for line in ensemble_lines
            for entry in line
            if entry[1] == "NON-DIGIT"
        )
        router_unk_total = sum(
            1 for line in ensemble_lines
            for entry in line
            if entry[1] == "ROUTER-UNKNOWN"
        )
        router_letter_total = sum(
            1 for line in ensemble_lines
            for entry in line
            if entry[1] in ("ROUTER-UC", "ROUTER-LC")
        )
        n_pred = len(final_chars)
        n_gt   = len(gt)
        correct = sum(1 for p, g in zip(final_chars, gt) if p == g)
        pct_acc = 100 * correct / n_gt if n_gt else 0.0

        print(f"  {'─'*50}")
        print(f"  ACCURACY  (ground truth: {gt})")
        print(f"  {'─'*50}")
        print(f"  Predicted chars : {n_pred}")
        print(f"  Ground truth    : {n_gt}")
        print(f"  Correct         : {correct} / {n_gt}  ({pct_acc:.1f}%)")
        if nd_total:
            print(f"  [warn] {nd_total} character(s) flagged as non-digit — excluded from accuracy count")
        if router_unk_total:
            print(f"  [warn] {router_unk_total} character(s) flagged unknown by the router "
                  f"— excluded from accuracy count")
        if router_letter_total:
            print(f"  [warn] {router_letter_total} character(s) classified as letters (UC/LC) by the "
                  f"router — excluded from digit accuracy count")
        if n_pred != n_gt:
            print(f"  [warn] Predicted {n_pred} digit chars but ground truth has {n_gt} — count mismatch")

    print(f"\n{'='*60}\n")


# ── Model loading ─────────────────────────────────────────────────────────────

def _is_router_or_letter_onnx(fname: str) -> bool:
    """True if fname is a router or letter-identity ONNX file — used by
    --model-dir's scan to keep those out of the digit ensemble (they need
    their own --router-model / --letter-uc-* / --letter-lc-* flags
    instead, same as --router-model already worked before this existed).
    """
    lower = fname.lower()
    return lower.startswith("v3_mnist_router_") or lower.startswith("v3_mnist_letter_")


def _load_onnx_model_list(model_paths, cuda_available, kind_label):
    """
    Loads a list of ONNX model paths into InferenceSessions, CUDA-first
    with automatic per-model CPU fallback (including the same "CUDA
    accepted but actually running on CPU" diagnostic, printed once per
    call). Originally inline in __main__ for the digit ensemble only —
    extracted here, byte-identical logic, so the new --letter-uc-models/
    --letter-lc-models loading can share it instead of a third
    near-identical copy. kind_label is used only in printed messages
    (e.g. "digit ensemble", "uppercase letter"). Returns (sessions,
    img_sizes, model_names, active_providers) — parallel lists, one entry
    per successfully loaded model.
    """
    sessions, img_sizes, model_names, active_providers = [], [], [], []
    cuda_diagnostic_shown = False
    print(f"\nLoading {len(model_paths)} {kind_label} model(s)...")
    for path in model_paths:
        s = None
        used_provider = None
        if cuda_available:
            try:
                # Both providers passed together (not CUDA alone) so ONNX
                # Runtime can fall back to CPU per-operator within a single
                # session if some op isn't CUDA-supported, not just fail the
                # whole session load. get_providers() after creation reports
                # which provider is actually active for this session.
                #
                # IMPORTANT: ONNX Runtime can accept CUDAExecutionProvider
                # in the providers list WITHOUT raising an exception, then
                # silently fall back to CPU for actual execution — this
                # commonly happens on a CUDA/cuDNN version mismatch between
                # what onnxruntime-gpu expects and what's installed on the
                # system. That failure mode does NOT raise here, so the
                # except block below never fires for it — it has to be
                # detected by checking get_providers() after the fact, and
                # ORT's own explanation (if any) goes to native stderr, not
                # a Python exception, so it's captured separately below.
                import io, contextlib
                stderr_capture = io.StringIO()
                with contextlib.redirect_stderr(stderr_capture):
                    s = ort.InferenceSession(
                        path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                    )
                used_provider = "CUDA" if s.get_providers()[0] == "CUDAExecutionProvider" else "CPU"
                if used_provider == "CPU" and not cuda_diagnostic_shown:
                    captured = stderr_capture.getvalue().strip()
                    print(f"  [CUDA] Session accepted CUDAExecutionProvider without error, "
                          f"but get_providers() shows CPU is actually active for {os.path.basename(path)}.")
                    print(f"  [CUDA] This usually means a CUDA/cuDNN version mismatch between "
                          f"onnxruntime-gpu and what's installed on this system, not a missing GPU.")
                    if captured:
                        print(f"  [CUDA] ONNX Runtime's own stderr output during load:")
                        for line in captured.splitlines():
                            print(f"    {line}")
                    else:
                        print(f"  [CUDA] No stderr output was captured — check `pip show onnxruntime-gpu` "
                              f"for the exact CUDA/cuDNN versions it was built against, and compare against "
                              f"the versions actually installed (nvidia-smi shows driver/CUDA version; "
                              f"cuDNN version has to be checked separately).")
                    cuda_diagnostic_shown = True  # only print this full diagnostic once per call,
                                                    # not once per model, to avoid flooding the log
            except Exception as e:
                print(f"  [CUDA] Failed to load {os.path.basename(path)} on GPU "
                      f"({type(e).__name__}: {e}) — falling back to CPU-only session for this model.")
                s = None
        if s is None:
            try:
                s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                used_provider = "CPU"
            except Exception as e:
                print(f"  Failed: {os.path.basename(path)}: {e}")
                continue

        sessions.append(s)
        img_sizes.append(get_model_input_size(s))
        model_names.append(short_model_name(path))
        active_providers.append(used_provider)
        print(f"  Loaded: {short_model_name(path):20s}  [{img_sizes[-1]}x{img_sizes[-1]}]  "
              f"({used_provider})  ({os.path.basename(path)})")

    return sessions, img_sizes, model_names, active_providers


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tee = _Tee()

    parser = argparse.ArgumentParser(
        description="MNIST OCR Pipeline — single model, ensemble, or any combination, with an optional router pre-filter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  Single model:
    python ocr_pipeline_mnist.py --models v3_mnist_digit_soap_64/v3_mnist_digit_soap_64.onnx digit.jpg

  Two-model ensemble:
    python ocr_pipeline_mnist.py --models v3_mnist_digit_soap_64/v3_mnist_digit_soap_64.onnx v3_mnist_digit_adamw_64/v3_mnist_digit_adamw_64.onnx digit.jpg

  Glob model paths:
    python ocr_pipeline_mnist.py --models v3_mnist_digit_soap_64/*.onnx digit.jpg

  Scan a directory for all .onnx files (recursive):
    python ocr_pipeline_mnist.py --model-dir . digit.jpg

  Scan directory, multiple test images:
    python ocr_pipeline_mnist.py --model-dir . test*.jpg

  With the router pre-filter (see v3_mnist_router_ranger_*.py):
    python ocr_pipeline_mnist.py --router-model v3_mnist_router_ranger_64/v3_mnist_router_ranger_64.onnx --model-dir . digit.jpg

  With the router AND letter-identity ensembles (see v3_mnist_letter_{uc,lc}_*.py):
    python ocr_pipeline_mnist.py --router-model v3_mnist_router_ranger_64/v3_mnist_router_ranger_64.onnx \\
        --letter-uc-model-dir . --letter-lc-model-dir . --model-dir . page.jpg
        """
    )
    model_src = parser.add_mutually_exclusive_group(required=True)
    model_src.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="One or more ONNX model paths (space-separated, supports globs)"
    )
    model_src.add_argument(
        "--model-dir",
        default=None,
        metavar="DIR",
        help="Directory to scan recursively for all .onnx files"
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Image file(s) or glob patterns"
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        metavar="STRING",
        help=(
            "Known correct answer for accuracy scoring. "
            "Digits only, no spaces (e.g. --ground-truth 5038). "
            "Compared against final read character by character."
        )
    )
    parser.add_argument(
        "--non-digit-floor",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            f"Override NON_DIGIT_CONF_FLOOR (default {NON_DIGIT_CONF_FLOOR}). "
            "If every model's top confidence for a character falls below this "
            "value, it's flagged [NON-DIGIT?] instead of output as a low-"
            "confidence digit guess. Exposed here so this can be tuned across "
            "runs against real test images without editing the script."
        )
    )
    parser.add_argument(
        "--router-model",
        default=None,
        metavar="ONNX_PATH",
        help=(
            "Optional path to a trained router ONNX model (see "
            "v3_mnist_router_ranger_*.py). If provided, every detected box "
            "is classified digit/UC/LC by the router BEFORE the digit "
            "ensemble runs. Only a confident 'digit' verdict proceeds to "
            "the ensemble; UC/LC boxes are output directly as terminal "
            "labels (no letter-identity model exists yet); a verdict below "
            "--router-unsure-floor is rendered [UNK]. Omit this flag for "
            "the pipeline's original, router-free behavior (every box goes "
            "straight to the ensemble)."
        )
    )
    parser.add_argument(
        "--router-unsure-floor",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            f"Override ROUTER_UNSURE_FLOOR (default {ROUTER_UNSURE_FLOOR}). "
            "If the router's own top-class confidence for a box falls below "
            "this value, the box is rendered [UNK] instead of its top class "
            "(digit/UC/LC). Only takes effect together with --router-model."
        )
    )
    letter_uc_src = parser.add_mutually_exclusive_group()
    letter_uc_src.add_argument(
        "--letter-uc-models",
        nargs="+",
        default=None,
        help=(
            "One or more uppercase letter-identity ONNX model paths (see "
            "v3_mnist_letter_uc_*.py). Only used for boxes the router "
            "classifies as UC; omit to keep the original bare-'UC'-label "
            "behavior for those boxes. Only takes effect together with "
            "--router-model."
        )
    )
    letter_uc_src.add_argument(
        "--letter-uc-model-dir",
        default=None,
        metavar="DIR",
        help="Directory to scan recursively for v3_mnist_letter_uc_*.onnx files."
    )
    letter_lc_src = parser.add_mutually_exclusive_group()
    letter_lc_src.add_argument(
        "--letter-lc-models",
        nargs="+",
        default=None,
        help="Same as --letter-uc-models, for boxes the router classifies as LC."
    )
    letter_lc_src.add_argument(
        "--letter-lc-model-dir",
        default=None,
        metavar="DIR",
        help="Directory to scan recursively for v3_mnist_letter_lc_*.onnx files."
    )
    args = parser.parse_args()

    if args.non_digit_floor is not None:
        NON_DIGIT_CONF_FLOOR = args.non_digit_floor
        print(f"  [Config] NON_DIGIT_CONF_FLOOR overridden to {NON_DIGIT_CONF_FLOOR} via --non-digit-floor")

    router_unsure_floor = ROUTER_UNSURE_FLOOR
    if args.router_unsure_floor is not None:
        router_unsure_floor = args.router_unsure_floor
        print(f"  [Config] Router unsure floor overridden to {router_unsure_floor} via --router-unsure-floor")

    # Resolve model paths
    # Directory names to never descend into when scanning --model-dir.
    # Prevents sweeping up venv/site-packages test fixtures (onnx ships
    # hundreds of tiny unit-test .onnx files, many literally named
    # "model.onnx") and any unrelated model-zoo files that happen to live
    # under the project root.
    EXCLUDED_DIRS = {"venv", ".venv", "env", "site-packages", "__pycache__",
                     ".git", "node_modules"}
    model_paths = []
    if args.model_dir:
        if not os.path.isdir(args.model_dir):
            print(f"  ERROR: --model-dir not found: {args.model_dir}")
            sys.exit(1)
        skipped_dirs = []
        skipped_router_letter = []
        for root, dirs, files in os.walk(args.model_dir):
            before = set(dirs)
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
            skipped = before - set(dirs)
            if skipped:
                skipped_dirs.extend(os.path.join(root, d) for d in skipped)
            dirs.sort()
            for fname in sorted(files):
                if not fname.lower().endswith(".onnx"):
                    continue
                if _is_router_or_letter_onnx(fname):
                    skipped_router_letter.append(os.path.join(root, fname))
                    continue
                model_paths.append(os.path.join(root, fname))
        if skipped_dirs:
            print(f"  Skipped {len(skipped_dirs)} excluded folder(s) "
                  f"(venv/site-packages/etc.) during scan:")
            for d in skipped_dirs:
                print(f"    - {d}")
        if skipped_router_letter:
            print(f"  Skipped {len(skipped_router_letter)} router/letter-identity .onnx file(s) "
                  f"during --model-dir scan (not digit-ensemble models — use --router-model / "
                  f"--letter-uc-model(-dir) / --letter-lc-model(-dir) to load them):")
            for f in skipped_router_letter:
                print(f"    - {f}")
        if not model_paths:
            print(f"  ERROR: No .onnx files found in: {args.model_dir}")
            sys.exit(1)
        print(f"  Scanned {args.model_dir} — found {len(model_paths)} digit-ensemble .onnx file(s)")
    else:
        for m in args.models:
            expanded = glob.glob(m)
            if expanded:
                model_paths.extend(sorted(expanded))
            elif os.path.isfile(m):
                model_paths.append(m)
            else:
                print(f"  [warn] Model not found, skipping: {m}")

    if not model_paths:
        print("  ERROR: No valid model paths provided.")
        sys.exit(1)

    image_paths = resolve_image_paths(args.images)
    if not image_paths:
        parser.print_help()
        sys.exit(1)

    # Load models
    # Try CUDA first, fall back to CPU automatically if CUDA isn't available
    # (either onnxruntime-gpu isn't installed, or no CUDA device is present).
    # ONNX Runtime's InferenceSession already does this fallback internally
    # given a provider list — the code below just makes that choice visible
    # per model instead of silently succeeding either way.
    available_providers = ort.get_available_providers()
    cuda_available = "CUDAExecutionProvider" in available_providers
    print(f"\nONNX Runtime providers available: {available_providers}")
    if cuda_available:
        print("  CUDA available — will use GPU where possible, falling back to CPU per-model on failure.")
    else:
        print("  [Warning] CUDAExecutionProvider not available (onnxruntime-gpu not installed, "
              "or no compatible CUDA device found) — running on CPU only.")

    sessions, img_sizes, model_names, active_providers = _load_onnx_model_list(
        model_paths, cuda_available, "digit ensemble"
    )

    if not sessions:
        print("  ERROR: No models loaded successfully.")
        sys.exit(1)

    # Load the router model (optional) — same CUDA-first/CPU-fallback
    # pattern as the digit ensemble above, just a single session instead of
    # a list, since there's only ever one router model active at a time
    # (matching whatever resolution tier the pipeline is otherwise using),
    # not an ensemble of them.
    router_session  = None
    router_img_size = None
    if args.router_model:
        if not os.path.isfile(args.router_model):
            print(f"  ERROR: --router-model not found: {args.router_model}")
            sys.exit(1)
        router_provider = None
        try:
            if cuda_available:
                router_session = ort.InferenceSession(
                    args.router_model,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                router_provider = "CUDA" if router_session.get_providers()[0] == "CUDAExecutionProvider" else "CPU"
            else:
                router_session = ort.InferenceSession(args.router_model, providers=["CPUExecutionProvider"])
                router_provider = "CPU"
        except Exception as e:
            print(f"  ERROR: Failed to load --router-model {args.router_model}: {e}")
            sys.exit(1)
        router_img_size = get_model_input_size(router_session)
        print(f"  Loaded router model: {short_model_name(args.router_model):20s}  "
              f"[{router_img_size}x{router_img_size}]  ({router_provider})  "
              f"({os.path.basename(args.router_model)})")
        print(f"  Router unsure floor: {router_unsure_floor} (top-class confidence below this -> [UNK] — "
              f"see --router-unsure-floor)")

    # Load the letter-identity ensembles (optional) — same CUDA-first/
    # CPU-fallback loading as the digit ensemble above (via
    # _load_onnx_model_list()), just directory scans filtered to each
    # case's own filename prefix specifically, so --letter-uc-model-dir
    # pointed at the project root doesn't also sweep up LC/digit/router
    # files (same discipline as the --model-dir fix above). Only used for
    # boxes the router classifies UC/LC; omitting these flags keeps the
    # original bare-label behavior for those boxes (see run_pipeline()'s
    # own docstring).
    uc_model_paths = []
    if args.letter_uc_model_dir:
        if not os.path.isdir(args.letter_uc_model_dir):
            print(f"  ERROR: --letter-uc-model-dir not found: {args.letter_uc_model_dir}")
            sys.exit(1)
        for root, dirs, files in os.walk(args.letter_uc_model_dir):
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
            dirs.sort()
            for fname in sorted(files):
                if fname.lower().endswith(".onnx") and fname.lower().startswith("v3_mnist_letter_uc_"):
                    uc_model_paths.append(os.path.join(root, fname))
    elif args.letter_uc_models:
        for m in args.letter_uc_models:
            expanded = glob.glob(m)
            if expanded:
                uc_model_paths.extend(sorted(expanded))
            elif os.path.isfile(m):
                uc_model_paths.append(m)
            else:
                print(f"  [warn] --letter-uc-models path not found, skipping: {m}")

    uc_sessions, uc_img_sizes, uc_model_names = [], [], []
    if uc_model_paths:
        uc_sessions, uc_img_sizes, uc_model_names, _ = _load_onnx_model_list(
            uc_model_paths, cuda_available, "uppercase letter"
        )
        if not uc_sessions:
            print("  [warn] No uppercase letter models loaded successfully — "
                  "UC-routed boxes will fall back to the bare 'UC' label.")

    lc_model_paths = []
    if args.letter_lc_model_dir:
        if not os.path.isdir(args.letter_lc_model_dir):
            print(f"  ERROR: --letter-lc-model-dir not found: {args.letter_lc_model_dir}")
            sys.exit(1)
        for root, dirs, files in os.walk(args.letter_lc_model_dir):
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
            dirs.sort()
            for fname in sorted(files):
                if fname.lower().endswith(".onnx") and fname.lower().startswith("v3_mnist_letter_lc_"):
                    lc_model_paths.append(os.path.join(root, fname))
    elif args.letter_lc_models:
        for m in args.letter_lc_models:
            expanded = glob.glob(m)
            if expanded:
                lc_model_paths.extend(sorted(expanded))
            elif os.path.isfile(m):
                lc_model_paths.append(m)
            else:
                print(f"  [warn] --letter-lc-models path not found, skipping: {m}")

    lc_sessions, lc_img_sizes, lc_model_names = [], [], []
    if lc_model_paths:
        lc_sessions, lc_img_sizes, lc_model_names, _ = _load_onnx_model_list(
            lc_model_paths, cuda_available, "lowercase letter"
        )
        if not lc_sessions:
            print("  [warn] No lowercase letter models loaded successfully — "
                  "LC-routed boxes will fall back to the bare 'LC' label.")

    # Print model roster once — not repeated per image
    print(f"\n  {'─'*50}")
    print(f"  MODELS ({len(sessions)} loaded)")
    print(f"  {'─'*50}")
    for name, size, provider in zip(model_names, img_sizes, active_providers):
        print(f"  {name:20s}  [{size}x{size}]  ({provider})")
    n_cuda = sum(1 for p in active_providers if p == "CUDA")
    n_cpu  = len(active_providers) - n_cuda
    if n_cpu and n_cuda:
        print(f"  [Note] Mixed providers: {n_cuda} model(s) on CUDA, {n_cpu} on CPU "
              f"(per-model fallback — see load messages above for why).")
    print()

    print(f"Processing {len(image_paths)} image(s) with {len(sessions)} model(s)...\n")


    for image_path in image_paths:
        run_pipeline(image_path, sessions, img_sizes, model_names, ground_truth=args.ground_truth,
                     router_session=router_session, router_img_size=router_img_size,
                     router_unsure_floor=router_unsure_floor,
                     uc_sessions=uc_sessions, uc_img_sizes=uc_img_sizes, uc_model_names=uc_model_names,
                     lc_sessions=lc_sessions, lc_img_sizes=lc_img_sizes, lc_model_names=lc_model_names)

    tee.close()
