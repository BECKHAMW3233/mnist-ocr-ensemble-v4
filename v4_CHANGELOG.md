# v4 Changelog

## 2026-08-13 — Renamed project from v3 to v4

**Change:** Renamed all 44 model training scripts from the `v3_mnist_*`
naming convention to `v4_mnist_*`, and updated every internal
self-reference and cross-file reference to those filenames throughout
the codebase (docstring titles, `Output:` lines, `OUTPUT_ROOT`
construction, `build_config()`/`prefix` variables, CLI-transcript
filenames, and every cross-reference in other scripts pointing at a
renamed file by name or wildcard pattern).

Files renamed (44): all scripts in `digit_models/`, `lowercase_models/`,
`uppercase_models/`, and `router_models/` — see the resulting directory
listing for the full set.

Files with internal-reference edits (no rename): `ocr_pipeline_mnist.py`
(including the functionally load-bearing `short_model_name()` regexes
and the letter-model directory-scan filename filters — without this
edit the pipeline would silently stop recognizing any v4 model file),
`run_all_training.ps1` (the `$scripts` array and its usage comment),
`supplementary_data.py`, `setup_packages.py`, `requirements.txt`,
`common/checkpointing.py`, `common/distributed.py`.

**Not changed:** references to `v3_CHANGELOG.md` and narrative text
describing what changed "in v3" relative to v2 (e.g. "New for v3 — no
v2 counterpart", "kept in the v3 roster") — these are historical
records of the v2→v3 transition and stay accurate as history regardless
of this v3→v4 rename, the same way v3's own docs reference v2. Every
one of the ~326 remaining `v3` occurrences in the codebase was read
individually and confirmed to fall into this category — none reference
a renamed filename.

**Why:** Per William's direct instruction — this project's own naming
convention bakes the version into filenames, and `E:\emnist_ensemble_v4`
is a separate copy of the v3 project that needs its own v4 naming
throughout, not a mix of v3 and v4. No external source — this is a
project-structure decision, not a technical/library correctness
question.

---

## 2026-08-13 — Fixed run_all_training.ps1's project-root path

**Change:** `run_all_training.ps1`'s `$root` variable changed from
`"E:\mnist_v3"` to `"E:\emnist_ensemble_v4"`.

**Why:** Per William's direct instruction, in response to a flagged
issue. `$root` is the path this script uses to locate and launch the 44
training scripts — left pointing at the old v3 directory, the script
would not find any of the renamed v4 scripts at their actual location.
No external source — a project-specific path correction.

---

## 2026-08-13 — Updated repo-name references from v3 to v4

**Change:** Three references to the old repo name
"mnist-ocr-ensemble-v3" updated to "mnist-ocr-ensemble-v4":
`setup_packages.py`'s module docstring (line 4) and printed setup
banner (line 189), and `requirements.txt`'s header comment (line 1).

**Why:** Per William's direct instruction, in response to a flagged
issue (repo name wasn't covered by the `v3_mnist_` filename-rename rule
and needed an explicit decision). No external source.

---

## 2026-08-13 — Updated generic "v3" project-description text in supplementary_data.py

**Change:** Two lines updated from "v3" to "v4": the module's own
docstring opening ("Shared supplementary dataset loader — v3." →
"— v4.") and a runtime error message ("Unsupported v3 resolution
tier" → "Unsupported v4 resolution tier").

**Why:** Per William's direct instruction, in response to a flagged
issue — these describe the project's own current version, not a
specific filename or historical v2→v3 event, so they track the v4
rename rather than staying as history. No external source.

---

## 2026-08-13 — Fixed incomplete module list in common/__init__.py docstring

**Change:** `common/__init__.py`'s docstring said its shared modules
are imported by "every v3 training script (the 15 digit models and the
5 router models)" — omitting the 24 uppercase/lowercase letter models,
which also import from `common/`. Updated to read "(the 15 digit
models, the 24 uppercase/lowercase letter models, and the 5 router
models)".

**Why:** Per William's direct instruction, after being flagged as a
pre-existing inaccuracy noticed while reviewing this file for the v3→v4
rename (unrelated to the rename itself — confirmed via
`grep`/read that every letter-model script does import `common.seeding`,
`common.telemetry`, etc.). No external source.

---

## 2026-08-13 — Created README.md

**Change:** Created `README.md` (didn't exist previously in this
directory) — project overview, current directory structure (verified
against the actual post-rename file listing on disk), usage pointers,
and requirements pointer.

**Why:** Per William's direct instruction, approving a minimal README
now rather than deferring to a later step. No external source —
content reflects this repo's own actual current state.

---

## 2026-08-13 — Added 6 new hardware telemetry fields to common/telemetry.py

**Change:** Extended `HardwareMonitor.sample()`/`epoch_summary()` in
`common/telemetry.py` and the two field-list constants in
`common/cli_logging.py` with: `vram_other_processes_gb` (derived),
`cpu_pct_per_core` (point-rows only, semicolon-delimited), `pcie_tx_kb_s`/
`pcie_rx_kb_s` (new second nvidia-smi XML call), `pcie_link_gen_current`/
`pcie_link_width_current`/`pcie_link_gen_max`/`pcie_link_width_max`, and
`gpu_temp_tlimit_c`. No call-site changes in any of the 44 training
scripts — every field flows through existing `hw`/`hw_raw`/`history`
plumbing, matching the pattern the existing shared-memory telemetry
entry established.

**Two items were reframed after investigation, not built as originally
asked:**
- The system-wide-VRAM request assumed `nvsmi_vram_used_gb` was
  process-scoped. Verified via NVIDIA Developer Forums ("Understanding
  memory.used of nvidia-smi": *"memory.used is the total memory used by
  all applications running on the GPU"*) and this machine's own
  `nvidia-smi --help-query-gpu` output (`memory.used` = *"Total memory
  allocated by active contexts"*) that it's already system-wide. Built
  `vram_other_processes_gb` (= `nvsmi_vram_used_gb` − `vram_reserved_gb`,
  floored at 0) instead — a derived field, not a new query.
- The `_precomputed_cache` disk-latency request (originally item 5) was
  dropped entirely, per William's instruction, after finding that
  `_precomputed_cache` reads happen once at dataset-load time in
  `supplementary_data.py`, before `HardwareMonitor.sample()` is ever
  called (`sample()` only runs inside the per-epoch training loop) — a
  `common/telemetry.py`-only field could never actually observe that
  read event, and the precise fix (timing the read directly in
  `supplementary_data.py`) was outside this batch's `common/`-only
  scope.

**GPU memory bandwidth (item 3):** confirmed via this machine's actual
`nvidia-smi --help-query-gpu` output (NVIDIA-SMI 610.88) that no real
bandwidth/throughput field exists in the `--query-gpu` CSV interface —
`utilization.memory` (`cuda_mem_util_pct` in this project) is a
time-occupancy percentage, not GB/s. True internal VRAM bandwidth
requires DCGM profiling metrics, not confirmed reliably available on
consumer GeForce/RTX cards. Added PCIe `tx_util`/`rx_util` instead, per
William's explicit instruction, as a labeled proxy — this measures PCIe
bus traffic (host↔GPU), not GPU-internal VRAM bandwidth, and is
documented as such in `_query_pcie_throughput()`'s own docstring so it
isn't misread as the metric originally asked for.

**PCIe link state (item 4):** `pcie.link.gen.current` is marked
deprecated on this machine's actual nvidia-smi in favor of
`pcie.link.gen.gpucurrent` — used the current name. Confirmed all four
fields return real values via a live query on this machine.

**Bonus:** `temperature.gpu.tlimit` (thermal margin to throttle, not the
raw temperature already logged) added — sits on the same nvidia-smi call
already being made. Checked for CPU package temp / GPU VRM temp per
William's instruction to surface anything trivially available from the
same source: `psutil.sensors_temperatures()` confirmed not implemented
on Windows (`giampaolo/psutil` GitHub issues); nvidia-smi has no VRM
field. Neither added.

**Why:** Per William's direct instruction, following a five-item
priority list from his own review of what `HW_TELEMETRY_FIELDS`/
`_POINT_FIELDS` currently capture. Every claim about what's available
was checked against current documentation/this machine's actual
nvidia-smi output rather than assumed, per CLAUDE.md's verification
standard — sources cited above and in the code's own comments/docstrings.
