# CLI Memory Attribution From XPlane Profiles

This document explains the exact CLI workflow used to extract per-chip memory attribution from TPU profile artifacts in:

- `/Users/dffarr/Downloads/gemma_inference_2133340691`

The goal is to attribute memory to semantic `jax.Array` buckets (for example `KV cache`, `model parameters`, `gradients`) without relying on TensorBoard UI.

## 1. Preconditions

This repo already defines the needed profiler stack in the `tensorboard` extra (`xprof`, `tensorboard-plugin-profile`).

Run commands with:

```bash
uv run --extra tensorboard python ...
```

## 2. Confirm inputs

```bash
PROFILE_DIR=/Users/dffarr/Downloads/gemma_inference_2133340691
ls -1 "$PROFILE_DIR" | rg '\.xplane\.pb$' | head
```

You should see one `*.xplane.pb` per worker (for example `...-w-0.xplane.pb` through `...-w-15.xplane.pb`).

## 3. Query available tools directly from `xplane.pb`

```bash
PROFILE_DIR=/Users/dffarr/Downloads/gemma_inference_2133340691
FIRST_XPLANE="$(ls "$PROFILE_DIR"/*.xplane.pb | head -n 1)"

uv run --extra tensorboard python - <<'PY'
import os
from xprof.convert import _pywrap_profiler_plugin as p

xplane = os.environ["FIRST_XPLANE"]
raw, ok = p.xspace_to_tools_data([xplane], "tool_names", {"use_saved_result": True})
print("ok:", ok)
print(raw.decode() if ok else raw.decode("utf-8", "ignore"))
PY
```

Key tool used for attribution:

- `memory_viewer`

Supporting tool:

- `memory_profile` (used to find module/program IDs)

## 4. Find the right HLO module/program id

`memory_viewer` needs:

- `module_name` (for example `jit_run_scan(4925098873813018656)`)
- `program_id` (for example `137`)

Extract them from `memory_profile`:

```bash
FIRST_XPLANE="$(ls /Users/dffarr/Downloads/gemma_inference_2133340691/*.xplane.pb | head -n 1)"

uv run --extra tensorboard python - <<'PY'
import json
import os
from xprof.convert import _pywrap_profiler_plugin as p

xplane = os.environ["FIRST_XPLANE"]
raw, ok = p.xspace_to_tools_data([xplane], "memory_profile", {"use_saved_result": True})
assert ok, "memory_profile failed"
obj = json.loads(raw)

run_scan = [
    m for m in obj.get("hloModules", [])
    if m.get("name", "").startswith("jit_run_scan(")
]
print("jit_run_scan entries:")
for m in run_scan:
    print(m)
PY
```

For this profile, `jit_run_scan(4925098873813018656)` appears on each TPU plane with IDs like `137/206/276/346`.

## 5. Pull structured memory attribution (`memory_viewer`)

Use one module/program pair and memory space `0` (HBM main space):

```bash
FIRST_XPLANE="$(ls /Users/dffarr/Downloads/gemma_inference_2133340691/*.xplane.pb | head -n 1)"

uv run --extra tensorboard python - <<'PY'
import json
import os
from xprof.convert import _pywrap_profiler_plugin as p

xplane = os.environ["FIRST_XPLANE"]
module_name = "jit_run_scan(4925098873813018656)"
program_id = "137"

raw, ok = p.xspace_to_tools_data(
    [xplane],
    "memory_viewer",
    {
        "module_name": module_name,
        "program_id": program_id,
        "memory_space": "0",
        "view_memory_allocation_timeline": False,
    },
)
assert ok, "memory_viewer failed"
obj = json.loads(raw)

print("peakHeapMib:", obj["peakHeapMib"])
print("entryComputationParametersMib:", obj.get("entryComputationParametersMib"))
print("maybeLiveOutMib:", obj.get("maybeLiveOutMib"))
print("totalBufferAllocationMib:", obj.get("totalBufferAllocationMib"))
print()
print("Top buffers at peak:")
for row in obj["maxHeapBySize"][:20]:
    print(
        row.get("logicalBufferSizeMib", 0.0),
        row.get("groupName", ""),
        row.get("instructionName", ""),
        row.get("tfOpName", ""),
    )
PY
```

Important fields used:

- `maxHeapBySize`: buffers that are live at peak, with per-buffer size and labels.
- `groupName`: coarse class (`Parameter`, `Temporary`, `Output`).
- `tfOpName`: source-level name (contains entries like `params[...]`, `init_carry[...]`).
- `entryComputationParametersMib`: total persistent entry buffers.
- `peakHeapMib`: full peak heap.

## 6. Exact classification rules used

From `memory_viewer["maxHeapBySize"]`, for rows where `groupName == "Parameter"`:

- `tfOpName in {"init_carry[2]", "init_carry[3]"}` -> `kv_cache`
- `tfOpName.startswith("params['")` -> `model_params`
- `tfOpName.startswith("init_carry[")` (other indices) -> `other_carry_state`
- else -> `other_parameter`

Gradient detection rule:

- Scan `label`, `instructionName`, `tfOpName`, `shapeString` with regex:
  - `\bgrad(ient)?s?\b|backward|optimizer|adam|momentum|rmsprop|weight_decay`
- If no hits, report gradients as absent in this profile.

## 7. End-to-end script across all workers

This is the script used to compute per-chip numbers across all `*.xplane.pb` files:

```bash
PROFILE_DIR=/Users/dffarr/Downloads/gemma_inference_2133340691

uv run --extra tensorboard python - <<'PY'
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from xprof.convert import _pywrap_profiler_plugin as p

PROFILE_DIR = Path("/Users/dffarr/Downloads/gemma_inference_2133340691")
MODULE_NAME = "jit_run_scan(4925098873813018656)"

grad_pat = re.compile(
    r"\bgrad(ient)?s?\b|backward|optimizer|adam|momentum|rmsprop|weight_decay",
    re.IGNORECASE,
)

def gib(mib: float) -> float:
    return mib / 1024.0

rows = []
for xplane in sorted(PROFILE_DIR.glob("*.xplane.pb")):
    # 1) Find program_id for run_scan on TPU:0 in this worker file.
    raw_memprof, ok = p.xspace_to_tools_data([str(xplane)], "memory_profile", {"use_saved_result": True})
    if not ok:
        continue
    memprof = json.loads(raw_memprof)
    run_scan_ids = [
        m["id"]
        for m in memprof.get("hloModules", [])
        if m.get("name", "").startswith("jit_run_scan(") and m.get("planeName") == "TPU:0"
    ]
    if not run_scan_ids:
        continue
    program_id = str(run_scan_ids[0])

    # 2) Pull memory_viewer attribution for memory space 0 (HBM).
    raw_mv, ok = p.xspace_to_tools_data(
        [str(xplane)],
        "memory_viewer",
        {
            "module_name": MODULE_NAME,
            "program_id": program_id,
            "memory_space": "0",
            "view_memory_allocation_timeline": False,
        },
    )
    if not ok:
        continue
    mv = json.loads(raw_mv)

    # 3) Group totals at peak.
    group_totals = defaultdict(float)
    for r in mv.get("maxHeapBySize", []):
        group_totals[r.get("groupName", "<empty>")] += float(r.get("logicalBufferSizeMib", 0.0))

    # 4) Parameter-only semantic split.
    kv_mib = 0.0
    model_params_mib = 0.0
    other_carry_mib = 0.0
    other_parameter_mib = 0.0

    for r in mv.get("maxHeapBySize", []):
        if r.get("groupName") != "Parameter":
            continue
        tf = str(r.get("tfOpName", ""))
        size = float(r.get("logicalBufferSizeMib", 0.0))
        if tf in {"init_carry[2]", "init_carry[3]"}:
            kv_mib += size
        elif tf.startswith("params['"):
            model_params_mib += size
        elif tf.startswith("init_carry["):
            other_carry_mib += size
        else:
            other_parameter_mib += size

    # 5) Gradient marker scan.
    grad_hits = 0
    for r in mv.get("maxHeapBySize", []):
        haystack = " ".join(
            str(r.get(k, "")) for k in ("label", "instructionName", "tfOpName", "shapeString")
        )
        if grad_pat.search(haystack):
            grad_hits += 1

    rows.append(
        {
            "host": xplane.name.replace(".xplane.pb", ""),
            "program_id": program_id,
            "peak_mib": float(mv.get("peakHeapMib", 0.0)),
            "entry_params_mib": float(mv.get("entryComputationParametersMib", 0.0)),
            "group_parameter_mib": group_totals.get("Parameter", 0.0),
            "group_temporary_mib": group_totals.get("Temporary", 0.0),
            "group_output_mib": group_totals.get("Output", 0.0),
            "kv_cache_mib": kv_mib,
            "model_params_mib": model_params_mib,
            "other_carry_state_mib": other_carry_mib,
            "other_parameter_mib": other_parameter_mib,
            "gradient_marker_hits": grad_hits,
        }
    )

print("workers analyzed:", len(rows))
print()
print(
    "host\tpeak_GiB\tentry_params_GiB\tkv_cache_GiB\tmodel_params_GiB\t"
    "temporary_GiB\toutput_GiB\tgradient_hits"
)
for r in rows:
    print(
        f"{r['host']}\t{gib(r['peak_mib']):.3f}\t{gib(r['entry_params_mib']):.3f}\t"
        f"{gib(r['kv_cache_mib']):.3f}\t{gib(r['model_params_mib']):.3f}\t"
        f"{gib(r['group_temporary_mib']):.3f}\t{gib(r['group_output_mib']):.3f}\t"
        f"{r['gradient_marker_hits']}"
    )

if rows:
    def col(name):
        return [r[name] for r in rows]

    print()
    print("Aggregate min/avg/max (GiB):")
    for key in (
        "peak_mib",
        "entry_params_mib",
        "kv_cache_mib",
        "model_params_mib",
        "group_temporary_mib",
        "group_output_mib",
    ):
        vals = col(key)
        print(
            f"{key}: {gib(min(vals)):.3f}/{gib(mean(vals)):.3f}/{gib(max(vals)):.3f}"
        )
    print("gradient_marker_hits:", min(col("gradient_marker_hits")), "/", max(col("gradient_marker_hits")))
PY
```

## 8. Sanity checks applied

For each worker/chip:

- `sum(group totals)` equals `peakHeapMib`
- `sum(Parameter rows)` approximately equals `entryComputationParametersMib`
- `kv_cache + model_params + other_carry + other_parameter` approximately equals `sum(Parameter rows)`

This catches parsing mistakes and confirms attribution consistency.

## 9. Notes and caveats

- `memory_space="0"` is the primary HBM allocation space for this run.
- `memory_space="1"` contains a much smaller footprint (tens of MiB here).
- This profile is inference-only, so gradient/optimizer buffers are not expected.
- If you profile training, the same parsing path works, but classification rules should be extended with your optimizer naming conventions.
