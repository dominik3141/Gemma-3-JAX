# XPlane Profiling Findings (Baseline)

This document records the baseline XPlane profiling results for the Gemma forward inference path. It summarizes memory traffic, arithmetic intensity, and per-op hotspots, and highlights a major measurement caveat discovered in the trace window.

## Source
- XPlane file: `/tmp/jax_trace_focus2/plugins/profile/2026_02_11_01_24_15/t1v-n-fc812640-w-0.xplane.pb`
- Parsed via `xprof` (installed from `tensorboard-plugin-profile`), using `xspace_to_tool_data()` for `roofline_model`, `hlo_stats`, and `op_profile`.

## High-level timing (op_profile)
- Trace window length: **0.94422481875 s**
- **IDLE time:** **0.92614802125 s (98.09%)**
- **Active `jit_forward_single`:** **0.01806712375 s (1.91%)**
- `jit__argmax`: **9.67375e-06 s (0.001%)**

**Key caveat:** The capture is dominated by idle time. This dilutes average bandwidth and FLOP rates and makes “overall” utilization look far lower than the active compute is actually achieving.

## Memory traffic + arithmetic intensity
From `roofline_model` (Program total row):
- Total time: **944,224.81875 us**
- Measured FLOP rate: **25.953 GFLOP/s**
- Measured memory BW: **11.585 GiB/s**
- Operational intensity (OI): **2.086 FLOP/Byte**
- Bound by: **HBM**

Derived from roofline totals:
- Total BF16 FLOPs: **~2.4506e10**
- Total bytes (from measured BW): **~1.1746e10 bytes (~10.94 GiB)**
- Derived OI: **~2.086 FLOP/Byte** (matches tool output)

From `op_profile` (by program):
- HBM bytes: **11,094,265,131**
- SRAM read bytes: **359,490,802**
- SRAM write bytes: **287,997,922**

## Bandwidth utilization
- **Whole trace (includes idle):** ~10.9–11.6 GiB/s HBM BW (very low due to 98% idle)
- **Active `jit_forward_single`:** ~572 GiB/s HBM BW (calculated from op_profile)
- HBM utilization for active forward (op_profile `bandwidthUtils[HBM]`): **~0.75**

## HLO stats (top kernels by time)
`hlo_stats` total time sum: **32,384.754 us (32.385 ms)**

Top HLO ops by total time:
1. `while.12` (while body): **13,859.035 us (42.79%)** — **Compute-bound**
2. `fusion.104`: **5,724.9825 us (17.68%)** — **HBM-bound**, OI ~2.25
3. `multiply_reduce_fusion`: **4,007.6775 us (12.38%)** — **HBM-bound**, OI ~2.50
4. `fusion.106`: **2,878.01125 us (8.89%)** — **HBM-bound**, OI ~2.50
5. `fusion.101`: **984.7775 us (3.04%)** — **HBM-bound**, OI ~1.25

Other notable ops:
- `cond.2.clone.7`: 835.965 us (2.58%) — HBM-bound
- `broadcast.136`: 441.164 us (1.36%) — OI 0 (pure bandwidth)
- `copy.26`: 313.204 us (0.97%) — VMEM write bound

## Summary interpretation
- The active compute is **memory-bound** (OI ~2, HBM-bound fusions).
- The profile window is **dominated by idle time**, which masks true utilization and makes roofline summary values misleading if interpreted as steady-state.
- The active forward loop **does hit high HBM bandwidth (~572 GiB/s)**, consistent with HBM-bound kernels.

## Next steps (profiling accuracy)
- Tighten profiling to **only the steady-state token loop** (exclude warmup/idle gaps).
- Use explicit `block_until_ready()` around the timing window.
- Re-emit xplane for the reduced window and re-check roofline + hlo_stats to get clean utilization metrics.

