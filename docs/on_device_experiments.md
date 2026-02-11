# On-Device Inference Experiments (Post-Idle Baseline)

This report documents the performance experiments run after eliminating the largest source of idle time by moving autoregressive generation fully on-device. The goal is to establish a cleaner measurement baseline and then re-evaluate earlier optimizations without Python-driven overhead skewing the results.

All results in this document are from **n=5** runs and are reported as **mean ± std dev**. Timing was collected from `uv run python -m core.gemma_forward_inference` using a small harness that extracts the prefill/generation totals printed by the script.

## Measurement Protocol

- Command: `uv run python -m core.gemma_forward_inference`
- Harness: `python /tmp/bench_gemma_infer.py 5`
- Environment: `EPATH_USE_TF=0` (forces epath to avoid TensorFlow import; TF is incompatible with NumPy 2.x in this environment)
- Metrics:
  - **Prefill total**: time to process the prompt tokens
  - **Generation total**: time to generate `max_new_tokens` tokens
  - **Total**: prefill + generation

Each experiment is committed with a detailed message, then reverted so subsequent runs are always compared against the same reference baseline. This preserves the experiment record while keeping the working state consistent.

## Key Baselines

### Original Baseline (Python-driven generation)
- Prefill: **3.051 ± 0.031 s**
- Generation: **23.653 ± 0.209 s**
- Total: **26.704 ± 0.224 s**

### New Baseline (On-device generation)
Change: replace Python per-token loop with a single jitted `lax.scan` (`generate_scan_jit`) for the full decode; transfer tokens back only once for post-hoc decoding/printing.

- Prefill: **3.102 ± 0.063 s**
- Generation: **0.594 ± 0.003 s**
- Total: **3.696 ± 0.061 s**

This new baseline removes most host-driven idle time in generation and is the reference point for all experiments below.

## Experiments on the On-Device Baseline

### 1) Prefill Scan (jitted scan over prompt)
**Change**: Replace the Python per-token prefill loop with `prefill_scan_jit`, a jitted `lax.scan` over the prompt tokens.

- Prefill: **0.272 ± 0.007 s**
- Generation: **0.592 ± 0.008 s**
- Total: **0.865 ± 0.014 s**

**Effect vs on-device baseline**:
- Total: **-2.831 s (-76.60%)**
- Prefill: **-2.830 s (-91.23%)**
- Generation: **-0.002 s (-0.34%)**

**Interpretation**: Once generation is on-device, prefill is the remaining dominant host-driven component. Moving prefill into a scan collapses the remaining idle gap and yields the largest additional gain.

---

### 2) Fused Q/K/V Projection Matmul
**Change**: Concatenate K/V/Q projection weights and perform a single matmul; slice result into Ks/Vs/Qs. This reduces the number of separate matmul launches per block from three to one.

- Prefill: **3.030 ± 0.037 s**
- Generation: **0.601 ± 0.004 s**
- Total: **3.631 ± 0.037 s**

**Effect vs on-device baseline**:
- Total: **-0.065 s (-1.76%)**
- Prefill: **-0.072 s (-2.32%)**
- Generation: **+0.007 s (+1.18%)**

**Interpretation**: Small but consistent improvement, likely due to reduced matmul launch overhead and better fusion opportunities in the per-token block path.

---

### 3) Remove Redundant Outer vmap in `group_attention_single`
**Change**: Pass all query vectors directly into `attnHead` with a vector of positions, eliminating an unnecessary outer `vmap` over singleton query slices.

- Prefill: **3.025 ± 0.015 s**
- Generation: **0.592 ± 0.005 s**
- Total: **3.617 ± 0.019 s**

**Effect vs on-device baseline**:
- Total: **-0.079 s (-2.14%)**
- Prefill: **-0.077 s (-2.48%)**
- Generation: **-0.002 s (-0.34%)**

**Interpretation**: Removing the outer vmap layer reduces per-token overhead in attention, producing a small but repeatable win.

---

### 4) Batched Attention in `attnHead` (Qs @ Ks^T)
**Change**: Replace per-query `vmap` with a single batched score computation (`Qs @ Ks^T`), apply causal/local masks in one tensor, and compute a single softmax and `@ Vs`.

- Prefill: **3.039 ± 0.070 s**
- Generation: **0.540 ± 0.004 s**
- Total: **3.579 ± 0.070 s**

**Effect vs on-device baseline**:
- Total: **-0.117 s (-3.17%)**
- Prefill: **-0.063 s (-2.03%)**
- Generation: **-0.054 s (-9.09%)**

**Interpretation**: This is the strongest per-block compute improvement so far. The attention inner loop benefits from a single batched compute path once Python overhead is removed.

## Summary of Findings

1. **Eliminating Python-driven token-by-token generation** is the largest single improvement and yields a clean baseline for compute-focused optimizations.
2. **Prefill scan** is the next dominant win after on-device generation, cutting total time by another ~76% relative to the on-device baseline.
3. **Attention implementation details matter** once host overhead is gone; both vmap simplifications and a batched attention formulation show consistent gains.
4. **Fusing Q/K/V projections** provides a small improvement and is likely worthwhile when combined with other changes.

## Recommended Next Steps

- **Stack the strongest changes** on top of the on-device baseline:
  - Prefill scan + batched `attnHead` + QKV fusion
- **Re-profile with XPlane** after stacking to confirm that remaining time is dominated by expected compute and not hidden overhead.
- **Explore memory layout and cache access** (e.g., KV layout or slicing patterns) now that host idle time is minimized.

