# New Baseline Findings (Prefill Scan + On-Device Generation)

This document summarizes the updated performance findings after making **both prefill and generation fully on-device**. It captures the new baseline, reruns the most promising optimizations on top of it, and records which changes still help (or no longer help) once Python overhead is removed.

## Measurement Setup

- Command: `uv run python -m core.gemma_forward_inference`
- Harness: `python /tmp/bench_gemma_infer.py 5`
- Environment: `EPATH_USE_TF=0`
- Metrics: prefill total, generation total, combined total (mean ± std)

All experiments below are **n=5** runs. Each optimization was committed with full details and then reverted, so the baseline remains consistent between trials.

## New Baseline

**Baseline = prefill scan + on-device generation**

- Prefill: **0.277 ± 0.006 s**
- Generation: **0.596 ± 0.003 s**
- Total: **0.873 ± 0.007 s**

This baseline removes per-token host synchronization from *both* prefill and generation and is now our reference point for future experiments.

## Rerun Results on the New Baseline

### 1) Batched attention (`attnHead` uses Qs @ Ksᵀ + mask + softmax)
- Prefill: **0.283 ± 0.017 s**
- Generation: **0.545 ± 0.016 s**
- Total: **0.828 ± 0.029 s**

**Delta vs baseline:**
- Total: **-0.045 s (-5.16%)**
- Generation: **-0.051 s (-8.56%)**
- Prefill: **+0.006 s (+2.17%)**

**Interpretation:** This is the strongest compute-side win on the new baseline. The attention path still benefits from a single batched compute formulation once host overhead is gone.

---

### 2) Remove redundant outer vmap in `group_attention_single`
- Prefill: **0.277 ± 0.005 s**
- Generation: **0.593 ± 0.003 s**
- Total: **0.870 ± 0.006 s**

**Delta vs baseline:**
- Total: **-0.003 s (-0.34%)**

**Interpretation:** Effect is within the noise envelope at this baseline; the extra vmap overhead is now negligible.

---

### 3) Fuse Q/K/V projection matmuls
- Prefill: **0.276 ± 0.005 s**
- Generation: **0.605 ± 0.014 s**
- Total: **0.881 ± 0.011 s**

**Delta vs baseline:**
- Total: **+0.008 s (+0.92%)**

**Interpretation:** QKV fusion *regresses* slightly at this baseline, likely due to increased memory pressure or reduced fusion opportunities elsewhere. This is a useful negative result.

---

### 4) Batched attention + outer vmap removal combined
- Prefill: **0.277 ± 0.013 s**
- Generation: **0.557 ± 0.018 s**
- Total: **0.833 ± 0.030 s**

**Delta vs baseline:**
- Total: **-0.040 s (-4.58%)**

**Interpretation:** The combined change helps but is slightly weaker than batched attention alone, likely due to interaction effects and increased variance.

## Key Takeaways

1. **The new baseline is ~0.87 s total**, so remaining wins are comparatively smaller and require careful measurement.
2. **Batched attention is the most consistent improvement**, still delivering a ~5% total speedup even when all Python overhead is removed.
3. **QKV fusion no longer helps** and slightly regresses at this baseline.
4. **Outer vmap removal alone is now noise-level**, suggesting that major overhead sources have shifted deeper into the attention math.

## Recommended Next Steps

- **Adopt batched attention as the next baseline** and re-evaluate smaller tweaks on top of it.
- **Investigate KV cache layout / memory access patterns**, since attention now dominates runtime and is likely memory-bandwidth sensitive.
- **Re-profile with XPlane** after adopting batched attention to confirm where the remaining time is concentrated and whether memory traffic patterns improved.

