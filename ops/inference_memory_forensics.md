# Inference Memory Forensics (Gemma 27B, v5e)

This report explains why the spreadsheet estimate is far below observed HBM usage, using the profile artifacts from:

- `/Users/dffarr/Downloads/gemma_inference_2133340691`

The focus is inference (not RL training), with concrete allocation mapping to code.

## Scope and data source

- Profile files: `*.xplane.pb` and `*.trace.json.gz` from 16 workers (`w-0..w-15`), each with 4 TPU chips.
- Total chips observed: `64`.
- Primary parser: `xprof.convert._pywrap_profiler_plugin.xspace_to_tools_data(...)`.
- Main views used:
  - `memory_profile` (global timeline, peak bytes in use)
  - `memory_viewer` (module-level buffer attribution with names, sizes, source info)

## Executive summary

- No evidence of a true leak (no unbounded growth pattern during generation).
- Parameters are sharded across all 64 chips and are not the dominant term.
- KV cache is replicated per chip in this setup and is the dominant persistent state.
- Peak memory is much higher than persistent state because of large temporary/output buffers from scan/update/copy patterns.
- Prefill currently keeps one full KV cache per prompt in Python lists, then stacks them, creating avoidable high-water allocations.

## Corrected table (per chip)

Numbers below are measured from `memory_viewer` for `jit_run_scan(4925098873813018656)` program `137` (TPU:0), and are identical across workers/chips in this profile.

| Object | Bytes (per chip) | GiB (per chip) | Source / Mapping |
|---|---:|---:|---|
| Model params (persistent) | 874,698,752 | 0.815 | `groupName=Parameter`, `tfOpName=params[...]` |
| KV cache (persistent) | 2,893,021,184 | 2.694 | `groupName=Parameter`, `tfOpName=init_carry[2/3]` |
| Other carry state | 2,098,176 | 0.002 | `init_carry[0]`, positions |
| **Persistent subtotal** | 3,769,830,912 | 3.511 | equals `entryComputationParametersMib` |
| Temporary buffers at peak | 4,058,439,680 | 3.780 | `groupName=Temporary` |
| Output/live-out buffers at peak | 2,895,119,360 | 2.696 | `groupName=Output` |
| **Module peak heap (`run_scan`)** | 10,723,442,688 | 9.987 | `peakHeapMib` |
| **Allocator peak bytes in use (profile)** | 14,670,334,464 | 13.663 | `memory_profile` peak |
| HBM capacity | 16,909,303,808 | 15.748 | `memory_profile` capacity |

Important distinction:

- `9.99 GiB` is the peak for the `run_scan` module.
- `13.66 GiB` is whole-process allocator peak and includes additional allocations outside that single module window.

## Why the spreadsheet is off

### 1) KV formula omitted critical factors

The spreadsheet line uses only `2 * bf16 * dim_kv = 512 bytes/token`, but misses:

- number of layers (`62`)
- number of KV heads (`16`)
- both K and V caches (`2`)
- batch size (`4`)
- cache length actually allocated (`1424` here)

Actual per-chip KV bytes in this run:

```text
batch * layers * cache_len * kv_heads * head_dim * bf16_bytes * 2
= 4 * 62 * 1424 * 16 * 128 * 2 * 2
= 2,893,021,184 bytes
= 2.694 GiB per chip
```

### 2) Inference profile has no Adam / grads / ref-model state

Those are training terms. This profile is inference. So rows like Adam/gradients/ref model are not part of measured inference HBM here.

### 3) Per-chip param term assumes wrong parallel interpretation

Code uses a 1D model mesh over all devices (`Mesh(jax.devices(), axis_names=("model",))`), i.e. 64-way model sharding in this run. Parameter shards are small on each chip (for example `[62,21504,84]`, where `84 * 64 = 5376`).

## Allocation mapping to code (largest offenders)

### A) KV replication and update path

- `core/gemma_forward_inference.py:73` (`Ks_cached.at[pos].set(K_new)`)
- `core/gemma_forward_inference.py:74` (`Vs_cached.at[pos].set(V_new)`)
- Called inside layer scan at `core/gemma_forward_inference.py:114`.

Largest peak buffers in `run_scan` include:

- `param.8`, `param.9` (`init_carry[2/3]`): `2 x 1379.5 MiB` (persistent KV)
- `bitcast_dynamic-update-slice_fusion.4` (from dynamic update path): `1379.5 MiB` (output)
- `copy.78`, `custom-call.16`, `copy.168`: additional `~1379.5 MiB` class buffers

This is why peak exceeds persistent state.

### B) Prefill staging overhead (avoidable high-water)

In `entrypoints/gemma_forward_inference.py`:

- per-prompt cache allocation: lines `108-125`
- keep full per-prompt caches alive in lists: lines `137-138`
- stack full caches into batched cache: lines `144-145`

This creates large transient memory pressure before generation:

- a single `jnp.stack` over four KV tensors shows a `2759 MiB` peak module (`jit_concatenate(...)`).
- this is on top of already-retained per-prompt caches in Python lists.

## Smoking gun: the four huge non-persistent KV buffers

Inside `jit_run_scan`, the main non-persistent spike is four buffers, each `1379.5 MiB` (`1.347 GiB`):

| Buffer | Group | Size MiB | Role |
|---|---|---:|---|
| `custom-call.16` | Temporary | 1379.5 | while-body temporary KV tensor |
| `copy.78` | Temporary | 1379.5 | layout/while copy of full KV tensor |
| `bitcast_dynamic-update-slice_fusion.4` | Output | 1379.5 | full updated KV tensor from `dynamic_update_slice` path |
| `copy.168` | Output | 1379.5 | output copy of full KV tensor (layout variant) |

These four alone are:

- `5518 MiB` (`5.389 GiB`)
- `53.96%` of `run_scan` module peak (`10226.67 MiB`)

If you include the two persistent full KV carry buffers (`param.8`, `param.9`, each `1379.5 MiB`):

- `8277 MiB` (`8.083 GiB`)
- `80.94%` of `run_scan` module peak

So this cluster is the dominant memory structure in inference.

Why 4 non-persistent full-KV buffers show up:

1. There are two logical caches (`K` and `V`) in carry.
2. Update is functional (`.at[pos].set(...)`), so old/new overlap exists during update:
   - `core/gemma_forward_inference.py:73`
   - `core/gemma_forward_inference.py:74`
3. The nested scans lower to while loops with different physical layouts (`[4,62,...]` vs `[62,4,...]`), introducing copy/bitcast variants:
   - `core/gemma_forward_inference.py:114`
   - `entrypoints/gemma_forward_inference.py:181`

In short: these are not random allocations; they are the full-cache in-flight tensors created by scan carry + functional updates + layout conversions.

## Leak check

No strong evidence of a leak:

- allocator peak time occurs during prefill/`forward_single`, not late in generation;
- generation is a fixed-shape `lax.scan` (`entrypoints/gemma_forward_inference.py:181`), which generally reuses buffers;
- observed peak is explained by static carry + temporary/output duplicates and prefill staging behavior.

## Update: New profile after decode changes (`gemma_inference_1668461314`)

New artifacts analyzed:

- `/Users/dffarr/Downloads/gemma_inference_1668461314`

### What improved

`jit_run_scan` memory dropped substantially (per chip, averaged across workers):

| Metric | Old (`2133340691`) | New (`1668461314`) | Delta |
|---|---:|---:|---:|
| `run_scan` peak heap | 9.987 GiB | 4.596 GiB | **-5.391 GiB** |
| `run_scan` Temporary group | 3.780 GiB | 1.085 GiB | -2.694 GiB |
| `run_scan` Output group | 2.696 GiB | 0.000 GiB | -2.696 GiB |
| `run_scan` Parameter group | 3.511 GiB | 3.511 GiB | 0.000 GiB |

The prior four huge non-persistent KV-shaped buffers are gone in `run_scan`:

- old: `4 x 1.347 GiB` non-Parameter buffers (`custom-call/copy/bitcast/copy`) = `5.389 GiB`
- new: `0 GiB` non-Parameter buffers above `1300 MiB`

So the decode-side copy pressure was removed as expected.

### What did not improve

Allocator peak HBM is effectively unchanged:

- old: `13.663 GiB` (`peakBytesInUse`)
- new: `13.663 GiB` (`peakBytesInUse`)

Peak-time module overlap is still `jit_forward_single` (prefill), not `jit_run_scan`, in all workers.

For allocator `0` at peak, old and new are nearly identical:

- `34 x 344.875 MiB` active allocations of shape `bf16[62,1424,16,128]`
- total from those allocations: `11.451 GiB` (`83.8%` of `13.663 GiB`)
- plus stack reservation: `0.793 GiB`
- together: `12.244 GiB` (`89.6%` of peak bytes in use)

This means decode is no longer the dominant peak source; prefill staging/dispatch is.

## Sharding findings

### What is sharded well

Parameter sharding rules (`utils/params_io_27b.py:48-57`) do produce 64-way model shards for major weights in this run (local dim `84`).

### What is not sharded

KV cache in inference carry is effectively replicated per chip (full `[batch, layers, cache_len, kv_heads, head_dim]`-style local tensors are present in peak buffers).

This is the key persistent memory driver for inference.

## Practical fixes (ordered by impact vs complexity)

1. Remove per-prompt cache list retention + stack path in prefill.
   - Current high-water path is still `entrypoints/gemma_forward_inference.py:100-146`.
   - Allocate batched KV directly and avoid retaining per-prompt full caches in Python lists.

2. Move prefill token loop to a jitted scan over sequence (batched), not Python `for pos` calls.
   - Current per-token `forward_single` calls in `entrypoints/gemma_forward_inference.py:128-132` can keep many full-cache versions live in-flight.
   - A compiled prefill loop gives XLA a chance to alias and reuse KV buffers similarly to decode.

3. Reduce `kv_cache_len` to actual needed length for the request, not `1024 + max_new_tokens` unconditionally.
   - Current code: `entrypoints/gemma_forward_inference.py:94`.
   - For short prompts this yields large, unnecessary KV overallocation.

4. Revisit Q/K/V partition strategy if inference memory is priority.
   - Current layout favors model-parallel compute strategy but leaves KV replicated.
   - A different sharding plan could reduce per-chip KV and copy overhead, at communication tradeoff cost.

## Bottom line

The measured inference footprint is high mostly because:

- KV cache is large and replicated per chip,
- temporary/output copies around cache update/scan are large,
- prefill staging keeps multiple full caches alive and then stacks them.

This is not primarily a 27B-parameter storage problem on v5e; it is mostly an inference-state + execution-structure memory problem in the current implementation.
