# Gemma 3 single file implementation in JAX

An extremely simple implementation of the 27b version of Gemma 3.
All important files are in `core`, all other code is boring and maintained by AI.

## Core files

1. `gemma_forward_parralel.py`. Defines a very simple forward function, so far very optimized for prefill and not at all great for autoregressive sampling (no KV cache, no flash attention).
2. `supervised_train.py` Simple next token prediction training utilities using the forward pass.
3. `rl.py` GRPO training logic for radicating integers (WIP).
4. `gemma_forward_inference.py` Inference forward logic (`forward_single`, KV cache prefill).

## Entrypoints

1. `entrypoints/rl.py` RL training script.
2. `entrypoints/supervised.py` Supervised training script.
3. `entrypoints/gemma_forward_inference.py` Inference runnable/test script (`main`).
4. `entrypoints/gemma_forward.py` Forward-pass runnable/test script (`main`).

## Status

This is very much a work in progress, but the current code already samples not obviously stupid responses.

## TensorBoard Profile Viewing (UV)

If you downloaded a profile session into a folder like
`/Users/dffarr/Downloads/tb-logdir/plugins/profile/<session_id>`, run:

```bash
uv run --extra tensorboard python -m tensorboard.main --logdir /Users/dffarr/Downloads/tb-logdir --port 6006
```

Then open `http://localhost:6006/#profile`.

For large multihost TPU profiles, prefer the remote VM workflow documented in
`ops/tensorboard_vm_runbook.md` instead of running TensorBoard locally.

## ToDo

1. Flash attention[^1]
2. All other inference optimizations I can come up with
3. Commit the HLO dumps so we can easily spot bisect expected XLA performance issues relative to code changes
4. RL memory ownership at the train-loop boundary is unclear. `main` is not jitted (logging/checkpoint side effects), so JAX cannot infer host-side liveness across iterations. Missing donation may keep old and new parameter buffers alive simultaneously, which could result in 2x parameter-buffer footprint. (Similar to the KV cache problem we had earlier).

[^1]: The reason I haven't done this yet is because I want to implement it myself and JAX primitives are insufficient, so I will have to go one level lower and implement it in Pallas. But first I have to learn Pallas, so this might take a week or two.
