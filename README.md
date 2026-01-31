# Gemma 3 single file implementation in JAX

An extremely simple implementation of the 1b version of Gemma 3.
All important files are in `core`, all other code is boring and maintained by AI.

## Core files

1. `gemma_forward.py`. Defines a very simple forward function, so far very optimized for prefill and not at all great for autoregressive sampling (no KV cache, no flash attention).
2. `supervised_train.py` Simple next token prediction training using the forward pass.
3. `rl.py` GRPO to recreate Deep Seek R1 Zero on the simple task of radicating integers (WIP).
4. `gemma_forward_inference.py`. Uses KV caching to predict a single next token.

## Status

This is very much a work in progress, but the current code already samples not obviously stupid responses.

## ToDo

1. Flash attention[^1]
2. Muon instead of Adam
3. All other inference optimizations I can come up with
4. Commit the HLO dumps so we can easily spot bisect expected XLA performance issues relative to code changes

[^1]: The reason I haven't done this yet is because I want to implement it myself and JAX primitives are insufficient, so I will have to go one level lower and implement it in Pallas. But first I have to learn Pallas, so this might take a week or two.
