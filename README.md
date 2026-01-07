# Gemma 3 single file implementation in JAX

An extremely simple implementation of the 1b version of Gemma 3.
All important files are in `core`, all other code is boring and maintained by AI.

## Core files

1. `gemma_forward.py`. Defines a very simple forward function, so far very optimized for prefill and not at all great for autoregressive sampling (no KV cache, no flash attention).
2. `supervised_train.py` Simple next token prediction training using the forward pass.

## Status

This is very much a work in progress, but the current code already samples not obviously stupid responses.

## ToDo

1. KV caching
2. Flash attention
3. All other inference optimizations I can come up with
