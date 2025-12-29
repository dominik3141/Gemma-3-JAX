# Gemma 3 single file implementation in JAX

An extremely simple implementation of the 1b version of Gemma 3.
The only important file here is `gemma_forward.py` which implements all interesting logic.
There is also a vibecoded `prompt.py` cli tool to sample using the forward function defined in `gemma_forward.py` (assuming the instruction tuned weights are in a specific format defined in `restack_weights.py`).

## Status

This is very much a work in progress, but the current code already samples not obviously stupid responses.

## ToDo

1. Differentiate properly between global and local attention layers, so far we only adjust the RoPE's theta to get a working model within the local context window. Things will break as soon as we go beyond 1024 tokens.
2. KV caching
3. Flash attention
4. Masking of padding tokens
5. All other inference optimizations I can come up with
