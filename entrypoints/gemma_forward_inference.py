"""Inference script entrypoint for single-token cached forward generation."""

import time

import jax.numpy as jnp

from core.gemma_forward import config
from core.gemma_forward_inference import forward_single
from utils.params_io_1b import DEFAULT_ORBAX_CHECKPOINT, load_params
from utils.tokenize_text import detokenize_ids, tokenize_text


def main() -> None:
    print("Loading weights from Orbax checkpoint...")
    params = load_params(DEFAULT_ORBAX_CHECKPOINT)
    print("Weights loaded.")

    prompt = "The capital of France is Paris. The capital of Germany is"
    tokens = tokenize_text(prompt)

    if tokens[0] != 2:
        tokens = [2] + tokens

    print(f"Prompt: '{prompt}'")
    print(f"Tokens: {tokens}")

    max_new_tokens = 1000
    kv_cache_len = 1024
    assert max_new_tokens + len(tokens) < kv_cache_len

    Ks_cached = jnp.zeros(
        (
            config.num_layers,
            kv_cache_len,
            config.num_key_value_heads,
            config.head_dim,
        ),
        dtype=jnp.bfloat16,
    )
    Vs_cached = jnp.zeros(
        (
            config.num_layers,
            kv_cache_len,
            config.num_key_value_heads,
            config.d_kvq,
        ),
        dtype=jnp.bfloat16,
    )

    print("Processing prompt (prefill)...")

    _warmup_token = jnp.array(tokens[0])
    _warmup_pos = 0
    _warmup_K = jnp.zeros_like(Ks_cached)
    _warmup_V = jnp.zeros_like(Vs_cached)
    _warmup_logits, _, _ = forward_single(
        _warmup_token, params, _warmup_pos, _warmup_K, _warmup_V
    )
    _warmup_logits.block_until_ready()

    logits = None
    prefill_start = time.perf_counter()
    for i, token in enumerate(tokens):
        token_id = jnp.array(token)
        logits, Ks_cached, Vs_cached = forward_single(
            token_id, params, i, Ks_cached, Vs_cached
        )
        logits.block_until_ready()
    prefill_elapsed = time.perf_counter() - prefill_start

    print("Prompt processed.")
    print("Generating:")

    generated_tokens: list[int] = []
    curr_pos = len(tokens)
    printed_trailing_newline = True

    generation_start = time.perf_counter()
    for _ in range(max_new_tokens):
        next_token = int(jnp.argmax(logits).item())
        generated_tokens.append(next_token)

        token_text = detokenize_ids([next_token])
        if token_text:
            print(token_text, end="", flush=True)
            printed_trailing_newline = token_text.endswith("\n")

        token_id = jnp.array(next_token)
        logits, Ks_cached, Vs_cached = forward_single(
            token_id, params, curr_pos, Ks_cached, Vs_cached
        )
        logits.block_until_ready()
        curr_pos += 1
    generation_elapsed = time.perf_counter() - generation_start

    if not printed_trailing_newline:
        print()

    final_text = detokenize_ids(tokens + generated_tokens)
    print(f"Final output: {final_text}")

    prefill_avg_s_per_token = prefill_elapsed / len(tokens) if tokens else 0.0
    generation_avg_s_per_token = (
        generation_elapsed / len(generated_tokens) if generated_tokens else 0.0
    )
    print(
        f"Prefill avg: {prefill_avg_s_per_token:.4f}s/token "
        f"({prefill_elapsed:.3f}s total for {len(tokens)} tokens)"
    )
    print(
        f"Generation avg: {generation_avg_s_per_token:.4f}s/token "
        f"({generation_elapsed:.3f}s total for {len(generated_tokens)} tokens)"
    )


if __name__ == "__main__":
    main()
