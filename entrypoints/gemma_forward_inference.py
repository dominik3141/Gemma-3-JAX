"""Inference script entrypoint for single-token cached forward generation."""

import time

import jax
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

    max_new_tokens = 200
    kv_cache_len = 1024 + max_new_tokens
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

    def token_callback(token_id):
        token_val = int(token_id)
        token_text = detokenize_ids([token_val])
        if token_text:
            print(token_text, end="", flush=True)
        return jnp.array(token_val, dtype=jnp.int32)

    def scan_body(carry, _):
        logits, curr_pos, Ks, Vs = carry
        next_token = jnp.argmax(logits).astype(jnp.int32)

        # Print via callback
        jax.experimental.io_callback(
            token_callback,
            jax.ShapeDtypeStruct((), jnp.int32),
            next_token,
            ordered=False,
        )

        new_logits, new_Ks, new_Vs = forward_single(
            next_token, params, curr_pos, Ks, Vs
        )
        return (new_logits, curr_pos + 1, new_Ks, new_Vs), next_token

    def run_scan(init_carry):
        return jax.lax.scan(scan_body, init_carry, None, length=max_new_tokens)

    # Initial carry
    init_carry = (logits, curr_pos, Ks_cached, Vs_cached)

    # Measure compilation time for jitted scan
    compile_start = time.perf_counter()
    run_scan_jit = jax.jit(run_scan)
    # Explicitly lower and compile to avoid running the computation
    lowered = run_scan_jit.lower(init_carry)
    compiled_scan = lowered.compile()
    compile_elapsed = time.perf_counter() - compile_start
    print(f"JIT compilation time: {compile_elapsed:.3f}s")

    generation_start = time.perf_counter()

    final_carry, generated_tokens_array = compiled_scan(init_carry)
    logits, curr_pos, Ks_cached, Vs_cached = final_carry
    generated_tokens_array.block_until_ready()
    generation_elapsed = time.perf_counter() - generation_start

    generated_tokens = generated_tokens_array.tolist()
    print()  # Ensure newline after streaming

    final_text = detokenize_ids(tokens + generated_tokens)
    print(f"Final output: {final_text}")

    prefill_tokens_per_s = len(tokens) / prefill_elapsed if prefill_elapsed > 0 else 0.0
    generation_tokens_per_s = (
        len(generated_tokens) / generation_elapsed
        if generation_elapsed > 0 and generated_tokens
        else 0.0
    )
    print(
        f"Prefill avg: {prefill_tokens_per_s:.1f} tokens/s "
        f"({prefill_elapsed:.3f}s total for {len(tokens)} tokens)"
    )
    print(
        f"Generation avg: {generation_tokens_per_s:.1f} tokens/s "
        f"({generation_elapsed:.3f}s total for {len(generated_tokens)} tokens)"
    )


if __name__ == "__main__":
    main()
