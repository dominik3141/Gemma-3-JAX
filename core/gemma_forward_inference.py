"""
An inference optimized version of the forward function for Gemma.

We only calculate Q,K,V for a single token, the newest token in our sequence, and expect the K,V
values for each layer to have been calculated prior to the invocation of our new forward function.

I'm not yet sure whether we should reuse our current forward function in `gemma_forward.py` for the prefill
step that is supposed to provide us with the prior K,V values, or if we need a third forward function.
We would lose some compute efficiency by reusing our current forward function as it is optimized
for pretraining and calculates the next token as every position, while for prefill we would only need
it to calculate the K,V values.

This is all getting a bit messy, but it looks like we can hardly avoid duplicating logic if we don't want
to accept a lot of conditional branching, which might make our code less efficient (and which generally
isn't something I would consider good style).

TODO:
    - Extract the logic that is shared among all forward functions to a separate file (RMSNorm etc.)
"""

import jax
import jax.numpy as jnp
from core.gemma_forward import (
    Params,
    RMSNorm,
    _model_prefix,
    attnHead,
    calc_qkv,
    config,
    extract_block_params,
    get_gemma3_layer_types,
    postAttn,
)


def group_attention_single(
    Ks: jax.Array,
    Vs: jax.Array,
    Qss: jax.Array,
    pos: jax.Array,
    is_local_attn: jax.Array,
) -> jax.Array:
    """
    Group attention for a single token and a single KV head group.
    """
    Qs = Qss[:, None, :]  # (num_queries_per_group, 1, d_kvq)
    pos_array = jnp.array([pos])
    xs = jax.vmap(attnHead, in_axes=(None, None, 0, None, None))(
        Ks, Vs, Qs, pos_array, is_local_attn
    )
    xs = xs[:, 0, :]  # (num_queries_per_group, d_kvq)
    return jnp.reshape(xs, (config.num_queries_per_group * config.d_kvq,))


def Block_KV_cached(inits, scans) -> jax.Array:
    """
    Gemma block for a single token. Communication with other tokens via KV cache.
    """
    x, pos = (
        inits  # should the position really be carried? Optimally would be a closure?
    )
    block_params, is_local_attn, Ks_cached, Vs_cached = scans

    # make a copy of x to keep the residual
    x_og = x
    x = RMSNorm(x, block_params["input_layernorm.weight"])

    # Calculate comm vectors for new token
    K_new, V_new, Qs = calc_qkv(x, block_params, pos, is_local_attn)

    # Update fixed-size KV cache
    Ks = Ks_cached.at[pos].set(K_new)
    Vs = Vs_cached.at[pos].set(V_new)

    # COMMUNICATION WITH OTHER TOKENS
    x = jax.vmap(
        group_attention_single,
        in_axes=(1, 1, 0, None, None),
    )(Ks, Vs, Qs, pos, is_local_attn)
    x = jnp.reshape(x, (config.num_attention_heads * config.d_kvq,))  # concat heads

    x = postAttn(x, x_og, block_params)

    return (x, pos), (Ks, Vs)


@jax.jit
def forward_single(
    x: jax.Array, params: Params, pos: int, Ks_cached: jax.Array, Vs_cached: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Predict the next token given only a single token together with the K,V vectors
    of all prior tokens.
    Also needs the position of x in the larger sequence so we can calculate the
    positional embedding.
    Returns predicted next token as well as updated K,V cache.
    """
    model_prefix = _model_prefix(params)

    # embed the token
    x = params[f"{model_prefix}embed_tokens.weight"][x]

    # normalize according to Gemma reference implementation
    x = jnp.sqrt(config.d_model) * x

    # single position should be wrapped in array to not confuse later vmaps
    pos = jnp.array(pos)

    # BLOCKS
    # Local/global attention pattern for Gemma 3
    is_local_attn = get_gemma3_layer_types(config.num_layers)
    (x, _), (Ks_cached, Vs_cached) = jax.lax.scan(
        Block_KV_cached,
        (x, pos),
        (
            extract_block_params(params, model_prefix),
            is_local_attn,
            Ks_cached,
            Vs_cached,
        ),
    )

    # final norm
    x = RMSNorm(x, params[f"{model_prefix}norm.weight"])

    # map to logits
    x = params[f"{model_prefix}embed_tokens.weight"] @ x

    return x, Ks_cached, Vs_cached


def get_KV(
    prompt: jax.Array, params: Params, cache_size: int
) -> tuple[jax.Array, jax.Array]:
    """
    Given a tokenized prompt, return the prefilled KV cache.
    """
    n = prompt.shape[0]  # prompt should have shape (seq_len)

    assert cache_size >= n

    K_cache = jnp.zeros(
        (
            config.num_layers,
            cache_size,
            config.num_key_value_heads,
            config.head_dim,
        ),
        dtype=jnp.bfloat16,
    )
    V_cache = jnp.zeros(
        (
            config.num_layers,
            cache_size,
            config.num_key_value_heads,
            config.d_kvq,
        ),
        dtype=jnp.bfloat16,
    )

    def forward_single_scanable(carry, scans):
        Ks_cached, Vs_cached = carry
        x, pos = scans

        _, Ks_cached, Vs_cached = forward_single(x, params, pos, Ks_cached, Vs_cached)

        return (Ks_cached, Vs_cached), None

    pos = jnp.arange(0, n)
    (K_cache, V_cache), _ = jax.lax.scan(
        forward_single_scanable, (K_cache, V_cache), (prompt, pos)
    )

    return K_cache, V_cache


def main() -> None:
    """Test function for forward_single with actual generation."""
    import time

    from utils.params_io_1b import load_params, DEFAULT_ORBAX_CHECKPOINT
    from utils.tokenize_text import tokenize_text, detokenize_ids

    print("Loading weights from Orbax checkpoint...")
    params = load_params(DEFAULT_ORBAX_CHECKPOINT)
    print("Weights loaded.")

    prompt = "The capital of France is Paris. The capital of Germany is"
    tokens = tokenize_text(prompt)

    # Add BOS token (2) if not present
    if tokens[0] != 2:
        tokens = [2] + tokens

    print(f"Prompt: '{prompt}'")
    print(f"Tokens: {tokens}")

    max_new_tokens = 100

    # Initialize cache
    num_layers = config.num_layers
    head_dim = config.head_dim
    kv_cache_len = max_new_tokens + len(tokens) + 1

    assert max_new_tokens + len(tokens) < kv_cache_len

    # Initialize with fixed size
    Ks_cached = jnp.zeros(
        (num_layers, kv_cache_len, config.num_key_value_heads, head_dim),
        dtype=jnp.bfloat16,
    )
    Vs_cached = jnp.zeros(
        (num_layers, kv_cache_len, config.num_key_value_heads, config.d_kvq),
        dtype=jnp.bfloat16,
    )

    print("Processing prompt (prefill)...")
    logits = None

    # Warmup/compile outside measured timings to avoid one-time JIT cost in stats.
    _warmup_token = jnp.array(tokens[0])
    _warmup_pos = 0
    _warmup_K = jnp.zeros_like(Ks_cached)
    _warmup_V = jnp.zeros_like(Vs_cached)
    _warmup_logits, _, _ = forward_single(
        _warmup_token, params, _warmup_pos, _warmup_K, _warmup_V
    )
    _warmup_logits.block_until_ready()

    # Process each token in the prompt
    prefill_start = time.perf_counter()
    for i, token in enumerate(tokens):
        token_id = jnp.array(token)
        pos = i  # Position in sequence

        logits, Ks_cached, Vs_cached = forward_single(
            token_id, params, pos, Ks_cached, Vs_cached
        )
        logits.block_until_ready()
    prefill_elapsed = time.perf_counter() - prefill_start

    print("Prompt processed.")

    # Generate new tokens
    print("Generating:")
    generated_tokens = []

    curr_pos = len(tokens)
    printed_trailing_newline = True

    generation_start = time.perf_counter()
    for _ in range(max_new_tokens):
        # Sample from logits (greedy)
        next_token = jnp.argmax(logits).item()
        generated_tokens.append(next_token)

        # Stream sampled token text immediately.
        token_text = detokenize_ids([next_token])
        if token_text:
            print(token_text, end="", flush=True)
            printed_trailing_newline = token_text.endswith("\n")

        # Feed back
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
