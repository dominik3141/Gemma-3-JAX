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
from beartype import beartype
from jaxtyping import Array, Bool, Float, Int, jaxtyped
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
    Ks: Float[Array, "cache_pos head_dim"],
    Vs: Float[Array, "cache_pos value_dim"],
    Qss: Float[Array, "q_per_group head_dim"],
    pos: int | Int[Array, ""],
    is_local_attn: bool | Bool[Array, ""],
) -> Float[Array, "q_group_dim"]:
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


def Block_KV_cached(
    inits: tuple[Float[Array, "d_model"], Int[Array, ""]],
    scans: tuple[
        Params,
        bool | Bool[Array, ""],
        Float[Array, "cache_pos kv_head head_dim"],
        Float[Array, "cache_pos kv_head value_dim"],
    ],
) -> tuple[
    tuple[Float[Array, "d_model"], Int[Array, ""]],
    tuple[
        Float[Array, "cache_pos kv_head head_dim"],
        Float[Array, "cache_pos kv_head value_dim"],
    ],
]:
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
@jaxtyped(typechecker=beartype)
def forward_single(
    x: int | Int[Array, ""],
    params: Params,
    pos: int | Int[Array, ""],
    Ks_cached: Float[Array, "layer cache_pos kv_head head_dim"],
    Vs_cached: Float[Array, "layer cache_pos kv_head value_dim"],
) -> tuple[
    Float[Array, "vocab"],
    Float[Array, "layer cache_pos kv_head head_dim"],
    Float[Array, "layer cache_pos kv_head value_dim"],
]:
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
    prompt: Int[Array, "seq"], params: Params, cache_size: int
) -> tuple[
    Float[Array, "layer cache_pos kv_head head_dim"],
    Float[Array, "layer cache_pos kv_head value_dim"],
]:
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


def allocate_kv_cache(
    batch_size: int, kv_cache_len: int
) -> tuple[
    Float[Array, "layer batch cache_pos kv_head head_dim"],
    Float[Array, "layer batch cache_pos kv_head value_dim"],
]:
    ks_cached = jnp.zeros(
        (
            config.num_layers,
            batch_size,
            kv_cache_len,
            config.num_key_value_heads,
            config.head_dim,
        ),
        dtype=jnp.bfloat16,
    )
    vs_cached = jnp.zeros(
        (
            config.num_layers,
            batch_size,
            kv_cache_len,
            config.num_key_value_heads,
            config.d_kvq,
        ),
        dtype=jnp.bfloat16,
    )
    return ks_cached, vs_cached


def prefill(
    params: Params,
    prompt_tokens: Int[Array, "batch prompt_len"],
    prompt_lengths: Int[Array, "batch"],
    prompt_last_tokens: Int[Array, "batch"],
    kv_cache_len: int,
) -> tuple[
    Float[Array, "batch vocab"],
    Float[Array, "layer batch cache_pos kv_head head_dim"],
    Float[Array, "layer batch cache_pos kv_head value_dim"],
]:
    init_ks_cached, init_vs_cached = allocate_kv_cache(
        prompt_tokens.shape[0], kv_cache_len
    )
    forward_batch = jax.vmap(
        forward_single, in_axes=(0, None, 0, 1, 1), out_axes=(0, 1, 1)
    )

    pos0 = jnp.zeros((prompt_tokens.shape[0],), dtype=jnp.int32)
    logits, ks_cached, vs_cached = forward_batch(
        prompt_tokens[:, 0], params, pos0, init_ks_cached, init_vs_cached
    )

    for t in range(1, prompt_tokens.shape[1]):
        is_active = t < prompt_lengths
        pos = jnp.where(is_active, t, prompt_lengths - 1).astype(jnp.int32)
        token_ids = jnp.where(
            is_active, prompt_tokens[:, t], prompt_last_tokens
        ).astype(jnp.int32)
        logits, ks_cached, vs_cached = forward_batch(
            token_ids, params, pos, ks_cached, vs_cached
        )

    return logits, ks_cached, vs_cached


def decode(
    params: Params,
    logits: Float[Array, "batch vocab"],
    curr_pos: Int[Array, "batch"],
    ks_cached: Float[Array, "layer batch cache_pos kv_head head_dim"],
    vs_cached: Float[Array, "layer batch cache_pos kv_head value_dim"],
    max_new_tokens: int,
) -> Int[Array, "decode_len batch"]:
    forward_batch = jax.vmap(
        forward_single, in_axes=(0, None, 0, 1, 1), out_axes=(0, 1, 1)
    )

    generated_tokens = []

    for _ in range(max_new_tokens):
        next_tokens = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        generated_tokens.append(next_tokens)
        logits, ks_cached, vs_cached = forward_batch(
            next_tokens, params, curr_pos, ks_cached, vs_cached
        )
        curr_pos = curr_pos + 1

    if generated_tokens:
        generated_tokens_array = jnp.stack(generated_tokens, axis=0)
    else:
        generated_tokens_array = jnp.zeros((0, logits.shape[0]), dtype=jnp.int32)

    return generated_tokens_array
