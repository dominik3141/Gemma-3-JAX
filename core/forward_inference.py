"""
An inference optimized version of the forward function for Gemma.

We only calculate Q,K,V for a single token, the newest token in our sequence, and expect the K,V
values for each layer to have been calculated prior to the invocation of our new forward function.

I'm not yet sure whether we should reuse our current forward function in `gemma_forward_parralel.py` for the prefill
step that is supposed to provide us with the prior K,V values, or if we need a third forward function.
We would lose some compute efficiency by reusing our current forward function as it is optimized
for pretraining and calculates the next token as every position, while for prefill we would only need
it to calculate the K,V values.

This is all getting a bit messy, but it looks like we can hardly avoid duplicating logic if we don't want
to accept a lot of conditional branching, which might make our code less efficient (and which generally
isn't something I would consider good style).
"""

import functools

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, jaxtyped
from core.forward_common import (
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
    Ks: Float[Array, "seq_len head_dim"],
    Vs: Float[Array, "seq_len value_dim"],
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
    carry: tuple[
        Float[Array, "d_model"],
        Int[Array, ""],
        Float[Array, "layer seq_len kv_head head_dim"],
        Float[Array, "layer seq_len kv_head value_dim"],
    ],
    scans: tuple[Params, bool | Bool[Array, ""], Int[Array, ""]],
) -> tuple[
    tuple[
        Float[Array, "d_model"],
        Int[Array, ""],
        Float[Array, "layer seq_len kv_head head_dim"],
        Float[Array, "layer seq_len kv_head value_dim"],
    ],
    None,
]:
    """
    Gemma block for a single token. Communication with other tokens via KV cache.
    """
    x, pos, Ks_all_layers, Vs_all_layers = carry
    block_params, is_local_attn, layer_idx = scans
    Ks_cached = Ks_all_layers[layer_idx]
    Vs_cached = Vs_all_layers[layer_idx]

    # make a copy of x to keep the residual
    x_og = x
    x = RMSNorm(x, block_params["input_layernorm.weight"])

    # Calculate comm vectors for new token
    K_new, V_new, Qs = calc_qkv(x, block_params, pos, is_local_attn)

    # Update fixed-size KV cache
    Ks_updated = Ks_cached.at[pos].set(K_new)
    Vs_updated = Vs_cached.at[pos].set(V_new)

    # COMMUNICATION WITH OTHER TOKENS
    x = jax.vmap(
        group_attention_single,
        in_axes=(1, 1, 0, None, None),
    )(Ks_updated, Vs_updated, Qs, pos, is_local_attn)
    x = jnp.reshape(x, (config.num_attention_heads * config.d_kvq,))  # concat heads

    x = postAttn(x, x_og, block_params)

    Ks_all_layers = Ks_all_layers.at[layer_idx].set(Ks_updated)
    Vs_all_layers = Vs_all_layers.at[layer_idx].set(Vs_updated)

    return (x, pos, Ks_all_layers, Vs_all_layers), None


@functools.partial(jax.jit, donate_argnames=("Ks_cached", "Vs_cached"))
@jaxtyped(typechecker=beartype)
def forward_single(
    x: int | Int[Array, ""],
    params: Params,
    pos: int | Int[Array, ""],
    Ks_cached: Float[Array, "layer seq_len kv_head head_dim"],
    Vs_cached: Float[Array, "layer seq_len kv_head value_dim"],
) -> tuple[
    Float[Array, "vocab"],
    Float[Array, "layer seq_len kv_head head_dim"],
    Float[Array, "layer seq_len kv_head value_dim"],
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
    layer_indices = jnp.arange(config.num_layers, dtype=jnp.int32)
    (x, _, Ks_cached, Vs_cached), _ = jax.lax.scan(
        Block_KV_cached,
        (x, pos, Ks_cached, Vs_cached),
        (
            extract_block_params(params, model_prefix),
            is_local_attn,
            layer_indices,
        ),
    )

    # final norm
    x = RMSNorm(x, params[f"{model_prefix}norm.weight"])

    # map to logits
    x = params[f"{model_prefix}embed_tokens.weight"] @ x

    return x, Ks_cached, Vs_cached


def allocate_kv_cache(
    batch_size: int,
    kv_cache_len: int,
) -> tuple[
    Float[Array, "layer batch seq_len kv_head head_dim"],
    Float[Array, "layer batch seq_len kv_head value_dim"],
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


@functools.partial(jax.jit, donate_argnames=("ks_cached", "vs_cached"))
def prefill(
    params: Params,
    tokens: Int[Array, "prompt_len"],
    seq_length: int | Int[Array, ""],
    ks_cached: Float[Array, "layer seq_len kv_head head_dim"],
    vs_cached: Float[Array, "layer seq_len kv_head value_dim"],
) -> tuple[
    Float[Array, "vocab"],
    Float[Array, "layer seq_len kv_head head_dim"],
    Float[Array, "layer seq_len kv_head value_dim"],
]:
    logits, ks_cached, vs_cached = forward_single(
        tokens[0], params, jnp.array(0, dtype=jnp.int32), ks_cached, vs_cached
    )
    last_real_logits = logits

    def prefill_step(
        carry: tuple[
            Float[Array, "vocab"],
            Float[Array, "layer seq_len kv_head head_dim"],
            Float[Array, "layer seq_len kv_head value_dim"],
            Float[Array, "vocab"],
        ],
        t: Int[Array, ""],
    ) -> tuple[
        tuple[
            Float[Array, "vocab"],
            Float[Array, "layer seq_len kv_head head_dim"],
            Float[Array, "layer seq_len kv_head value_dim"],
            Float[Array, "vocab"],
        ],
        None,
    ]:
        _, ks_curr, vs_curr, last_logits_curr = carry
        logits_curr, ks_next, vs_next = forward_single(
            tokens[t],
            params,
            t,
            ks_curr,
            vs_curr,
        )
        last_logits_next = jnp.where(t < seq_length, logits_curr, last_logits_curr)
        return (logits_curr, ks_next, vs_next, last_logits_next), None

    (_, ks_cached, vs_cached, last_real_logits), _ = jax.lax.scan(
        prefill_step,
        (logits, ks_cached, vs_cached, last_real_logits),
        jnp.arange(1, tokens.shape[0], dtype=jnp.int32),
    )

    return last_real_logits, ks_cached, vs_cached


@functools.partial(
    jax.jit,
    static_argnames=("max_new_tokens", "temperature"),
    donate_argnames=("ks_cached", "vs_cached"),
)
def decode(
    params: Params,
    key: PRNGKeyArray,
    logits: Float[Array, "vocab"],
    curr_pos: int | Int[Array, ""],
    ks_cached: Float[Array, "layer seq_len kv_head head_dim"],
    vs_cached: Float[Array, "layer seq_len kv_head value_dim"],
    max_new_tokens: int,
    temperature: float,
) -> tuple[
    Int[Array, "decode_len"],
    Float[Array, "decode_len"],
    Float[Array, "layer seq_len kv_head head_dim"],
    Float[Array, "layer seq_len kv_head value_dim"],
]:
    curr_pos = jnp.asarray(curr_pos, dtype=jnp.int32)
    sample_keys = jax.random.split(key, max_new_tokens)

    def decode_step(
        carry: tuple[
            Float[Array, "vocab"],
            Int[Array, ""],
            Float[Array, "layer seq_len kv_head head_dim"],
            Float[Array, "layer seq_len kv_head value_dim"],
        ],
        step_key: PRNGKeyArray,
    ) -> tuple[
        tuple[
            Float[Array, "vocab"],
            Int[Array, ""],
            Float[Array, "layer seq_len kv_head head_dim"],
            Float[Array, "layer seq_len kv_head value_dim"],
        ],
        tuple[Int[Array, ""], Float[Array, ""]],
    ]:
        logits_curr, pos_curr, ks_curr, vs_curr = carry
        gumbel_noise = jax.random.gumbel(
            step_key, logits_curr.shape, dtype=logits_curr.dtype
        )
        next_token = jnp.argmax(logits_curr + temperature * gumbel_noise).astype(
            jnp.int32
        )

        # Easy mistake: temperature=0 is a valid greedy decode mode for sampling, but
        # log-probs of softmax(logits / temperature) are undefined at temperature=0.
        # Convention here: when temperature==0, report log-probs under unscaled logits
        # (equivalent to temperature=1). Callers must treat this as a convention and not
        # as the same policy semantics as non-zero-temperature sampling log-probs.
        log_prob_temperature = jnp.where(temperature == 0.0, 1.0, temperature)
        scaled_logits = logits_curr / log_prob_temperature
        next_token_logit = scaled_logits[next_token]
        next_token_log_prob = next_token_logit - jax.nn.logsumexp(
            scaled_logits, axis=-1
        )

        logits_next, ks_next, vs_next = forward_single(
            next_token,
            params,
            pos_curr,
            ks_curr,
            vs_curr,
        )
        return (logits_next, pos_curr + 1, ks_next, vs_next), (
            next_token,
            next_token_log_prob,
        )

    (_, _, ks_cached, vs_cached), (generated_tokens, generated_log_probs) = jax.lax.scan(
        decode_step,
        (logits, curr_pos, ks_cached, vs_cached),
        sample_keys,
    )

    return generated_tokens, generated_log_probs, ks_cached, vs_cached
