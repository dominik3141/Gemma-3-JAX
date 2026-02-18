r"""
All comments that include concrete dimensionality numbers are written with the 1B version of Gemma in mind.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Float, Int, jaxtyped

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


def group_attention(
    xs: Float[Array, "seq d_model"],
    Ks: Float[Array, "seq head_dim"],
    Vs: Float[Array, "seq value_dim"],
    Qss: Float[Array, "seq q_per_group head_dim"],
    sequence_len: int,
    is_local_attn: bool | Bool[Array, ""],
    pos: Int[Array, "seq"],
) -> Float[Array, "seq q_group_dim"]:
    """
    In GQA, there might be mutliple queries per group.
    """
    Qss = jnp.transpose(Qss, (1, 0, 2))  # different queries as first axis

    # inside the group, we descend onto the level of individual queries
    xs = jax.vmap(
        attnHead,
        in_axes=(None, None, 0, None, None),
    )(Ks, Vs, Qss, pos, is_local_attn)
    xs = jnp.transpose(xs, (1, 0, 2))  # seq_len first
    xs = jnp.reshape(
        xs, (sequence_len, config.num_queries_per_group * config.d_kvq)
    )  # concat query results

    return xs


@jax.jit  # the scan should already compile this, but better to be explicit
@jax.checkpoint  # OOM problems without this
def Block(
    xs: Float[Array, "seq d_model"],
    scans: tuple[Params, bool | Bool[Array, ""]],
) -> tuple[Float[Array, "seq d_model"], None]:
    r"""
    Plan:
    1.  Input layernorm
    2.  Derive K,V and all four Q matrices.
        Get K, V, Q_{1, 2, 3, 4}: (256,)
    3.  Calculate attention
        4 x (256,)
    4.  Concat heads
        (4 x 256,)
    5.  Output projection
        (1152,)
    6.  Layernorm
        (1152,)
    7.  Add residual
        (1152,)
    8.  Layernorm
        (1152,)
    9.  MLP
        (1152,) -> (6912,) -> (1152,)
    10. Layernorm
        (1152,)
    """
    block_params, is_local_attn = scans

    # make a copy of x to keep the residual
    # maybe this should be done after the layernorm?
    xs_og = xs
    xs = jax.vmap(lambda x: RMSNorm(x, block_params["input_layernorm.weight"]))(xs)

    # we need to also keep track of a tokens position within the sequence of tokens so that we can calculate the
    # positional embedding
    sequence_len = xs.shape[0]
    pos = jnp.arange(0, sequence_len, 1, dtype=int)

    Kss, Vss, Qsss = jax.vmap(calc_qkv, in_axes=(0, None, 0, None))(
        xs, block_params, pos, is_local_attn
    )
    # COMMUNICATION WITH OTHER TOKENS
    r"""
    The usual representation of the attention formula hides a lot of interesting stuff behind matrix operations,
    we try to write a much cleaner and more explicit version here while still preserving the usual performance. 

    For a fixed a, we want to calculate
        Z_a := \sum_b Attn(a,b) * V_b,
        where Attn(a,b) := softmax((Q_a K_b^T)/sqrt(d_k))
    Z_a can be thought of as a version of a that has been enriched with information from other tokens.

    That would be easy enough for single head attention, but with four heads we have some extra level to take care of.
    """
    # we go onto the level of individual groups
    xs = jax.vmap(
        group_attention,
        in_axes=(None, 1, 1, 1, None, None, None),
    )(
        xs,
        Kss,
        Vss,
        Qsss,
        sequence_len,
        is_local_attn,
        pos,
    )
    xs = jnp.transpose(xs, (1, 0, 2))
    xs = jnp.reshape(
        xs, (sequence_len, config.num_attention_heads * config.d_kvq)
    )  # concat heads

    xs = jax.vmap(postAttn, in_axes=(0, 0, None))(xs, xs_og, block_params)

    return xs, None

@jax.jit
@jaxtyped(typechecker=beartype)
def forward_parralel(
    xs: Int[Array, "seq"], params: Params
) -> Float[Array, "seq vocab"]:
    r"""
    Input is a sequence of ids, each referencing a specific token in vocab
    i.e. [1233, 12, 83238, ....]
    we use 0 as the padding token (not our decision)

    Plan:
    1.  Padding
    2.  Embed the tokens
        (seq_len, ) -> (seq_len, 1152)
    3.  Iterate over blocks
        (seq_len, 1152) -> (seq_len, 1152)
    4.  Final norm
        (seq_len, 1152) -> (seq_len, 1152)
    5.  Map to logits
        (seq_len, 1152) -> (seq_len, 262144)
    """
    model_prefix = _model_prefix(params)

    # embedding the tokens
    xs = params[f"{model_prefix}embed_tokens.weight"][xs]

    # normalize according to Gemma reference implementation
    xs = jnp.sqrt(config.d_model) * xs

    # BLOCKS
    # Generate the pattern list based on the 5:1 interleaving rule
    is_local_attn = get_gemma3_layer_types(config.num_layers)
    xs, _ = jax.lax.scan(
        Block, xs, (extract_block_params(params, model_prefix), is_local_attn)
    )

    # final norm
    xs = jax.vmap(RMSNorm, in_axes=(0, None))(xs, params[f"{model_prefix}norm.weight"])

    # map to logits
    xs = xs @ params[f"{model_prefix}embed_tokens.weight"].T

    return xs
