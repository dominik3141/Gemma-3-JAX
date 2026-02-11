r"""
All comments that include concrete dimensionality numbers are written with the 1B version of Gemma in mind.
"""

import jax.numpy as jnp
import jax
from functools import partial
from config.model import gemma_3_1b as config


Params = dict[str, jax.Array]


def RMSNorm(x: jax.Array, gamma: jax.Array) -> jax.Array:
    # the weights for this norm are simply named '*layernorm*'
    # x and gamma should have the same shape
    epsilon = 1e-6
    return x / (jnp.sqrt(jnp.mean(jnp.square(x)) + epsilon)) * (1 + gamma)


def RoPE(x: jax.Array, position: int, theta: float) -> jax.Array:
    d = x.shape[-1]
    if d % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, got {d}")

    x_dtype = x.dtype
    x_f = x.astype(jnp.float32)

    half = d // 2
    x1 = x_f[:half]
    x2 = x_f[half:]

    idx = jnp.arange(0, d, 2, dtype=jnp.float32)  # length = half
    pos = jnp.asarray(position, dtype=jnp.float32)
    freqs = pos / (theta ** (idx / d))  # length = half

    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)

    # Complex multiply: (x1 + i x2) * (cos + i sin)
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos

    y = jnp.concatenate([y1, y2], axis=-1)
    return y.astype(x_dtype)


def mlp(
    x_0: jax.Array,
    down_proj_weight: jax.Array,
    gate_proj_weight: jax.Array,
    up_proj_weight: jax.Array,
) -> jax.Array:
    x_a = jax.nn.gelu(gate_proj_weight @ x_0, approximate=True)
    x_b = up_proj_weight @ x_0
    x = x_a * x_b  # elementwise multiplication
    x = down_proj_weight @ x

    return x


def calc_qkv(
    x: jax.Array, block_params: Params, pos: jax.Array, is_local_attn: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # Prepare for attention
    theta = jnp.where(is_local_attn, 10_000.0, 1_000_000.0)

    Ks = block_params["self_attn.k_proj.weight"] @ x
    Ks = jnp.reshape(Ks, (config.num_key_value_heads, config.head_dim))
    Ks = jax.vmap(RMSNorm, in_axes=(0, None))(
        Ks, block_params["self_attn.k_norm.weight"]
    )
    Ks = jax.vmap(RoPE, in_axes=(0, None, None))(Ks, pos, theta)

    Vs = block_params["self_attn.v_proj.weight"] @ x
    Vs = jnp.reshape(Vs, (config.num_key_value_heads, config.d_kvq))

    Qss = block_params["self_attn.q_proj.weight"] @ x
    Qss = jnp.reshape(
        Qss,
        (config.num_key_value_heads * config.num_queries_per_group, config.d_kvq),
    )
    Qss = jax.vmap(lambda Q: RMSNorm(Q, block_params["self_attn.q_norm.weight"]))(Qss)
    Qss = jax.vmap(lambda Q, p, theta: RoPE(Q, p, theta), in_axes=(0, None, None))(
        Qss, pos, theta
    )
    Qss = jnp.reshape(
        Qss,
        (config.num_key_value_heads, config.num_queries_per_group, config.d_kvq),
    )

    return Ks, Vs, Qss


def postAttn(x: jax.Array, x_og: jax.Array, block_params: Params) -> jax.Array:
    # map attention output back to d_model
    x = block_params["self_attn.o_proj.weight"] @ x

    # Norm and residual
    x = RMSNorm(x, block_params["post_attention_layernorm.weight"])
    x = x + x_og

    # MLP
    x_mlp_residual = x
    x = RMSNorm(x, block_params["pre_feedforward_layernorm.weight"])
    x = mlp(
        x,
        block_params["mlp.down_proj.weight"],
        block_params["mlp.gate_proj.weight"],
        block_params["mlp.up_proj.weight"],
    )
    x = RMSNorm(x, block_params["post_feedforward_layernorm.weight"])
    x = x + x_mlp_residual

    return x


def AttnScores(
    Q_a: jax.Array, Ks: jax.Array, idx_a: jax.Array, seq_indices, local_attn: bool
) -> jax.Array:
    """
    Calculates masked attention scores.
    """
    d_k = Q_a.shape[0]

    scale = jnp.asarray(1.0 / (d_k**0.5), dtype=Q_a.dtype)
    scores = (Q_a @ jnp.transpose(Ks)) * scale

    # causal masking
    scores = jnp.where(seq_indices <= idx_a, scores, -jnp.inf)

    if local_attn:
        scores = jnp.where(
            idx_a - seq_indices <= config.sliding_window, scores, -jnp.inf
        )

    return jax.nn.softmax(scores.astype(jnp.float32)).astype(scores.dtype)


def attnHead(Ks, Vs, Qs, pos_a, is_local_attn) -> jax.Array:
    r"""
    We define
    Z_a := \sum_b Attn(a,b) * V_b,
    where Attn(a,b) := softmax((Q_a K_b^T)/sqrt(d_k))
    """

    def Attn(is_local_attn: bool, Ks, Vs, Qs, pos_a, seq_indices) -> jax.Array:
        return jax.vmap(
            lambda Ks, Vs, Q_a, idx_a, seq_indices: AttnScores(
                Q_a, Ks, idx_a, seq_indices, is_local_attn
            )
            @ Vs,
            in_axes=(None, None, 0, 0, None),
        )(Ks, Vs, Qs, pos_a, seq_indices)

    localAttn = partial(Attn, True)
    globalAttn = partial(Attn, False)

    seq_indices = jnp.arange(0, Ks.shape[0], 1)
    Z_a = jax.lax.cond(
        is_local_attn, localAttn, globalAttn, Ks, Vs, Qs, pos_a, seq_indices
    )

    return Z_a


def group_attention(
    xs, Ks, Vs, Qss, sequence_len: int, is_local_attn, pos
) -> jax.Array:
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
def Block(xs: jax.Array, scans) -> jax.Array:
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


def _model_prefix(params: Params) -> str:
    if "language_model.model.embed_tokens.weight" in params:
        return "language_model.model."
    if "model.embed_tokens.weight" in params:
        return "model."
    raise KeyError("Missing embed_tokens.weight in params (model prefix not found).")


def extract_block_params(params: Params, prefix: str) -> Params:
    block_params = {}
    layer_prefix = f"{prefix}layers."

    for key, val in params.items():
        if key.startswith(layer_prefix):
            _, suffix = key.split(layer_prefix, 1)
            block_params[suffix] = val

    return block_params


def get_gemma3_layer_types(num_layers: int) -> jax.Array:
    """
    Generates a boolean mask for Gemma 3 attention layers.
    True = Local sliding window attention
    False = Global attention

    Pattern: 5 local, 1 global (Ratio 5:1), starting with local.
    """
    # Create indices: [0, 1, 2, ..., num_layers - 1]
    indices = jnp.arange(num_layers)

    # Global layers are at indices 5, 11, 17, etc. (where (i+1) % 6 == 0)
    # We want True for local, so we check for NOT being a global index.
    is_local = (indices + 1) % 6 != 0

    return is_local


@jax.jit
def forward(xs: jax.Array, params: Params) -> jax.Array:
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


@jax.jit
def main():
    from utils.params_io_27b import DEFAULT_ORBAX_CHECKPOINT, load_params

    params = load_params(DEFAULT_ORBAX_CHECKPOINT)
    xs = jnp.array([2, 4237, 3234, 1293094])

    xs = forward(xs, params)

    return xs


if __name__ == "__main__":
    print(main())
