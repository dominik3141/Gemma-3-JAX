import jax.numpy as jnp
import jax
from functools import partial


Params = dict[str, jax.Array]


def RMSNorm(x: jax.Array, gamma: jax.Array) -> jax.Array:
    # the weights for this norm are simply named '*layernorm*'
    # x and gamma should have the same shape
    epsilon = 1e-6
    return x / (jnp.sqrt(jnp.mean(jnp.square(x)) + epsilon)) * (1 + gamma)


def RoPE(x: jax.Array, position: int, theta: float) -> jax.Array:
    """
    Gemma-style RoPE matching the PyTorch reference you pasted (complex rotation
    with half/half layout, not even/odd interleaving).
    """
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
    K = block_params["self_attn.k_proj.weight"] @ x
    K = RMSNorm(K, block_params["self_attn.k_norm.weight"])
    K = RoPE(K, pos, theta)

    V = block_params["self_attn.v_proj.weight"] @ x

    Qs = block_params["self_attn.q_proj.weight"] @ x
    Qs = jnp.reshape(Qs, (4, 256))
    Qs = jax.vmap(lambda Q: RMSNorm(Q, block_params["self_attn.q_norm.weight"]))(Qs)
    Qs = jax.vmap(lambda Q, p, theta: RoPE(Q, p, theta), in_axes=(0, None, None))(
        Qs, pos, theta
    )

    return K, V, Qs


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

    scores = (Q_a @ jnp.transpose(Ks)) / jnp.sqrt(d_k)

    # causal masking
    scores = jnp.where(seq_indices <= idx_a, scores, -jnp.inf)

    if local_attn:
        # we need a mask m of the form
        # [0,...,0,1,...1,0,...0],
        # where m_i = 1 iff | i - idx_a | <= 512
        # because of causal masking, we can simplify this and always assume
        # that we are looking back
        scores = jnp.where(idx_a - seq_indices <= 1024, scores, -jnp.inf)

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

    Ks, Vs, Qss = jax.vmap(calc_qkv, in_axes=(0, None, 0, None))(
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
    # first we go onto the level of individual heads
    Qss = jnp.transpose(Qss, (1, 0, 2))  # head dimension should be first
    xs = jax.vmap(
        attnHead,
        in_axes=(None, None, 0, None, None),
    )(Ks, Vs, Qss, pos, is_local_attn)
    xs = jnp.transpose(xs, (1, 0, 2))  # (Seq, 4, 256)
    xs = jnp.reshape(xs, (sequence_len, 4 * 256))  # concat heads

    xs = jax.vmap(postAttn, in_axes=(0, 0, None))(xs, xs_og, block_params)

    return xs, None


def extract_block_params(params: Params) -> Params:
    block_params = {}

    for key, val in params.items():
        if key.startswith("model.layers_stacked"):
            prefix, suffix = key.split("model.layers_stacked.")
            block_params[suffix] = val

    return block_params


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
    # Padding
    # we enforce a minimum sequence length of 1024 tokens in order to guarantee
    # that there are enough tokens for proper dynamic slicing during the local attention
    # layers
    # To avoid constant jit recompilation the padding should already have been added by the caller
    input_length = xs.shape[0]
    padding_tokens = max(0, 1024 - input_length)
    xs = jnp.concatenate([xs, jnp.zeros_like(xs, shape=(padding_tokens,))])

    # embedding the tokens
    xs = params["model.embed_tokens.weight"][xs]

    # normalize according to Gemma reference implementation
    xs = jnp.sqrt(1152) * xs

    # BLOCKS
    # Create the pattern list based on the config
    # 26 layers total: (5 local, 1 global) * 4 + 2 local
    layer_types = (["local"] * 5 + ["global"]) * 4 + ["local"] * 2
    is_local_attn = jnp.array(
        [t == "local" for t in layer_types]
    )  # shape (26,), [1,1...,1,1]
    xs, _ = jax.lax.scan(Block, xs, (extract_block_params(params), is_local_attn))

    # remove padding again (to save memory)
    xs = xs[:input_length]

    # final norm
    xs = jax.vmap(RMSNorm, in_axes=(0, None))(xs, params["model.norm.weight"])

    # map to logits
    xs = xs @ params["model.embed_tokens.weight"].T

    return xs
