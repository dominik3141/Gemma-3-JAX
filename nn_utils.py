import jax.numpy as jnp
import jax
from functools import partial


Params = dict[str, jax.Array]


def RMSNorm(x: jax.Array, gamma: jax.Array) -> jax.Array:
    # the weights for this norm are simply named '*layernorm*'
    # x and gamma should have the same shape
    epsilon = 1e-6
    return x / (jnp.sqrt(jnp.mean(jnp.square(x)) + epsilon)) * gamma


def RoPE(x: jax.Array, position: int) -> jax.Array:
    """
    Implemented by Gemini as I got annoyed with it.
    Needs some optimization later, but good enough for now.
    """
    theta_base = 10000.0

    # head_dim is the last dimension of our tensor
    d = x.shape[-1]

    # Compute the frequency constants for the dimensions
    # We only need d//2 frequencies because we work in 2D planes
    idx = jnp.arange(0, d, 2)
    freqs = position / (theta_base ** (idx / d))

    # Calculate cos and sin components
    # We repeat them to match the full head_dim
    cos = jnp.cos(jnp.concatenate([freqs, freqs], axis=-1))
    sin = jnp.sin(jnp.concatenate([freqs, freqs], axis=-1))

    # Split x into [x_left, x_right] to perform the rotation
    # For a vector [x1, x2, x3, x4], x_left is [x1, x2], x_right is [x3, x4]
    half = d // 2
    x_left = x[..., :half]
    x_right = x[..., half:]

    # The 'rotated' part of the formula: [-x_right, x_left]
    # This corresponds to the complex rotation (x + iy) * (cos + isin)
    x_rotated = jnp.concatenate([-x_right, x_left], axis=-1)

    return (x * cos) + (x_rotated * sin)


def mlp(
    x_0: jax.Array,
    down_proj_weight: jax.Array,
    gate_proj_weight: jax.Array,
    up_proj_weight: jax.Array,
    activation_fn,
) -> jax.Array:
    x_a = activation_fn(gate_proj_weight @ x_0)
    x_b = up_proj_weight @ x_0
    x = x_a * x_b  # elementwise multiplication
    x = down_proj_weight @ x

    return x


def preAttn(
    x: jax.Array, block_params: Params, pos: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # Prepare for attention
    K = block_params["self_attn.k_proj.weight"] @ x
    K = RMSNorm(K, block_params["self_attn.k_norm.weight"])
    K = RoPE(K, pos)
    V = block_params["self_attn.v_proj.weight"] @ x
    Qs = block_params["self_attn.q_proj.weight"] @ x
    Qs = jnp.reshape(Qs, (4, 256))
    Qs = jax.vmap(lambda Q: RMSNorm(Q, block_params["self_attn.q_norm.weight"]))(Qs)
    Qs = jax.vmap(RoPE, in_axes=(0, None))(Qs, pos)

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
        jax.nn.gelu,
    )
    x = x + x_mlp_residual
    x = RMSNorm(x, block_params["post_feedforward_layernorm.weight"])

    return x


def Attn(Q_a: jax.Array, Ks: jax.Array) -> jax.Array:
    d_k = Q_a.shape[0]
    assert d_k == 256

    return jax.nn.softmax((Q_a @ jnp.transpose(Ks)) / jnp.sqrt(d_k))


def attnHead(Ks, Vs, Qs) -> jax.Array:
    r"""
    We define
    Z_a := \sum_b Attn(a,b) * V_b,
    where Attn(a,b) := softmax((Q_a K_b^T)/sqrt(d_k))
    """
    Z_a = jax.vmap(lambda Ks, Vs, Q_a: Attn(Q_a, Ks) @ Vs, in_axes=(None, None, 0))(
        Ks, Vs, Qs
    )

    return Z_a


def Block(params: Params, xs: jax.Array, block_id: int) -> jax.Array:
    r"""
    block_params has keys:
    input_layernorm.weight (1152,)
    mlp.down_proj.weight (1152, 6912)
    mlp.gate_proj.weight (6912, 1152)
    mlp.up_proj.weight (6912, 1152)
    post_attention_layernorm.weight (1152,)
    post_feedforward_layernorm.weight (1152,)
    pre_feedforward_layernorm.weight (1152,)
    self_attn.k_norm.weight (256,)
    self_attn.k_proj.weight (256, 1152)
    self_attn.o_proj.weight (1152, 1024)
    self_attn.q_norm.weight (256,)
    self_attn.q_proj.weight (1024, 1152)
    self_attn.v_proj.weight (256, 1152)

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
    block_params = _block_params(params, block_id)

    # make a copy of x to keep the residual
    # maybe this should be done after the layernorm?
    xs_og = xs
    xs = jax.vmap(lambda x: RMSNorm(x, block_params["input_layernorm.weight"]))(xs)

    # we need to also keep track of a tokens position within the sequence of tokens so that we can calculate the
    # positional embedding
    sequence_len = xs.shape[0]
    pos = jnp.arange(0, sequence_len, 1, dtype=int)

    Ks, Vs, Qss = jax.vmap(preAttn, in_axes=(0, None, 0))(xs, block_params, pos)

    # COMMUNICATION WITH OTHER TOKENS
    r"""
    The usual representation of the attention formula hides a lot of interesting stuff behind matrix operations,
    we try to write a much cleaner and more explicit version here while still preserving the usual performance. 

    For a fixed a, we want to calculate
        Z_a := \sum_b Attn(a,b) * V_b,
        where Attn(a,b) := softmax((Q_a K_b^T)/sqrt(d_k))
    Z_a can be thought of as a version of a that has been enriched
    with information from other tokens.

    That would be easy enough for single head attention, but with four heads we have some extra level to take care of.
    """
    # first we go onto the level of individual heads
    xs = jax.vmap(lambda Ks, Vs, Qs: attnHead(Ks, Vs, Qs), in_axes=(None, None, 0))(
        Ks, Vs, Qss
    )
    xs = jnp.reshape(xs, (sequence_len, 1024))

    xs = jax.vmap(postAttn, in_axes=(0, 0, None))(xs, xs_og, block_params)

    return xs, None


def _block_params(params: Params, block_id: int) -> Params:
    """
    Returns the block parameters for a block of given id.

    block_params has keys:
    input_layernorm.weight (1152,)
    mlp.down_proj.weight (1152, 6912)
    mlp.gate_proj.weight (6912, 1152)
    mlp.up_proj.weight (6912, 1152)
    post_attention_layernorm.weight (1152,)
    post_feedforward_layernorm.weight (1152,)
    pre_feedforward_layernorm.weight (1152,)
    self_attn.k_norm.weight (256,)
    self_attn.k_proj.weight (256, 1152)
    self_attn.o_proj.weight (1152, 1024)
    self_attn.q_norm.weight (256,)
    self_attn.q_proj.weight (1024, 1152)
    self_attn.v_proj.weight (256, 1152)
    """
    prefix = f"model.layers.{block_id}."

    block_params = {
        "input_layernorm.weight": params[prefix + "input_layernorm.weight"],
        "mlp.down_proj.weight": params[prefix + "mlp.down_proj.weight"],
        "mlp.gate_proj.weight": params[prefix + "mlp.gate_proj.weight"],
        "mlp.up_proj.weight": params[prefix + "mlp.up_proj.weight"],
        "post_attention_layernorm.weight": params[
            prefix + "post_attention_layernorm.weight"
        ],
        "post_feedforward_layernorm.weight": params[
            prefix + "post_feedforward_layernorm.weight"
        ],
        "pre_feedforward_layernorm.weight": params[
            prefix + "pre_feedforward_layernorm.weight"
        ],
        "self_attn.k_norm.weight": params[prefix + "self_attn.k_norm.weight"],
        "self_attn.k_proj.weight": params[prefix + "self_attn.k_proj.weight"],
        "self_attn.o_proj.weight": params[prefix + "self_attn.o_proj.weight"],
        "self_attn.q_norm.weight": params[prefix + "self_attn.q_norm.weight"],
        "self_attn.q_proj.weight": params[prefix + "self_attn.q_proj.weight"],
        "self_attn.v_proj.weight": params[prefix + "self_attn.v_proj.weight"],
    }

    return block_params


def forward(xs: jax.Array, params: Params) -> jax.Array:
    r"""
    Plan:
    1.  Embedd the tokens
        (seq_len, 262144) -> (seq_len, 1152)
    2.  Shift left
        (seq_len, 1152) -> (seq_len, 1152)
    3.  Iterate over blocks
        (seq_len, 1152) -> (seq_len, 1152)
    4.  Final norm
        (seq_len, 1152) -> (seq_len, 1152)
    5.  Map to logits
        (seq_len, 1152) -> (seq_len, 262144)
    """
    # embedding the tokens
    xs = jax.vmap(lambda x: jnp.transpose(params["model.embed_tokens.weight"]) @ x)(xs)

    # shift left
    xs = jnp.concat([xs[1:], jnp.zeros_like(xs[:1])])

    # BLOCKS
    block_ids = jnp.arange(0, 25, 1)  # 0, 1, 2, ..., 25
    Block_fn_with_param = partial(Block, params)
    xs, _ = jax.lax.scan(Block_fn_with_param, xs, block_ids)

    # final norm
    xs = jax.vmap(RMSNorm, in_axes=(0, None))(xs, params["model.norm.weight"])

    # map to logits
    xs = jax.vmap(lambda x: params["model.embed_tokens.weight"] @ x)(xs)

    return xs
