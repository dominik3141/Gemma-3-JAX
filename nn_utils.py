import jax.numpy as jnp
import jax
from main import Params


def RMSNorm(x: jax.Array, gamma: jax.Array) -> jax.Array:
    # the weights for this norm are simply named '*layernorm*'
    # x and gamma should have the same shape
    epsilon = 1e-6
    return x / (jnp.sqrt(jnp.mean(jnp.square(x)) + epsilon)) * gamma


def RoPE(x: jax.Array, pos: jax.Array) -> jax.Array:
    pass


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


def gelu(x: jax.Array) -> jax.Array:
    pass


def preAttn(
    x: jax.Array, block_params: Params
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # Prepare for attention
    K = block_params["self_attn.k_proj.weight"] @ x
    K = RMSNorm(K, block_params["self_attn.k_norm.weight"])
    V = block_params["self_attn.v_proj.weight"] @ x
    Qs = block_params["self_attn.q_proj.weight"] @ x
    Qs = jnp.reshape(Qs, (4, 256))
    Qs = jax.vmap(lambda Q: RMSNorm(Q, block_params["self_attn.q_norm.weight"]))(Qs)

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
        gelu,
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


def Block(xs: jax.Array, block_params: Params) -> jax.Array:
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
    # make a copy of x to keep the residual
    # maybe this should be done after the layernorm?
    xs_og = xs
    xs = jax.vmap(lambda x: RMSNorm(x, block_params["input_layernorm.weight"]))(xs)

    Ks, Vs, Qss = jax.vmap(preAttn, in_axes=(0, None))(xs, block_params)

    # COMMUNICATION WITH OTHER TOKENS
    """
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
    sequence_len = xs.shape[0]
    xs = jnp.reshape(xs, (sequence_len, 1024))

    xs = jax.vmap(postAttn, in_axes=(0, 0, None))(xs, xs_og, block_params)

    return xs
