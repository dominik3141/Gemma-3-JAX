import jax.numpy as jnp
import jax
from main import Params


def RMSNorm(x: jax.Array, gamma: jax.Array, epsilon: float) -> jax.Array:
    # the weights for this norm are simply named '*layernorm*'
    # x and gamma should have the same shape
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


def Block(x: jax.Array, block_params: Params) -> jax.Array:
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
    epsilon = 1e-6
    # make a copy of x to keep the residual
    # maybe this should be done after the layernorm?
    x_og = x

    x = RMSNorm(x, block_params["input_layernorm.weight"], epsilon)

    # Prepare for attention
    K = block_params["self_attn.k.proj.weight"] @ x
    K = RMSNorm(K, block_params["self_attn.k.norm.weight"], epsilon)
    V = block_params["self_attn.v.proj.weight"] @ x
    Qs = block_params["self_attn.q.proj.weight"] @ x
    Qs = jnp.reshape(Qs, (4, 256))
    Qs = jax.vmap(
        lambda Q: RMSNorm(Q, block_params["self_attn.q.norm.weight"], epsilon)
    )

    # COMMUNICATION WITH OTHER TOKENS
    # x = ?

    # map attention output back to d_model
    x = block_params["self_attn.o.proj.weight"] @ x

    # Norm and residual
    x = RMSNorm(x, block_params["post_attention_layernorm.weight"], epsilon)
    x = x + x_og

    # MLP
    x = RMSNorm(x, block_params["pre_feedforward_layernorm.weight"], epsilon)
    x = mlp(
        x,
        block_params["mlp.down_proj.weight"],
        block_params["mlp.gate_proj.weight"],
        block_params["mlp.up_proj.weight"],
        gelu,
    )
    x = RMSNorm(x, block_params["post_feedforward_layernorm.weight"], epsilon)

    return x
