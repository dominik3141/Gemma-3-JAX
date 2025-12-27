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


def Block(x: jax.Array, block_params: Params, layer: int) -> jax.Array:
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
    pass
