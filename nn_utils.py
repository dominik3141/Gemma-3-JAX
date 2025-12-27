import jax.numpy as jnp
import jax


def RMSNorm(x: jax.Array, gamma: jax.Array, epsilon: float) -> jax.Array:
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
