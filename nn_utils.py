import jax.numpy as jnp
import jax


def RMSNorm(x: jax.Array, gamma: jax.Array, epsilon: float) -> jax.Array:
    # x and gamma should have the same shape
    return x / (jnp.sqrt(jnp.mean(jnp.square(x)) + epsilon)) * gamma
