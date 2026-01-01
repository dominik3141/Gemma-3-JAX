from gemma_forward import forward
from inspect_weights import load_weights_as_dict
import jax.numpy as jnp


if __name__ == "__main__":
    params = load_weights_as_dict("model_stacked.safetensors")
    # Convert all weights to JAX arrays once at the start
    params = {k: jnp.array(v) for k, v in params.items()}

    # initialize an empty kv cache
    kv_cache = jnp.zeros((26, 0, 2))

    xs = jnp.array([2, 153637, 532, 622])
    xs = forward(xs, params, kv_cache)
    print(xs)
