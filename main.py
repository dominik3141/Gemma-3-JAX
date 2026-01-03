from gemma_forward import forward
from inspect_weights import load_weights_as_dict
import jax.numpy as jnp


if __name__ == "__main__":
    params = load_weights_as_dict("model_stacked.safetensors")

    xs = jnp.array([2, 153637, 532, 622])
    xs = forward(xs, params)
    print(xs)
