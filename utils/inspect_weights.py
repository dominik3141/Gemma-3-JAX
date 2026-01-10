from safetensors import safe_open
import numpy as np
import jax.numpy as jnp
import jax
import ml_dtypes  # noqa: F401


def load_weights_as_dict(path: str) -> dict[str, jax.Array]:
    """
    Reads a safetensors file and returns a dictionary of jax arrays.
    """
    tensors: dict[str, np.ndarray] = {}

    with safe_open(path, framework="np", device="cpu") as f:
        for key in f.keys():
            # .get_tensor() returns a view into the memory-mapped file
            tensors[key] = f.get_tensor(key)

    # Convert all weights to JAX arrays
    for k, v in tensors.items():
        tensors[k] = jnp.array(v)

    return tensors


if __name__ == "__main__":
    weights = load_weights_as_dict("data/model_stacked_pt.safetensors")
    for key in weights.keys():
        print(key, weights[key].shape)

    print(weights["model.layers_stacked.input_layernorm.weight"])
