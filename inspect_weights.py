from safetensors import safe_open
import numpy as np
import ml_dtypes  # noqa: F401


def load_weights_as_dict(path: str) -> dict[str, np.ndarray]:
    """
    Reads a safetensors file and returns a dictionary of numpy arrays.
    """
    tensors: dict[str, np.ndarray] = {}

    with safe_open(path, framework="np", device="cpu") as f:
        for key in f.keys():
            # .get_tensor() returns a view into the memory-mapped file
            tensors[key] = f.get_tensor(key)

    return tensors


if __name__ == "__main__":
    weights = load_weights_as_dict("model_stacked.safetensors")
    for key in weights.keys():
        print(key, weights[key].shape)
