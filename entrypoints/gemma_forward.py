import jax.numpy as jnp

from core.gemma_forward import forward
from utils.params_io_27b import DEFAULT_ORBAX_CHECKPOINT, load_params


def main():
    params = load_params(DEFAULT_ORBAX_CHECKPOINT)
    xs = jnp.array([2, 4237, 3234, 1293094])
    return forward(xs, params)


if __name__ == "__main__":
    print(main())
