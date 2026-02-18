import jax.numpy as jnp

from core.gemma_forward_parralel import forward_parralel
from utils.params_io_1b import DEFAULT_ORBAX_CHECKPOINT, load_params


def main():
    params = load_params(DEFAULT_ORBAX_CHECKPOINT)
    xs = jnp.array([2, 4237, 3234, 1293094])
    return forward_parralel(xs, params)


if __name__ == "__main__":
    print(main())
