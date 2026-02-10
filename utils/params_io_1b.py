import jax
import orbax.checkpoint as ocp
import os

Params = dict[str, jax.Array]

DEFAULT_ORBAX_CHECKPOINT = "gs://gemma-3-1b-pt-orbax-b76114af/gemma-3-1b-pt-orbax"


def load_params(path: str) -> Params:
    "Use orbax to load parameters"
    ckpt = ocp.StandardCheckpointer().restore(path)

    return ckpt
