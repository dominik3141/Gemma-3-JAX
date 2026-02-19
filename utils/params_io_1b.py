r"""Load Gemma 3 1B parameters from Orbax checkpoints.

This module intentionally hardcodes the expected checkpoint schema from:
gs://gemma-3-1b-pt-orbax-b76114af/gemma-3-1b-pt-orbax
"""

from __future__ import annotations

import logging
import os

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from jaxtyping import Array, Float

LOGGER = logging.getLogger(__name__)

Params = dict[str, Float[Array, "..."]]

# get parameter only from gcp bucket if not running locally
if os.environ.get("GEMMA_RUNNING_LOCALLY") == "true":
    LOGGER.info("Running locally, using local checkpoint")
    DEFAULT_ORBAX_CHECKPOINT = os.path.abspath("data/gemma-3-1b/gemma-3-1b-pt-orbax")
else:
    LOGGER.info("Running on GCP, using GCP checkpoint")
    DEFAULT_ORBAX_CHECKPOINT = "gs://gemma-3-1b-pt-orbax-b76114af/gemma-3-1b-pt-orbax"

_EXPECTED_DTYPE = jnp.bfloat16

EXPECTED_TARGET_SUFFIX_SPECS: dict[str, jax.ShapeDtypeStruct] = {
    "embed_tokens.weight": jax.ShapeDtypeStruct((262144, 1152), _EXPECTED_DTYPE),
    "layers.input_layernorm.weight": jax.ShapeDtypeStruct((26, 1152), _EXPECTED_DTYPE),
    "layers.mlp.down_proj.weight": jax.ShapeDtypeStruct((26, 1152, 6912), _EXPECTED_DTYPE),
    "layers.mlp.gate_proj.weight": jax.ShapeDtypeStruct((26, 6912, 1152), _EXPECTED_DTYPE),
    "layers.mlp.up_proj.weight": jax.ShapeDtypeStruct((26, 6912, 1152), _EXPECTED_DTYPE),
    "layers.post_attention_layernorm.weight": jax.ShapeDtypeStruct(
        (26, 1152), _EXPECTED_DTYPE
    ),
    "layers.post_feedforward_layernorm.weight": jax.ShapeDtypeStruct(
        (26, 1152), _EXPECTED_DTYPE
    ),
    "layers.pre_feedforward_layernorm.weight": jax.ShapeDtypeStruct(
        (26, 1152), _EXPECTED_DTYPE
    ),
    "layers.self_attn.k_norm.weight": jax.ShapeDtypeStruct((26, 256), _EXPECTED_DTYPE),
    "layers.self_attn.k_proj.weight": jax.ShapeDtypeStruct((26, 256, 1152), _EXPECTED_DTYPE),
    "layers.self_attn.o_proj.weight": jax.ShapeDtypeStruct(
        (26, 1152, 1024), _EXPECTED_DTYPE
    ),
    "layers.self_attn.q_norm.weight": jax.ShapeDtypeStruct((26, 256), _EXPECTED_DTYPE),
    "layers.self_attn.q_proj.weight": jax.ShapeDtypeStruct(
        (26, 1024, 1152), _EXPECTED_DTYPE
    ),
    "layers.self_attn.v_proj.weight": jax.ShapeDtypeStruct((26, 256, 1152), _EXPECTED_DTYPE),
    "norm.weight": jax.ShapeDtypeStruct((1152,), _EXPECTED_DTYPE),
}

_MODEL_PREFIX = "model."


def _build_target() -> dict[str, jax.ShapeDtypeStruct]:
    """
    Build the abstract pytree Orbax restores into for Gemma 3 1B.

    We keep the schema explicit by hardcoding known parameter suffixes and then
    attaching the fixed 1B model prefix (`model.`) to each key.
    """
    return {
        f"{_MODEL_PREFIX}{suffix}": jax.ShapeDtypeStruct(spec.shape, spec.dtype)
        for suffix, spec in EXPECTED_TARGET_SUFFIX_SPECS.items()
    }


def load_params(path: str) -> Params:
    """
    Load 1B parameters via a metadata-first restore flow.

    This function is intentionally opinionated:
    1. read metadata first so the checkpoint structure is inspectable before load
    2. build the known hardcoded abstract target tree
    3. let Orbax restore directly into that target
    """
    checkpointer = ocp.StandardCheckpointer()
    # Intentionally fetch metadata before restore so schema is inspectable ahead of load.
    checkpointer.metadata(path)
    target = _build_target()
    return checkpointer.restore(path, target)
