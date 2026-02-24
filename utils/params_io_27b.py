r"""
Load and save Gemma 3 27B parameters.

This module intentionally hardcodes the expected checkpoint schema from:
gs://gemma-3-27b-pt-orbax-b76114af/gemma-3-27b-pt-orbax
"""

from __future__ import annotations

import uuid

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array, Float
from optax.contrib import MuonDimensionNumbers

Params = dict[str, Float[Array, "..."]]

DEFAULT_ORBAX_CHECKPOINT = "gs://gemma-3-27b-pt-orbax-b76114af/gemma-3-27b-pt-orbax"
DEFAULT_GCS_SAVE_ROOT = "gs://gemma-tpu-weights-us-west4-482802-euw4/checkpoints"
_EXPECTED_DTYPE = jnp.bfloat16


EXPECTED_TARGET_SPECS: dict[str, jax.ShapeDtypeStruct] = {
    "language_model.model.embed_tokens.weight": jax.ShapeDtypeStruct(
        (262208, 5376), _EXPECTED_DTYPE
    ),
    "language_model.model.layers.input_layernorm.weight": jax.ShapeDtypeStruct(
        (62, 5376), _EXPECTED_DTYPE
    ),
    "language_model.model.layers.mlp.down_proj.weight": jax.ShapeDtypeStruct(
        (62, 5376, 21504), _EXPECTED_DTYPE
    ),
    "language_model.model.layers.mlp.gate_proj.weight": jax.ShapeDtypeStruct(
        (62, 21504, 5376), _EXPECTED_DTYPE
    ),
    "language_model.model.layers.mlp.up_proj.weight": jax.ShapeDtypeStruct(
        (62, 21504, 5376), _EXPECTED_DTYPE
    ),
    "language_model.model.layers.post_attention_layernorm.weight": jax.ShapeDtypeStruct(
        (62, 5376), _EXPECTED_DTYPE
    ),
    "language_model.model.layers.post_feedforward_layernorm.weight": jax.ShapeDtypeStruct(
        (62, 5376), _EXPECTED_DTYPE
    ),
    "language_model.model.layers.pre_feedforward_layernorm.weight": jax.ShapeDtypeStruct(
        (62, 5376), _EXPECTED_DTYPE
    ),
    "language_model.model.layers.self_attn.k_norm.weight": jax.ShapeDtypeStruct(
        (62, 128), _EXPECTED_DTYPE
    ),
    "language_model.model.layers.self_attn.k_proj.weight": jax.ShapeDtypeStruct(
        (62, 2048, 5376), _EXPECTED_DTYPE
    ),
    "language_model.model.layers.self_attn.o_proj.weight": jax.ShapeDtypeStruct(
        (62, 5376, 4096), _EXPECTED_DTYPE
    ),
    "language_model.model.layers.self_attn.q_norm.weight": jax.ShapeDtypeStruct(
        (62, 128), _EXPECTED_DTYPE
    ),
    "language_model.model.layers.self_attn.q_proj.weight": jax.ShapeDtypeStruct(
        (62, 4096, 5376), _EXPECTED_DTYPE
    ),
    "language_model.model.layers.self_attn.v_proj.weight": jax.ShapeDtypeStruct(
        (62, 2048, 5376), _EXPECTED_DTYPE
    ),
    "language_model.model.norm.weight": jax.ShapeDtypeStruct((5376,), _EXPECTED_DTYPE),
    "multi_modal_projector.mm_input_projection_weight": jax.ShapeDtypeStruct(
        (1152, 5376), _EXPECTED_DTYPE
    ),
    "multi_modal_projector.mm_soft_emb_norm.weight": jax.ShapeDtypeStruct(
        (1152,), _EXPECTED_DTYPE
    ),
    "vision_tower.vision_model.embeddings.patch_embedding.bias": jax.ShapeDtypeStruct(
        (1152,), _EXPECTED_DTYPE
    ),
    "vision_tower.vision_model.embeddings.patch_embedding.weight": jax.ShapeDtypeStruct(
        (1152, 3, 14, 14), _EXPECTED_DTYPE
    ),
    "vision_tower.vision_model.embeddings.position_embedding.weight": jax.ShapeDtypeStruct(
        (4096, 1152), _EXPECTED_DTYPE
    ),
    "vision_tower.vision_model.encoder.layers.layer_norm1.bias": jax.ShapeDtypeStruct(
        (27, 1152), _EXPECTED_DTYPE
    ),
    "vision_tower.vision_model.encoder.layers.layer_norm1.weight": jax.ShapeDtypeStruct(
        (27, 1152), _EXPECTED_DTYPE
    ),
    "vision_tower.vision_model.encoder.layers.layer_norm2.bias": jax.ShapeDtypeStruct(
        (27, 1152), _EXPECTED_DTYPE
    ),
    "vision_tower.vision_model.encoder.layers.layer_norm2.weight": jax.ShapeDtypeStruct(
        (27, 1152), _EXPECTED_DTYPE
    ),
    "vision_tower.vision_model.encoder.layers.mlp.fc1.bias": jax.ShapeDtypeStruct(
        (27, 4304), _EXPECTED_DTYPE
    ),
    "vision_tower.vision_model.encoder.layers.mlp.fc1.weight": jax.ShapeDtypeStruct(
        (27, 4304, 1152), _EXPECTED_DTYPE
    ),
    "vision_tower.vision_model.encoder.layers.mlp.fc2.bias": jax.ShapeDtypeStruct(
        (27, 1152), _EXPECTED_DTYPE
    ),
    "vision_tower.vision_model.encoder.layers.mlp.fc2.weight": jax.ShapeDtypeStruct(
        (27, 1152, 4304), _EXPECTED_DTYPE
    ),
    "vision_tower.vision_model.encoder.layers.self_attn.k_proj.bias": jax.ShapeDtypeStruct(
        (27, 1152), _EXPECTED_DTYPE
    ),
    "vision_tower.vision_model.encoder.layers.self_attn.k_proj.weight": jax.ShapeDtypeStruct(
        (27, 1152, 1152), _EXPECTED_DTYPE
    ),
    "vision_tower.vision_model.encoder.layers.self_attn.out_proj.bias": jax.ShapeDtypeStruct(
        (27, 1152), _EXPECTED_DTYPE
    ),
    "vision_tower.vision_model.encoder.layers.self_attn.out_proj.weight": jax.ShapeDtypeStruct(
        (27, 1152, 1152), _EXPECTED_DTYPE
    ),
    "vision_tower.vision_model.encoder.layers.self_attn.q_proj.bias": jax.ShapeDtypeStruct(
        (27, 1152), _EXPECTED_DTYPE
    ),
    "vision_tower.vision_model.encoder.layers.self_attn.q_proj.weight": jax.ShapeDtypeStruct(
        (27, 1152, 1152), _EXPECTED_DTYPE
    ),
    "vision_tower.vision_model.encoder.layers.self_attn.v_proj.bias": jax.ShapeDtypeStruct(
        (27, 1152), _EXPECTED_DTYPE
    ),
    "vision_tower.vision_model.encoder.layers.self_attn.v_proj.weight": jax.ShapeDtypeStruct(
        (27, 1152, 1152), _EXPECTED_DTYPE
    ),
    "vision_tower.vision_model.post_layernorm.bias": jax.ShapeDtypeStruct(
        (1152,), _EXPECTED_DTYPE
    ),
    "vision_tower.vision_model.post_layernorm.weight": jax.ShapeDtypeStruct(
        (1152,), _EXPECTED_DTYPE
    ),
}

SHARDING_PLAN: dict[str, PartitionSpec] = {
    "language_model.model.embed_tokens.weight": PartitionSpec("model", None),
    "language_model.model.layers.input_layernorm.weight": PartitionSpec(None, None),
    "language_model.model.layers.mlp.down_proj.weight": PartitionSpec(
        None, None, "model"
    ),
    "language_model.model.layers.mlp.gate_proj.weight": PartitionSpec(
        None, "model", None
    ),
    "language_model.model.layers.mlp.up_proj.weight": PartitionSpec(
        None, "model", None
    ),
    "language_model.model.layers.post_attention_layernorm.weight": PartitionSpec(
        None, None
    ),
    "language_model.model.layers.post_feedforward_layernorm.weight": PartitionSpec(
        None, None
    ),
    "language_model.model.layers.pre_feedforward_layernorm.weight": PartitionSpec(
        None, None
    ),
    "language_model.model.layers.self_attn.k_norm.weight": PartitionSpec(None, None),
    "language_model.model.layers.self_attn.k_proj.weight": PartitionSpec(
        None, "model", None
    ),
    "language_model.model.layers.self_attn.o_proj.weight": PartitionSpec(
        None, None, "model"
    ),
    "language_model.model.layers.self_attn.q_norm.weight": PartitionSpec(None, None),
    "language_model.model.layers.self_attn.q_proj.weight": PartitionSpec(
        None, "model", None
    ),
    "language_model.model.layers.self_attn.v_proj.weight": PartitionSpec(
        None, "model", None
    ),
    "language_model.model.norm.weight": PartitionSpec(None),
    "multi_modal_projector.mm_input_projection_weight": PartitionSpec(None, None),
    "multi_modal_projector.mm_soft_emb_norm.weight": PartitionSpec(None),
    "vision_tower.vision_model.embeddings.patch_embedding.bias": PartitionSpec(None),
    "vision_tower.vision_model.embeddings.patch_embedding.weight": PartitionSpec(
        None, None, None, None
    ),
    "vision_tower.vision_model.embeddings.position_embedding.weight": PartitionSpec(
        None, None
    ),
    "vision_tower.vision_model.encoder.layers.layer_norm1.bias": PartitionSpec(
        None, None
    ),
    "vision_tower.vision_model.encoder.layers.layer_norm1.weight": PartitionSpec(
        None, None
    ),
    "vision_tower.vision_model.encoder.layers.layer_norm2.bias": PartitionSpec(
        None, None
    ),
    "vision_tower.vision_model.encoder.layers.layer_norm2.weight": PartitionSpec(
        None, None
    ),
    "vision_tower.vision_model.encoder.layers.mlp.fc1.bias": PartitionSpec(None, None),
    "vision_tower.vision_model.encoder.layers.mlp.fc1.weight": PartitionSpec(
        None, None, None
    ),
    "vision_tower.vision_model.encoder.layers.mlp.fc2.bias": PartitionSpec(None, None),
    "vision_tower.vision_model.encoder.layers.mlp.fc2.weight": PartitionSpec(
        None, None, None
    ),
    "vision_tower.vision_model.encoder.layers.self_attn.k_proj.bias": PartitionSpec(
        None, None
    ),
    "vision_tower.vision_model.encoder.layers.self_attn.k_proj.weight": PartitionSpec(
        None, None, "model"
    ),
    "vision_tower.vision_model.encoder.layers.self_attn.out_proj.bias": PartitionSpec(
        None, None
    ),
    "vision_tower.vision_model.encoder.layers.self_attn.out_proj.weight": PartitionSpec(
        None, None, None
    ),
    "vision_tower.vision_model.encoder.layers.self_attn.q_proj.bias": PartitionSpec(
        None, None
    ),
    "vision_tower.vision_model.encoder.layers.self_attn.q_proj.weight": PartitionSpec(
        None, None, "model"
    ),
    "vision_tower.vision_model.encoder.layers.self_attn.v_proj.bias": PartitionSpec(
        None, None
    ),
    "vision_tower.vision_model.encoder.layers.self_attn.v_proj.weight": PartitionSpec(
        None, None, "model"
    ),
    "vision_tower.vision_model.post_layernorm.bias": PartitionSpec(None),
    "vision_tower.vision_model.post_layernorm.weight": PartitionSpec(None),
}


def _build_target(mesh: Mesh) -> dict[str, jax.ShapeDtypeStruct]:
    """
    Build the abstract pytree Orbax restores into.

    We restore directly as sharded arrays. The expected structure comes from
    our hardcoded schema, and the sharding comes from SHARDING_PLAN.
    """
    if "model" not in mesh.axis_names:
        raise ValueError("Mesh is missing required axis name 'model'.")

    target: dict[str, jax.ShapeDtypeStruct] = {}
    for key, base_target in EXPECTED_TARGET_SPECS.items():
        spec = SHARDING_PLAN[key]
        sharding = NamedSharding(mesh, spec)
        target[key] = jax.ShapeDtypeStruct(
            base_target.shape,
            base_target.dtype,
            sharding=sharding,
        )
    return target


def muon_weight_dimension_numbers_for_27b(
    params: Params,
) -> dict[str, MuonDimensionNumbers | None]:
    """
    Build Muon reshape specs for the Gemma 27B parameter tree.

    Plan:
    1.  1D leaves are marked as ``None`` so Muon routes them to Adam.
    2.  2D leaves are interpreted as ``W @ x`` matrices with
        ``reduction_axis=1`` and ``output_axis=0``.
    3.  3D leaves use axis 0 as a batch axis (stacked layers), with
        ``reduction_axis=2`` and ``output_axis=1``.
    4.  The vision patch embedding convolution uses a dedicated 4D spec that
        treats ``(in_channels, kernel_h, kernel_w)`` as reduction axes.
    """
    dim_numbers: dict[str, MuonDimensionNumbers | None] = {}
    for key, value in params.items():
        if key not in EXPECTED_TARGET_SPECS:
            raise ValueError(f"Unexpected 27B parameter key: {key!r}")

        # During masked optimizer updates, Optax may pass sentinel leaf values
        # (e.g. MaskedNode) that intentionally do not expose ndarray attributes.
        # Treat those leaves as non-Muon for this call site.
        ndim = getattr(value, "ndim", None)
        if ndim is None:
            dim_numbers[key] = None
            continue

        if key == "vision_tower.vision_model.embeddings.patch_embedding.weight":
            if ndim != 4:
                raise ValueError(
                    "Vision patch embedding weight must be rank-4, got "
                    f"{ndim} for {key!r}."
                )
            dim_numbers[key] = MuonDimensionNumbers(
                reduction_axis=(1, 2, 3),
                output_axis=0,
            )
            continue

        if ndim == 1:
            dim_numbers[key] = None
        elif ndim == 2:
            dim_numbers[key] = MuonDimensionNumbers(reduction_axis=1, output_axis=0)
        elif ndim == 3:
            dim_numbers[key] = MuonDimensionNumbers(reduction_axis=2, output_axis=1)
        elif ndim == 4:
            raise ValueError(
                f"Unexpected rank-4 parameter {key!r}. Add an explicit Muon reshape rule."
            )
        else:
            raise ValueError(f"Unsupported parameter rank {ndim} for key {key!r}.")
    return dim_numbers


def _join_checkpoint_path(root: str, name: str) -> str:
    """Join checkpoint root and id without caring about trailing slashes."""
    return f"{root.rstrip('/')}/{name}"


def load_params(
    checkpoint_path: str,
    mesh: Mesh,
) -> Params:
    """
    Load the 27B checkpoint into a user-provided mesh.

    This function is intentionally opinionated now:
    1. build the known checkpoint schema with sharding attached
    2. let Orbax restore directly into those sharded arrays
    """
    ckptr = ocp.StandardCheckpointer()
    target = _build_target(mesh)
    return ckptr.restore(checkpoint_path, target)


def save_params(params: Params) -> str:
    """
    Save params to the default GCS checkpoint root with a random checkpoint id.
    """
    checkpoint_id = uuid.uuid4().hex
    checkpoint_path = _join_checkpoint_path(DEFAULT_GCS_SAVE_ROOT, checkpoint_id)
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(checkpoint_path, params)
    return checkpoint_path
