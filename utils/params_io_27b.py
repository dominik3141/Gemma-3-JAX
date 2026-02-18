r"""
Load and save Gemma 3 27B parameters.

This module intentionally hardcodes the expected checkpoint schema from:
gs://gemma-3-27b-pt-orbax-b76114af/gemma-3-27b-pt-orbax
"""

from __future__ import annotations

import uuid

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array, Float

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
    "language_model.model.embed_tokens.weight": PartitionSpec(None, "model"),
    "language_model.model.layers.input_layernorm.weight": PartitionSpec(None, None),
    "language_model.model.layers.mlp.down_proj.weight": PartitionSpec(
        None, "model", None
    ),
    "language_model.model.layers.mlp.gate_proj.weight": PartitionSpec(
        None, None, "model"
    ),
    "language_model.model.layers.mlp.up_proj.weight": PartitionSpec(None, None, "model"),
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
        None, None, "model"
    ),
    "language_model.model.layers.self_attn.o_proj.weight": PartitionSpec(
        None, "model", None
    ),
    "language_model.model.layers.self_attn.q_norm.weight": PartitionSpec(None, None),
    "language_model.model.layers.self_attn.q_proj.weight": PartitionSpec(
        None, None, "model"
    ),
    "language_model.model.layers.self_attn.v_proj.weight": PartitionSpec(
        None, None, "model"
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

EXPECTED_KEYS: tuple[str, ...] = tuple(EXPECTED_TARGET_SPECS.keys())


def _build_target() -> dict[str, jax.ShapeDtypeStruct]:
    """
    Build the abstract pytree Orbax restores into.

    The checkpoint bytes are fixed, so this only encodes expected shape/dtype.
    We attach sharding later in a separate step.
    """
    return dict(EXPECTED_TARGET_SPECS)


def _restore_to_host(
    ckptr: ocp.StandardCheckpointer,
    checkpoint_path: str,
) -> Params:
    """
    Restore tensors to CPU RAM first.

    This keeps loading and sharding as separate steps, which makes multihost
    behavior easier to reason about and avoids relying on implicit placement.
    """
    # restore to host memory first, then explicitly place/shard below
    cpu_device = jax.devices("cpu")[0]
    target = _build_target()
    with jax.default_device(cpu_device):
        return ckptr.restore(checkpoint_path, target)


def _slice_for_host(
    value: np.ndarray,
    spec: PartitionSpec,
    mesh: Mesh,
) -> np.ndarray:
    """
    Take only the local shard for this process when a tensor is model-sharded.

    We derive the split from the mesh's `model` axis, not from process_count.
    That means additional mesh axes (for example `data`) can replicate params
    without changing the sharding plan.
    """
    axes = tuple(spec)
    if "model" not in axes:
        return value

    if axes.count("model") > 1:
        raise ValueError("Multiple 'model' axes not supported in host slicing.")

    if "model" not in mesh.axis_names:
        raise ValueError("Mesh is missing required axis name 'model'.")

    tensor_axis = axes.index("model")
    model_axis = mesh.axis_names.index("model")
    model_axis_size = mesh.devices.shape[model_axis]

    dim = value.shape[tensor_axis]
    if dim % model_axis_size != 0:
        raise ValueError(
            f"Axis {tensor_axis} with size {dim} is not divisible by "
            f"mesh model axis size {model_axis_size}."
        )

    process_index = jax.process_index()
    model_positions: list[int] = []
    for model_i in range(model_axis_size):
        index = [slice(None)] * mesh.devices.ndim
        index[model_axis] = model_i
        device_slice = mesh.devices[tuple(index)]
        process_ids = {device.process_index for device in np.ravel(device_slice)}
        if process_index in process_ids:
            model_positions.append(model_i)

    if not model_positions:
        raise ValueError(
            f"Process {process_index} owns no positions on mesh model axis."
        )

    if any(b != a + 1 for a, b in zip(model_positions, model_positions[1:])):
        raise ValueError(
            "Model-axis positions for this process are not contiguous; "
            "cannot slice host tensor as one block."
        )

    chunk = dim // model_axis_size
    start = model_positions[0] * chunk
    end = (model_positions[-1] + 1) * chunk
    slices = [slice(None)] * value.ndim
    slices[tensor_axis] = slice(start, end)
    return value[tuple(slices)]


def _shard_to_mesh(params: Params, mesh: Mesh) -> Params:
    """
    Convert host arrays into global arrays with our hardcoded sharding.

    On single host this is a straight `device_put` with NamedSharding.
    On multihost we first take the local chunk and then stitch globals with
    `host_local_array_to_global_array`.
    """
    sharded: Params = {}
    is_multihost = jax.process_count() > 1

    for key in EXPECTED_KEYS:
        value = params[key]
        spec = SHARDING_PLAN[key]
        value_np = np.asarray(value)

        if is_multihost:
            # each host only materializes its local model-axis chunk
            local_value = _slice_for_host(
                value_np,
                spec,
                mesh,
            )
            sharded[key] = multihost_utils.host_local_array_to_global_array(
                local_value,
                mesh,
                spec,
            )
        else:
            sharding = NamedSharding(mesh, spec)
            sharded[key] = jax.device_put(value_np, sharding)
    return sharded


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
    1. restore the known checkpoint schema to host memory
    2. apply the fixed sharding plan using the given mesh
    3. return sharded params
    """
    ckptr = ocp.StandardCheckpointer()
    # this avoids full-tensor device_put on multihost setups
    cpu_state = _restore_to_host(ckptr, checkpoint_path)
    params = _shard_to_mesh(cpu_state, mesh)
    return params


def save_params(params: Params) -> str:
    """
    Save params to the default GCS checkpoint root with a random checkpoint id.
    """
    checkpoint_id = uuid.uuid4().hex
    checkpoint_path = _join_checkpoint_path(DEFAULT_GCS_SAVE_ROOT, checkpoint_id)
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(checkpoint_path, params)
    return checkpoint_path
