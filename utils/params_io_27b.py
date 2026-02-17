r"""
Can retrieve and save parameters of Gemma 3 27B.
Logic for this version is a bit more complicated than for the 1b version due to the need for sharded loading.
"""

from __future__ import annotations

import uuid
from typing import Iterable, Tuple, Any

import jax
import numpy as np
from jaxtyping import Array, Float
from jax.sharding import NamedSharding, PartitionSpec
import orbax.checkpoint as ocp
from jax.experimental import multihost_utils


DEFAULT_ORBAX_CHECKPOINT = "gs://gemma-3-27b-pt-orbax-b76114af/gemma-3-27b-pt-orbax"
DEFAULT_GCS_SAVE_ROOT = "gs://gemma-tpu-weights-us-west4-482802-euw4/checkpoints"


def _normalize_spec_for_rank(
    spec: PartitionSpec,
    rank: int,
) -> PartitionSpec:
    axes = tuple(spec)
    if len(axes) < rank:
        axes = axes + (None,) * (rank - len(axes))
    elif len(axes) > rank:
        raise ValueError(
            f"PartitionSpec {axes} has more axes than tensor rank {rank}."
        )
    return PartitionSpec(*axes)


def get_sharding_spec(
    name: str,
    rank: int,
) -> PartitionSpec:
    """
    Returns the PartitionSpec for a given weight name.
    """

    if "embed_tokens" in name:
        spec = PartitionSpec(None, "model")
    elif "q_proj" in name or "k_proj" in name or "v_proj" in name:
        spec = PartitionSpec("model", None)
    elif "o_proj" in name:
        spec = PartitionSpec(None, "model")
    elif "gate_proj" in name or "up_proj" in name:
        spec = PartitionSpec("model", None)
    elif "down_proj" in name:
        spec = PartitionSpec(None, "model")
    else:
        spec = PartitionSpec(None)

    return _normalize_spec_for_rank(spec, rank)


def _unwrap_metadata(meta_tree: Any) -> Any:
    if hasattr(meta_tree, "item_metadata"):
        return meta_tree.item_metadata
    return meta_tree


def _path_to_key(path: Iterable[Any]) -> str:
    if len(path) == 1:
        p = path[0]
        if hasattr(p, "key"):
            return p.key
        if isinstance(p, str):
            return p
    parts = []
    for p in path:
        if hasattr(p, "key"):
            parts.append(str(p.key))
        elif hasattr(p, "idx"):
            parts.append(str(p.idx))
        else:
            parts.append(str(p))
    return ".".join(parts)


def _iter_metadata(meta_tree: Any) -> Iterable[Tuple[str, Any]]:
    if isinstance(meta_tree, dict):
        for key, meta in meta_tree.items():
            yield _normalize_key(key), meta
        return

    leaves, _ = jax.tree_util.tree_flatten_with_path(meta_tree)
    for path, meta in leaves:
        key = _path_to_key(path)
        yield _normalize_key(key), meta


def _normalize_key(key: Any) -> str:
    if isinstance(key, tuple) and len(key) == 1 and isinstance(key[0], str):
        return key[0]
    if isinstance(key, str):
        return key
    return str(key)


def _meta_shape_dtype(meta: Any) -> Tuple[Tuple[int, ...], Any]:
    if hasattr(meta, "shape"):
        shape = tuple(meta.shape)
    elif isinstance(meta, dict) and "shape" in meta:
        shape = tuple(meta["shape"])
    else:
        raise ValueError("Missing shape in checkpoint metadata.")

    if hasattr(meta, "dtype"):
        dtype = meta.dtype
    elif isinstance(meta, dict) and "dtype" in meta:
        dtype = meta["dtype"]
    else:
        raise ValueError("Missing dtype in checkpoint metadata.")

    return shape, dtype


def _join_checkpoint_path(root: str, name: str) -> str:
    if root.endswith("/"):
        return f"{root}{name}"
    return f"{root}/{name}"


def _build_target(
    meta_tree: Any,
    *,
    mesh: jax.sharding.Mesh | None,
    sharded: bool,
) -> dict[str, jax.ShapeDtypeStruct]:
    target: dict[str, jax.ShapeDtypeStruct] = {}
    for key, meta in _iter_metadata(meta_tree):
        shape, dtype = _meta_shape_dtype(meta)
        if sharded:
            if mesh is None:
                raise ValueError("mesh is required when sharded=True")
            spec = get_sharding_spec(key, len(shape))
            sharding = NamedSharding(mesh, spec)
            target[key] = jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
        else:
            target[key] = jax.ShapeDtypeStruct(shape, dtype)
    return target


def _restore_to_host(
    ckptr: ocp.StandardCheckpointer,
    checkpoint_path: str,
    meta_tree: Any,
) -> dict[str, Float[Array, "..."]]:
    cpu_device = jax.devices("cpu")[0]
    target = _build_target(meta_tree, mesh=None, sharded=False)
    with jax.default_device(cpu_device):
        return ckptr.restore(checkpoint_path, target)


def _empty_host_tree(meta_tree: Any) -> dict[str, np.ndarray]:
    host_tree: dict[str, np.ndarray] = {}
    for key, meta in _iter_metadata(meta_tree):
        shape, dtype = _meta_shape_dtype(meta)
        host_tree[key] = np.zeros(shape, dtype=dtype)
    return host_tree


def _restore_to_host_multihost(
    ckptr: ocp.StandardCheckpointer,
    checkpoint_path: str,
    meta_tree: Any,
    *,
    is_source: bool,
) -> dict[str, np.ndarray]:
    # Temporary fallback: each host restores directly to CPU memory to avoid
    # TPU HBM pressure from broadcast_one_to_all.
    return _restore_to_host(ckptr, checkpoint_path, meta_tree)


def _slice_for_host(
    value: np.ndarray,
    spec: PartitionSpec,
    *,
    host_index: int,
    host_count: int,
) -> np.ndarray:
    axes = tuple(spec)
    if "model" not in axes or host_count == 1:
        return value
    if axes.count("model") > 1:
        raise ValueError("Multiple 'model' axes not supported in host slicing.")
    axis = axes.index("model")
    dim = value.shape[axis]
    if dim % host_count != 0:
        raise ValueError(
            f"Axis {axis} with size {dim} is not divisible by host count {host_count}."
        )
    chunk = dim // host_count
    start = host_index * chunk
    end = start + chunk
    slices = [slice(None)] * value.ndim
    slices[axis] = slice(start, end)
    return value[tuple(slices)]


def _shard_to_mesh(
    params: dict[str, Float[Array, "..."]],
    mesh: jax.sharding.Mesh,
) -> dict[str, Float[Array, "..."]]:
    sharded: dict[str, Float[Array, "..."]] = {}
    is_multihost = jax.process_count() > 1
    host_index = jax.process_index()
    host_count = jax.process_count()
    for key, value in params.items():
        spec = get_sharding_spec(key, value.ndim)
        if is_multihost:
            local_value = _slice_for_host(
                np.asarray(value),
                spec,
                host_index=host_index,
                host_count=host_count,
            )
            sharded[key] = multihost_utils.host_local_array_to_global_array(
                local_value, mesh, spec
            )
        else:
            sharding = NamedSharding(mesh, spec)
            sharded[key] = jax.device_put(value, sharding)
    return sharded


def load_params(
    checkpoint_path: str,
    mesh: jax.sharding.Mesh | None = None,
    *,
    mesh_factory=None,
    host_first: bool = True,
    return_mesh: bool = False,
) -> (
    dict[str, Float[Array, "..."]]
    | tuple[dict[str, Float[Array, "..."]], jax.sharding.Mesh]
):
    ckptr = ocp.StandardCheckpointer()
    meta_tree = _unwrap_metadata(ckptr.metadata(checkpoint_path))

    if host_first:
        # For now, restore on every host to avoid full-tensor device_put during
        # multihost broadcast.
        cpu_state = _restore_to_host(ckptr, checkpoint_path, meta_tree)
        if mesh is None:
            if mesh_factory is None:
                mesh_factory = lambda: jax.sharding.Mesh(
                    jax.devices(), axis_names=("model",)
                )
            mesh = mesh_factory()
        params = _shard_to_mesh(cpu_state, mesh)
        del cpu_state
    else:
        if mesh is None:
            if mesh_factory is None:
                mesh_factory = lambda: jax.sharding.Mesh(
                    jax.devices(), axis_names=("model",)
                )
            mesh = mesh_factory()
        target = _build_target(meta_tree, mesh=mesh, sharded=True)
        params = ckptr.restore(checkpoint_path, target)

    if return_mesh:
        return params, mesh
    return params


def save_params(
    params: dict[str, Float[Array, "..."]],
) -> str:
    checkpoint_id = uuid.uuid4().hex
    checkpoint_path = _join_checkpoint_path(DEFAULT_GCS_SAVE_ROOT, checkpoint_id)
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(checkpoint_path, params)
    return checkpoint_path
