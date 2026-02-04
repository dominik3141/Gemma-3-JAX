from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import uuid
from typing import Dict

import jax
import ml_dtypes  # noqa: F401  # ensures bfloat16 support when saving
import numpy as np
from safetensors.numpy import save_file

from core.gemma_forward import Params

CHECKPOINT_ROOT = "checkpoints"
GCS_UPLOAD_PREFIX = "gs://gemma-3-checkpoints-482802/checkpoints"
SHARD_THRESHOLD_BYTES = 4 * 1024**3


def _run(cmd: list[str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _ensure_gcloud_storage() -> None:
    if shutil.which("gcloud"):
        return
    raise RuntimeError("gcloud CLI not found on PATH; cannot upload to GCS")


def _tensor_nbytes(x: jax.Array) -> int:
    return int(x.size) * int(np.dtype(x.dtype).itemsize)


def _plan_shards(params: Params, shard_size_bytes: int) -> tuple[list[list[str]], int]:
    names = sorted(params.keys())
    shard_plan: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    total_size = 0

    for name in names:
        size = _tensor_nbytes(params[name])
        total_size += size

        if current and current_size + size > shard_size_bytes:
            shard_plan.append(current)
            current = []
            current_size = 0

        current.append(name)
        current_size += size

        if size > shard_size_bytes:
            shard_plan.append(current)
            current = []
            current_size = 0

    if current:
        shard_plan.append(current)

    return shard_plan, total_size


def _write_single_file(params: Params, out_dir: str) -> None:
    host_params: Dict[str, np.ndarray] = jax.tree_util.tree_map(
        lambda x: np.asarray(jax.device_get(x)), params
    )
    out_path = os.path.join(out_dir, "model_stacked_pt.safetensors")
    save_file(host_params, out_path)


def _write_sharded(params: Params, out_dir: str, shard_size_bytes: int) -> None:
    shard_plan, total_size = _plan_shards(params, shard_size_bytes)
    num_shards = len(shard_plan)
    weight_map: dict[str, str] = {}

    for shard_idx, names in enumerate(shard_plan, start=1):
        shard_filename = f"model-{shard_idx:05d}-of-{num_shards:05d}.safetensors"
        local_path = os.path.join(out_dir, shard_filename)

        shard_dict: Dict[str, np.ndarray] = {}
        for name in names:
            shard_dict[name] = np.asarray(jax.device_get(params[name]))
            weight_map[name] = shard_filename

        save_file(shard_dict, local_path)

        shard_dict.clear()
        gc.collect()

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    index_path = os.path.join(out_dir, "model.safetensors.index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)


def _total_params_bytes(params: Params) -> int:
    return int(sum(_tensor_nbytes(v) for v in params.values()))


def save_params(params: Params, upload_to_gcs: bool) -> None:
    """
    Save model parameters to a local checkpoint directory.

    This automatically shards based on total size and optionally uploads to GCS.
    """
    checkpoint_id = uuid.uuid4().hex
    out_dir = os.path.join(CHECKPOINT_ROOT, checkpoint_id)
    os.makedirs(out_dir, exist_ok=True)

    total_size = _total_params_bytes(params)
    if total_size <= SHARD_THRESHOLD_BYTES:
        _write_single_file(params, out_dir)
    else:
        _write_sharded(params, out_dir, SHARD_THRESHOLD_BYTES)

    if upload_to_gcs:
        _ensure_gcloud_storage()
        dest = f"{GCS_UPLOAD_PREFIX}/{checkpoint_id}"
        _run(["gcloud", "storage", "cp", "-r", out_dir, dest])
