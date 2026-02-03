from __future__ import annotations

import gc
import json
import os
import tempfile
import uuid
from typing import Dict

import jax
import ml_dtypes  # noqa: F401  # ensures bfloat16 support when saving
import numpy as np
from google.cloud import storage
from safetensors.numpy import save_file

from core.gemma_forward import Params

# Default bucket/object to write when none is provided
DEFAULT_GCS_OUTPUT_PATH = (
    "gs://gemma-tpu-weights-us-west4-482802/model_stacked_pt.safetensors"
)
DEFAULT_GCS_SHARDED_OUTPUT_PATH = "gs://gemma-tpu-weights-us-west4-482802/sharded"


def save_params(params: Params) -> None:
    """
    Save model parameters as a stacked safetensors file and upload to GCS.

    Environment variable:
      GCS_OUTPUT_PATH: Required. Full gs://bucket/path/to/file.safetensors
                       If only a bucket is provided (gs://bucket), the object
                       name defaults to model_stacked_pt.safetensors.
    """
    gcs_path = os.environ.get("GCS_OUTPUT_PATH", DEFAULT_GCS_OUTPUT_PATH)
    if not gcs_path or not gcs_path.startswith("gs://"):
        raise ValueError("GCS_OUTPUT_PATH must be set to a gs://bucket/path")

    # Parse bucket and blob
    without_scheme = gcs_path[len("gs://") :]
    parts = without_scheme.split("/", 1)
    bucket_name = parts[0]
    blob_name = (
        parts[1] if len(parts) > 1 and parts[1] else "model_stacked_pt.safetensors"
    )
    if not bucket_name:
        raise ValueError("GCS_OUTPUT_PATH is missing a bucket name")

    # Add a UUID suffix to avoid overwriting existing checkpoints
    base, ext = os.path.splitext(blob_name)
    ext = ext or ".safetensors"
    blob_name = f"{base}-{uuid.uuid4().hex}{ext}"

    # Ensure params live on host and are numpy arrays
    host_params: Dict[str, np.ndarray] = jax.tree_util.tree_map(
        lambda x: np.asarray(jax.device_get(x)), params
    )

    # Write to a temp file then upload
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_file(host_params, tmp_path)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _is_gcs_path(path: str) -> bool:
    return path.startswith("gs://")


def _parse_gcs_path(path: str) -> tuple[str, str]:
    without_scheme = path[len("gs://") :]
    parts = without_scheme.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def _normalize_output_dir(path: str) -> str:
    if path.endswith(".safetensors") or path.endswith(".json"):
        return os.path.dirname(path)
    return path


def _shard_name(idx: int, total: int) -> str:
    return f"model-{idx:05d}-of-{total:05d}.safetensors"


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


def save_params_sharded(
    params: Params,
    output_path: str | None = None,
    shard_size_gb: float = 4.0,
) -> None:
    """
    Save model parameters in HF-style sharded safetensors format.

    This writes:
      - model-00001-of-000XX.safetensors, ...
      - model.safetensors.index.json

    The implementation never holds all weights on host at the same time.
    """
    if output_path is None:
        output_path = os.environ.get(
            "GCS_SHARDED_OUTPUT_PATH", DEFAULT_GCS_SHARDED_OUTPUT_PATH
        )

    output_path = _normalize_output_dir(output_path)
    is_gcs = _is_gcs_path(output_path)

    if is_gcs:
        bucket_name, prefix = _parse_gcs_path(output_path)
        if not bucket_name:
            raise ValueError("Sharded output path is missing a bucket name")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
    else:
        os.makedirs(output_path, exist_ok=True)
        prefix = ""

    shard_size_bytes = max(1, int(shard_size_gb * (1024**3)))
    shard_plan, total_size = _plan_shards(params, shard_size_bytes)
    num_shards = len(shard_plan)
    weight_map: dict[str, str] = {}

    for shard_idx, names in enumerate(shard_plan, start=1):
        shard_filename = _shard_name(shard_idx, num_shards)
        if is_gcs:
            local_path = os.path.join(tempfile.gettempdir(), shard_filename)
        else:
            local_path = os.path.join(output_path, shard_filename)

        shard_dict: Dict[str, np.ndarray] = {}
        for name in names:
            shard_dict[name] = np.asarray(jax.device_get(params[name]))
            weight_map[name] = shard_filename

        save_file(shard_dict, local_path)

        shard_dict.clear()
        gc.collect()

        if is_gcs:
            object_name = f"{prefix}/{shard_filename}" if prefix else shard_filename
            bucket.blob(object_name).upload_from_filename(local_path)
            os.remove(local_path)

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    index_filename = "model.safetensors.index.json"

    if is_gcs:
        index_path = os.path.join(tempfile.gettempdir(), index_filename)
    else:
        index_path = os.path.join(output_path, index_filename)

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    if is_gcs:
        object_name = f"{prefix}/{index_filename}" if prefix else index_filename
        bucket.blob(object_name).upload_from_filename(index_path)
        os.remove(index_path)
