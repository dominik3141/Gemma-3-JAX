from __future__ import annotations

import os
import tempfile
import uuid
from typing import Dict

import jax
import ml_dtypes  # noqa: F401  # ensures bfloat16 support when saving
import numpy as np
from google.cloud import storage
from safetensors.numpy import save_file

from gemma_forward import Params

# Default bucket/object to write when none is provided
DEFAULT_GCS_OUTPUT_PATH = "gs://gemma-tpu-weights-us-west4-482802/model_stacked_pt.safetensors"


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
    blob_name = parts[1] if len(parts) > 1 and parts[1] else "model_stacked_pt.safetensors"
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
