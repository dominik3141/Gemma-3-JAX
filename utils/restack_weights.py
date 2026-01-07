from __future__ import annotations

import re
from typing import Dict, Tuple

import numpy as np
import ml_dtypes  # noqa: F401  (needed so numpy can represent bfloat16)
from safetensors import safe_open
from safetensors.numpy import save_file


_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def repack_layers_to_stacked_safetensors(
    in_path: str,
    out_path: str,
    *,
    layers_out_prefix: str = "model.layers_stacked.",  # e.g. model.layers_stacked.self_attn.q_proj.weight
    keep_original_layer_tensors: bool = False,
    strict: bool = True,
) -> None:
    """
    Offline one-time conversion:
      - Reads a safetensors file with per-layer keys like:
            model.layers.{i}.self_attn.q_proj.weight
      - Stacks all layer tensors for the same suffix into one tensor with shape (L, ...).
      - Writes a new safetensors file containing:
            - all non-layer tensors copied as-is (model.embed_tokens.weight, model.norm.weight, ...)
            - stacked tensors under `layers_out_prefix + suffix`
            - optionally, the original per-layer tensors too (if keep_original_layer_tensors=True)

    Notes:
      - This does allocate new stacked buffers (inevitable when going from many tensors -> one tensor).
      - We preserve dtype as stored (e.g. bfloat16 stays bfloat16).
      - Layer indices can be non-sorted in file; we sort numerically and require contiguity by default.

    Parameters:
      strict:
        If True:
          - require layers are contiguous [0..L-1]
          - require every suffix exists for every layer
          - require shapes/dtypes match across layers for each suffix
        If False:
          - stacks only layers present for a suffix (still sorted); may produce ragged-missing error if shapes differ.
    """
    # Read all tensors as numpy arrays (views into mmapped file where possible)
    with safe_open(in_path, framework="np", device="cpu") as f:
        keys = list(f.keys())

        nonlayer: Dict[str, np.ndarray] = {}
        per_layer: Dict[Tuple[int, str], np.ndarray] = {}
        suffixes: set[str] = set()
        layer_ids: set[int] = set()

        for k in keys:
            m = _LAYER_RE.match(k)
            if m is None:
                nonlayer[k] = f.get_tensor(k)
                continue
            layer_id = int(m.group(1))
            suffix = m.group(2)
            per_layer[(layer_id, suffix)] = f.get_tensor(k)
            suffixes.add(suffix)
            layer_ids.add(layer_id)

    if not layer_ids:
        raise ValueError(
            f"No layer tensors found in {in_path} matching 'model.layers.<i>.<suffix>'"
        )

    sorted_layers = sorted(layer_ids)
    if strict:
        expected = list(range(sorted_layers[0], sorted_layers[-1] + 1))
        if sorted_layers != expected or sorted_layers[0] != 0:
            raise ValueError(
                f"Layer ids are not contiguous starting at 0. Found: {sorted_layers[:10]}... "
                f"(min={sorted_layers[0]}, max={sorted_layers[-1]})"
            )

    L = sorted_layers[-1] + 1 if strict else len(sorted_layers)

    out_tensors: Dict[str, np.ndarray] = dict(nonlayer)

    # Build stacked tensors per suffix
    for suffix in sorted(suffixes):
        arrays = []
        ref_shape = None
        ref_dtype = None

        for lid in sorted_layers:
            key = (lid, suffix)
            if key not in per_layer:
                if strict:
                    raise ValueError(
                        f"Missing tensor for layer {lid} suffix '{suffix}'"
                    )
                else:
                    continue

            arr = per_layer[key]
            if ref_shape is None:
                ref_shape = arr.shape
                ref_dtype = arr.dtype
            else:
                if strict and (arr.shape != ref_shape or arr.dtype != ref_dtype):
                    raise ValueError(
                        f"Inconsistent shape/dtype for suffix '{suffix}': "
                        f"expected {ref_shape}/{ref_dtype}, got {arr.shape}/{arr.dtype} at layer {lid}"
                    )
            arrays.append(arr)

        if strict and len(arrays) != L:
            raise ValueError(
                f"Suffix '{suffix}' did not have L={L} tensors; got {len(arrays)}"
            )

        stacked = np.stack(arrays, axis=0)  # (L, ...) new contiguous buffer
        out_tensors[layers_out_prefix + suffix] = stacked

    if keep_original_layer_tensors:
        for (lid, suffix), arr in per_layer.items():
            out_tensors[f"model.layers.{lid}.{suffix}"] = arr

    save_file(out_tensors, out_path)


# Example usage:
if __name__ == "__main__":
    repack_layers_to_stacked_safetensors(
        in_path="model.safetensors",
        out_path="model_stacked.safetensors",
        layers_out_prefix="model.layers_stacked.",
        keep_original_layer_tensors=False,
        strict=True,
    )
    print("Wrote model_stacked.safetensors")
