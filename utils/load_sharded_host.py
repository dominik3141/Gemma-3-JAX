import json
import os
import re
from collections import defaultdict
from typing import Iterable

import numpy as np
import safetensors


def _normalize_spec_for_rank(spec: tuple, rank: int) -> tuple:
    if len(spec) < rank:
        spec = spec + (None,) * (rank - len(spec))
    elif len(spec) > rank:
        spec = spec[:rank]
    return spec


def get_stacked_sharding_spec_host(
    name: str, rank: int, is_stacked: bool = False
) -> tuple:
    def make_spec(*dims):
        if is_stacked:
            return (None,) + dims
        return dims

    if "embed_tokens" in name:
        spec = make_spec(None, "model")
    elif "q_proj" in name or "k_proj" in name or "v_proj" in name:
        spec = make_spec(None, "model")
    elif "o_proj" in name:
        spec = make_spec("model", None)
    elif "gate_proj" in name or "up_proj" in name:
        spec = make_spec(None, "model")
    elif "down_proj" in name:
        spec = make_spec("model", None)
    else:
        spec = make_spec(None) if is_stacked else (None,)

    return _normalize_spec_for_rank(spec, rank)


def _load_tensor_host(filepath: str, tensor_name: str) -> np.ndarray:
    with safetensors.safe_open(filepath, framework="numpy", device="cpu") as f:
        return f.get_tensor(tensor_name)


def _load_stacked_tensor_host(
    base_dir: str,
    weight_map: dict[str, str],
    key_template: str,
    num_layers: int,
    indices: Iterable[int],
) -> np.ndarray:
    layer_indices = [i for i in sorted(indices) if i < num_layers]
    stacked = None

    for i in layer_indices:
        full_key = key_template.format(i)
        filename = weight_map.get(full_key)
        if not filename:
            raise KeyError(f"Key {full_key} not found in weight map during loading.")
        filepath = os.path.join(base_dir, filename)
        tensor = _load_tensor_host(filepath, full_key)

        if stacked is None:
            stacked = np.empty((num_layers,) + tensor.shape, dtype=tensor.dtype)

        stacked[i] = tensor

    if stacked is None:
        return np.array([], dtype=np.float32)

    return stacked


def load_stacked_sharded_model_host(
    model_path: str,
    max_layers: int | None = None,
    include_stacked: bool = True,
    include_global: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, tuple]]:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    weight_map = index_data["weight_map"]
    layer_pattern = re.compile(r"^(.*?)layers\.(\d+)\.(.+)$")

    stacks = defaultdict(lambda: defaultdict(set))
    global_weights = {}

    for key, filename in weight_map.items():
        match = layer_pattern.match(key)
        if match:
            prefix = match.group(1)
            idx = int(match.group(2))
            suffix = match.group(3)
            stacks[prefix][suffix].add(idx)
        else:
            global_weights[key] = filename

    print(f"Detected {len(stacks)} stack groups (e.g. language_model, vision_tower).")

    host_params: dict[str, np.ndarray] = {}
    sharding_specs: dict[str, tuple] = {}

    if include_stacked:
        for prefix, suffixes in stacks.items():
            print(f"Processing stack: '{prefix}layers...'")

            for suffix, indices in suffixes.items():
                num_layers = max(indices) + 1
                if max_layers is not None:
                    num_layers = min(num_layers, max_layers)
                if num_layers <= 0:
                    continue

                key_template = f"{prefix}layers.{{}}.{suffix}"
                stacked = _load_stacked_tensor_host(
                    model_path, weight_map, key_template, num_layers, indices
                )
                if stacked.size == 0:
                    continue

                spec = get_stacked_sharding_spec_host(
                    suffix, stacked.ndim, is_stacked=True
                )
                new_key = f"{prefix}layers_stacked.{suffix}"
                host_params[new_key] = stacked
                sharding_specs[new_key] = spec

    if include_global:
        print(f"Processing {len(global_weights)} global weights...")
        for key, filename in global_weights.items():
            filepath = os.path.join(model_path, filename)
            tensor = _load_tensor_host(filepath, key)
            spec = get_stacked_sharding_spec_host(key, tensor.ndim, is_stacked=False)
            host_params[key] = tensor
            sharding_specs[key] = spec

    return host_params, sharding_specs
