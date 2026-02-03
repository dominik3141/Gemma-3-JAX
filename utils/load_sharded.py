import os
import json
import re
import typing

import jax
import jax.numpy as jnp
import numpy
import safetensors
import jax.experimental.mesh_utils
from collections import defaultdict
import time

# --- 1. Sharding Logic ---


def _normalize_spec_for_rank(
    spec: jax.sharding.PartitionSpec,
    rank: int,
) -> jax.sharding.PartitionSpec:
    axes = tuple(spec)
    if len(axes) < rank:
        axes = axes + (None,) * (rank - len(axes))
    elif len(axes) > rank:
        axes = axes[:rank]
    return jax.sharding.PartitionSpec(*axes)


def get_stacked_sharding_spec(
    name: str,
    rank: int,
    is_stacked: bool = False,
) -> jax.sharding.PartitionSpec:
    """
    Returns the PartitionSpec for a given weight name.
    """

    # Helper to prepend None for the layer dimension if stacked
    def make_spec(*dims):
        if is_stacked:
            return jax.sharding.PartitionSpec(None, *dims)
        return jax.sharding.PartitionSpec(*dims)

    if "embed_tokens" in name:
        spec = make_spec(None, "model")

    elif "q_proj" in name or "k_proj" in name or "v_proj" in name:
        spec = make_spec(None, "model")  # Shard Head Dim
    elif "o_proj" in name:
        spec = make_spec("model", None)  # Shard Head Dim

    elif "gate_proj" in name or "up_proj" in name:
        spec = make_spec(None, "model")  # Shard Hidden
    elif "down_proj" in name:
        spec = make_spec("model", None)  # Shard Hidden

    else:
        spec = make_spec(None) if is_stacked else jax.sharding.PartitionSpec(None)

    return _normalize_spec_for_rank(spec, rank)


# --- 2. Virtual Loading Callbacks ---


def create_stacked_callback(
    base_dir: str,
    weight_map: dict[str, str],
    key_template: str,
    num_layers: int,
    dtype: str,
) -> typing.Callable[[tuple[slice, ...]], numpy.ndarray]:
    """
    Reads multiple files to construct a single stacked tensor slice on the fly.
    """

    def callback(index: tuple[slice, ...]) -> numpy.ndarray:
        layer_slice = index[0]

        start = layer_slice.start if layer_slice.start is not None else 0
        step = layer_slice.step if layer_slice.step is not None else 1

        stop = layer_slice.stop if layer_slice.stop is not None else num_layers

        layer_indices = range(start, stop, step)
        stacked_data = []

        inner_index = index[1:]

        for i in layer_indices:
            full_key = key_template.format(i)

            if full_key not in weight_map:
                raise KeyError(
                    f"Key {full_key} not found in weight map during loading."
                )

            filename = weight_map[full_key]
            filepath = os.path.join(base_dir, filename)

            with safetensors.safe_open(filepath, framework="numpy", device="cpu") as f:
                tensor_data = f.get_slice(full_key)[inner_index]
                stacked_data.append(tensor_data)

        if not stacked_data:
            return numpy.array([], dtype=getattr(numpy, dtype, numpy.float32))

        return numpy.stack(stacked_data, axis=0)

    return callback


def create_passthrough_callback(
    filepath: str, tensor_name: str, dtype: str
) -> typing.Callable[[tuple[slice, ...]], numpy.ndarray]:
    def callback(index: tuple[slice, ...]) -> numpy.ndarray:
        with safetensors.safe_open(filepath, framework="numpy", device="cpu") as f:
            slice_data = f.get_slice(tensor_name)[index]
        return slice_data

    return callback


# --- 3. Main Logic ---


def load_stacked_sharded_model(
    model_path: str,
    mesh: jax.sharding.Mesh,
    max_layers: int | None = None,
    include_stacked: bool = True,
    include_global: bool = True,
) -> dict[str, jax.Array]:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index_data = json.load(f)

    weight_map = index_data["weight_map"]

    # Robust Regex
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

    params: dict[str, jax.Array] = {}

    # 2. Process Stacked Layers
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

                first_idx = min(indices)
                key_peek = key_template.format(first_idx)

                if key_peek not in weight_map:
                    print(f"  [Warn] Missing {key_peek}, skipping.")
                    continue

                file_peek = weight_map[key_peek]
                path_peek = os.path.join(model_path, file_peek)

                with safetensors.safe_open(
                    path_peek, framework="numpy", device="cpu"
                ) as f_peek:
                    slice_peek = f_peek.get_slice(key_peek)
                    # FIX: safetensors returns a list, cast to tuple for concatenation
                    shape_peek = tuple(slice_peek.get_shape())

                    dtype_str = "bfloat16"
                    jax_dtype = jnp.bfloat16

                # Now this concatenation is safe: (tuple) + (tuple)
                stacked_shape = (num_layers,) + shape_peek

                spec = get_stacked_sharding_spec(
                    suffix, len(stacked_shape), is_stacked=True
                )
                sharding = jax.sharding.NamedSharding(mesh, spec)

                abstract_array = jax.ShapeDtypeStruct(
                    stacked_shape, jax_dtype, sharding=sharding
                )

                cb = create_stacked_callback(
                    model_path, weight_map, key_template, num_layers, dtype_str
                )

                new_key = f"{prefix}layers_stacked.{suffix}"

                params[new_key] = jax.make_array_from_callback(
                    abstract_array.shape, abstract_array.sharding, cb
                )

    # 3. Process Global Weights
    if include_global:
        print(f"Processing {len(global_weights)} global weights...")
        for key, filename in global_weights.items():
            filepath = os.path.join(model_path, filename)

            with safetensors.safe_open(filepath, framework="numpy", device="cpu") as f:
                slice_g = f.get_slice(key)
                shape = tuple(slice_g.get_shape())  # Cast here too for consistency

            dtype_str = "bfloat16"
            jax_dtype = jnp.bfloat16

            spec = get_stacked_sharding_spec(key, len(shape), is_stacked=False)
            sharding = jax.sharding.NamedSharding(mesh, spec)

            abstract_array = jax.ShapeDtypeStruct(shape, jax_dtype, sharding=sharding)
            cb = create_passthrough_callback(filepath, key, dtype_str)

            params[key] = jax.make_array_from_callback(
                abstract_array.shape, abstract_array.sharding, cb
            )

    return params


# --- 4. Execution Block ---

if __name__ == "__main__":
    # Example setup for 4 devices
    print("Initializing device mesh...", flush=True)
    devices = jax.experimental.mesh_utils.create_device_mesh((4,))
    mesh = jax.sharding.Mesh(devices, axis_names=("model",))
    print(f"Device mesh ready: {len(devices)} devices", flush=True)

    # Test-only: read from local SSD copy.
    GCS_PATH = "data/gemma-3-27b"
    max_layers_env = os.getenv("MAX_LAYERS")
    max_layers = int(max_layers_env) if max_layers_env else None
    timed_subset = os.getenv("TIMED_SUBSET") is not None
    subset_layers_env = os.getenv("SUBSET_LAYERS")
    subset_layers = int(subset_layers_env) if subset_layers_env else 1
    do_calc = os.getenv("DO_CALC") is not None

    try:
        with mesh:
            if timed_subset:
                print(f"Timing subset load: {subset_layers} layer(s)", flush=True)
                t0 = time.perf_counter()
                _ = load_stacked_sharded_model(
                    GCS_PATH,
                    mesh,
                    max_layers=subset_layers,
                    include_global=False,
                )
                subset_time = time.perf_counter() - t0

                print("Timing global weights...", flush=True)
                t1 = time.perf_counter()
                _ = load_stacked_sharded_model(
                    GCS_PATH,
                    mesh,
                    include_stacked=False,
                    include_global=True,
                )
                global_time = time.perf_counter() - t1

                # Estimate total layers for the main language model stack
                index_path = os.path.join(GCS_PATH, "model.safetensors.index.json")
                with open(index_path, "r") as f:
                    index_data = json.load(f)
                weight_map = index_data["weight_map"]
                layer_pattern = re.compile(r"^(.*?)layers\.(\d+)\.(.+)$")
                max_layer = -1
                for key in weight_map.keys():
                    m = layer_pattern.match(key)
                    if not m:
                        continue
                    prefix = m.group(1)
                    if prefix != "language_model.model.":
                        continue
                    idx = int(m.group(2))
                    if idx > max_layer:
                        max_layer = idx
                total_layers = max_layer + 1 if max_layer >= 0 else 0

                est_stacked = subset_time * (total_layers / max(1, subset_layers))
                est_total = est_stacked + global_time

                print(
                    f"Subset time (stacked, {subset_layers}): {subset_time:.2f}s",
                    flush=True,
                )
                print(f"Globals time: {global_time:.2f}s", flush=True)
                print(
                    f"Estimated stacked total ({total_layers} layers): {est_stacked:.2f}s",
                    flush=True,
                )
                print(f"Estimated full load: {est_total:.2f}s", flush=True)
                params = {}
            else:
                print("Loading sharded model...", flush=True)
                params = load_stacked_sharded_model(
                    GCS_PATH, mesh, max_layers=max_layers
                )

        print("Success: Model weights are resident on TPUs.")

        if do_calc and params:
            print("Running post-load calculations...", flush=True)
            calc_keys = []
            for k in params.keys():
                calc_keys.append(k)
                if len(calc_keys) >= 2:
                    break
            for k in calc_keys:
                t0 = time.perf_counter()
                x = params[k]
                y = jnp.sum(jnp.square(x))
                y.block_until_ready()
                dt = time.perf_counter() - t0
                print(f"Calc {k}: {dt:.4f}s", flush=True)

        # Verification
        for k, v in params.items():
            if "layers_stacked" in k and "q_proj" in k:
                print(f"Key: {k}")
                print(f"  Shape: {v.shape}")
                print(f"  Sharding: {v.sharding}")
                break

    except Exception as e:
        print(f"Failed: {e}")
        import traceback

        traceback.print_exc()
