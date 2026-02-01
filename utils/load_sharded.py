import os
import json
import re
import typing
import jax
import jax.numpy as jnp
import numpy
import safetensors
import jax.experimental.mesh_utils

# --- 1. Sharding Logic ---


def get_stacked_sharding_spec(
    name: str, is_stacked: bool = False
) -> jax.sharding.PartitionSpec:
    """
    Returns the PartitionSpec.
    If is_stacked is True, we prepend a 'None' axis for the layer dimension.

    Standard Tensor Parallelism (TP) Logic for Gemma/Llama:
    - Weights are (In, Out).
    - Column Linear (q, k, v, gate, up): Shard Axis 1 (Out).
    - Row Linear (o, down): Shard Axis 0 (In).
    - Embeddings: Shard Axis 1 (Model Dim) or 0 (Vocab) depending on pref.
      Here we usually shard Vocab (Axis 0) for large vocabs, or Model (Axis 1).
      Let's assume standard Megatron-style: Shard Vocab (Axis 0) for input,
      Shard Embedding (Axis 1) for output.
    """

    # Helper to prepend None for the layer dimension if stacked
    def make_spec(*dims):
        if is_stacked:
            return jax.sharding.PartitionSpec(None, *dims)
        return jax.sharding.PartitionSpec(*dims)

    if "embed_tokens" in name:
        # (Vocab, Embed) -> Shard Vocab for TP usually, or Embed.
        # For simple TP, often strictly model dim is safer to avoid vocab padding issues.
        return make_spec(None, "model")

    # Attention Projections
    elif "q_proj" in name or "k_proj" in name or "v_proj" in name:
        # Shape: (Hidden, Heads*HeadDim) -> Shard Output (Axis 1)
        return make_spec(None, "model")
    elif "o_proj" in name:
        # Shape: (Heads*HeadDim, Hidden) -> Shard Input (Axis 0)
        return make_spec("model", None)

    # MLP Projections
    elif "gate_proj" in name or "up_proj" in name:
        # Shape: (Hidden, Intermediate) -> Shard Output (Axis 1)
        return make_spec(None, "model")
    elif "down_proj" in name:
        # Shape: (Intermediate, Hidden) -> Shard Input (Axis 0)
        return make_spec("model", None)

    # Norms and Biases (usually small, replicate them)
    else:
        return make_spec(None, None) if is_stacked else jax.sharding.PartitionSpec(None)


# --- 2. Virtual Loading Callbacks ---


def create_stacked_callback(
    base_dir: str, layer_map: dict[int, str], param_suffix: str, dtype: str
) -> typing.Callable[[tuple[slice, ...]], numpy.ndarray]:
    """
    Reads multiple files to construct a single stacked tensor slice on the fly.
    """

    def callback(index: tuple[slice, ...]) -> numpy.ndarray:
        # index[0] is the slice along the new "Layers" dimension
        layer_slice = index[0]

        # Handle standard python slicing logic
        start = layer_slice.start if layer_slice.start is not None else 0
        stop = layer_slice.stop if layer_slice.stop is not None else len(layer_map)
        step = layer_slice.step if layer_slice.step is not None else 1

        layer_indices = range(start, stop, step)
        stacked_data = []

        # Inner dims are passed directly to the file reader
        inner_index = index[1:]

        for i in layer_indices:
            filename = layer_map[i]
            filepath = os.path.join(base_dir, filename)

            # Reconstruct the physical key: "model.layers.0.self_attn.q_proj.weight"
            # Note: We assume the prefix is standard.
            flat_key = f"model.layers.{i}.{param_suffix}"

            with safetensors.safe_open(filepath, framework="numpy", device="cpu") as f:
                tensor_data = f.get_slice(flat_key)[inner_index]
                stacked_data.append(tensor_data)

        if not stacked_data:
            return numpy.array([], dtype=getattr(numpy, dtype, numpy.float32))

        # Stack along axis 0
        return numpy.stack(stacked_data, axis=0)

    return callback


def create_passthrough_callback(
    filepath: str, tensor_name: str, dtype: str
) -> typing.Callable[[tuple[slice, ...]], numpy.ndarray]:
    """
    Standard callback for non-stacked weights (embeddings, final norm).
    """

    def callback(index: tuple[slice, ...]) -> numpy.ndarray:
        with safetensors.safe_open(filepath, framework="numpy", device="cpu") as f:
            slice_data = f.get_slice(tensor_name)[index]
        return slice_data

    return callback


# --- 3. Main Logic ---


def load_stacked_sharded_model(
    model_path: str, mesh: jax.sharding.Mesh
) -> dict[str, jax.Array]:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index_data = json.load(f)

    weight_map = index_data["weight_map"]

    # 1. Parse Structure
    # We need to separate "Global" weights from "Layer" weights
    # and map layer indices to files.
    layer_pattern = re.compile(r"model\.layers\.(\d+)\.(.+)$")

    layer_map: dict[int, str] = {}  # {0: "model-00001.safetensors", ...}
    layer_suffixes: set[str] = set()  # {"self_attn.q_proj.weight", ...}
    global_weights: dict[str, str] = {}  # {"model.embed_tokens.weight": "file1"}

    max_layer_idx = -1

    for key, filename in weight_map.items():
        match = layer_pattern.search(key)
        if match:
            idx = int(match.group(1))
            suffix = match.group(2)
            layer_map[idx] = filename
            layer_suffixes.add(suffix)
            max_layer_idx = max(max_layer_idx, idx)
        else:
            global_weights[key] = filename

    num_layers = max_layer_idx + 1
    print(f"Detected {num_layers} layers and {len(global_weights)} global tensors.")

    params: dict[str, jax.Array] = {}

    # 2. Process Stacked Layers
    # We need to peek at Layer 0 to get the shape/dtype for the whole stack
    first_layer_file = os.path.join(model_path, layer_map[0])

    # Optimization: Open Layer 0 file once to read all shapes
    with safetensors.safe_open(
        first_layer_file, framework="numpy", device="cpu"
    ) as f_peek:
        for suffix in layer_suffixes:
            # Construct key for layer 0
            key_0 = f"model.layers.0.{suffix}"

            # Get metadata
            slice_0 = f_peek.get_slice(key_0)
            shape_0 = slice_0.get_shape()
            # safetensors doesn't expose dtype string easily in slice, assume BF16 for Gemma
            dtype_str = "bfloat16"
            jax_dtype = jnp.bfloat16

            # Create Stacked Spec
            # New Shape: (Layers, Dim0, Dim1)
            stacked_shape = (num_layers,) + shape_0
            spec = get_stacked_sharding_spec(suffix, is_stacked=True)
            sharding = jax.sharding.NamedSharding(mesh, spec)

            abstract_array = jax.ShapeDtypeStruct(
                stacked_shape, jax_dtype, sharding=sharding
            )

            # Create Virtual Callback
            cb = create_stacked_callback(model_path, layer_map, suffix, dtype_str)

            # New Key Name: "layers_stacked"
            new_key = f"model.layers_stacked.{suffix}"
            params[new_key] = jax.make_array_from_callback(
                abstract_array.shape, abstract_array.sharding, cb
            )

    # 3. Process Global Weights (Non-Stacked)
    for key, filename in global_weights.items():
        filepath = os.path.join(model_path, filename)

        # We need shape for this specific global weight
        with safetensors.safe_open(filepath, framework="numpy", device="cpu") as f:
            slice_g = f.get_slice(key)
            shape = slice_g.get_shape()

        dtype_str = "bfloat16"
        jax_dtype = jnp.bfloat16

        spec = get_stacked_sharding_spec(key, is_stacked=False)
        sharding = jax.sharding.NamedSharding(mesh, spec)

        abstract_array = jax.ShapeDtypeStruct(shape, jax_dtype, sharding=sharding)
        cb = create_passthrough_callback(filepath, key, dtype_str)

        params[key] = jax.make_array_from_callback(
            abstract_array.shape, abstract_array.sharding, cb
        )

    return params


# --- 4. Execution Block ---

if __name__ == "__main__":
    # Example setup for 4 v6e TPUs
    devices = jax.experimental.mesh_utils.create_device_mesh((4,))
    mesh = jax.sharding.Mesh(devices, axis_names=("model",))

    # Mounted by setup.py via gcsfuse.
    GCS_PATH = "data/gemma-3-27b"

    with mesh:
        # Returns a dict where layers are stacked
        params = load_stacked_sharded_model(GCS_PATH, mesh)

    # Validation
    for k, v in params.items():
        # Print only a few to verify
        if "layers_stacked" in k and "q_proj" in k:
            print(f"Key: {k}")
            print(f"  Global Shape: {v.shape}")
            print(f"  Sharding: {v.sharding}")
            break
