from functools import partial

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from core.supervised_train import train_loop
from utils.params_io_27b import DEFAULT_ORBAX_CHECKPOINT, load_params


def main(num_batches: int = 100):
    print("--- entrypoints.supervised.main() started ---")
    key = jax.random.key(42)

    print("Loading weights...")

    def _make_mesh() -> Mesh:
        num_devices: int = jax.device_count()
        model_axis_size = min(num_devices, 2)
        batch_axis_size = num_devices // model_axis_size
        device_mesh = mesh_utils.create_device_mesh((batch_axis_size, model_axis_size))
        return Mesh(device_mesh, axis_names=("batch", "model"))

    mesh = _make_mesh()
    params = load_params(DEFAULT_ORBAX_CHECKPOINT, mesh)
    print("Weights loaded.")

    data_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    num_devices = mesh.devices.size
    batch_size = num_devices * 2
    keys = jax.random.split(key, num_devices * num_batches)

    params, losses = jax.lax.scan(
        partial(train_loop, data_sharding, batch_size),
        params,
        keys,
    )

    print("XLA returned control")
    print(losses)

    return params


if __name__ == "__main__":
    main()
