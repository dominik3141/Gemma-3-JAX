from functools import partial
import logging

import jax
import optax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from core.supervised_train import LEARNING_RATE, train_loop
from utils.params_io_27b import DEFAULT_ORBAX_CHECKPOINT, load_params
from utils.params_io_27b import muon_weight_dimension_numbers_for_27b

LOGGER = logging.getLogger(__name__)


def make_mesh() -> Mesh:
    num_devices: int = jax.device_count()
    model_axis_size = 16
    batch_axis_size = num_devices // model_axis_size
    device_mesh = mesh_utils.create_device_mesh((batch_axis_size, model_axis_size))
    return Mesh(device_mesh, axis_names=("batch", "model"))


def main(num_batches: int = 100):
    LOGGER.info("--- entrypoints.supervised.main() started ---")
    key = jax.random.key(42)

    LOGGER.info("Loading weights...")

    mesh = make_mesh()
    params = load_params(DEFAULT_ORBAX_CHECKPOINT, mesh)
    LOGGER.info("Weights loaded.")

    data_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    batch_shard_size = mesh.devices.size
    batch_size = batch_shard_size * 2
    keys = jax.random.split(key, batch_shard_size * num_batches)

    optimizer_state = optax.contrib.muon(
        learning_rate=LEARNING_RATE,
        muon_weight_dimension_numbers=muon_weight_dimension_numbers_for_27b,
    ).init(params)
    (params, _), losses = jax.lax.scan(
        partial(train_loop, data_sharding, batch_size),
        (params, optimizer_state),
        keys,
    )

    LOGGER.info("XLA returned control")
    LOGGER.info("Losses: %s", losses)

    return params


if __name__ == "__main__":
    main()
