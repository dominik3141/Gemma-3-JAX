r"""
We will be using our nice prefill optimized forward function to do some supervised
finetraining of Gemma.

Plan:
1.  Take sequence from training data
2.  Encode to tokens
3.  Calculate loss
    - Simple forward on input tokens
    - Loss defined as | x_n - x_{n-1} | with an appropriate norm
4.  Take \nabla loss
5.  Adjust parameters (let's start with SGD)

We might have to adjust our forward function to also take a mask so that we can
properly deal with batches of different input lengths.
But for now we just always take sequences of the same length to avoid this problem.
"""

import jax
import os

# init distributed training communications (blocking)
# jax.distributed.initialize()  # must happen before the train_data import, therefore moved to top

import jax.numpy as jnp
from gemma_forward import forward
from inspect_weights import load_weights_as_dict
import optax
from gemma_forward import Params
from sft_data import get_training_batch
from functools import partial


def loss_fn(xs, params) -> jax.Array:
    predictions = forward(xs, params)
    input_len = xs.shape[0]
    predictions = predictions[: input_len - 1]  # remove padding
    labels = xs[1:]
    loss = optax.softmax_cross_entropy_with_integer_labels(predictions, labels)
    return jnp.mean(loss)


def SGD(params, grads, lr) -> Params:
    return jax.tree_util.tree_map(lambda param, grad: param - lr * grad, params, grads)


def train(
    key, batch_size, params, seq_length, lr, data_sharding
) -> tuple[Params, jax.Array]:
    def loss_batched(xss, params) -> jax.Array:
        return jnp.mean(jax.vmap(loss_fn, in_axes=(0, None))(xss, params))

    train_data = get_training_batch(key, batch_size, seq_length)
    train_data = jax.lax.with_sharding_constraint(train_data, data_sharding)

    loss, grads = jax.value_and_grad(loss_batched, argnums=1)(train_data, params)
    return SGD(params, grads, lr), loss


def train_loop(data_sharding, batch_size, params, key) -> tuple[Params, jax.Array]:
    new_params, loss = train(key, batch_size, params, 1024, 0.01, data_sharding)

    return new_params, loss


from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils


# TESTING
def main(num_batches=100):
    # keys and parameters
    key = jax.random.key(42)
    params = load_weights_as_dict("model_stacked_pt.safetensors")

    # Distributed training
    num_devices: int = jax.device_count()
    print(f"Number of devices: {num_devices}")
    print(f"Devices: {jax.devices()}")
    print(f"Local devices: {jax.local_devices()}")
    print(f"Backend: {jax.default_backend()}")

    device_mesh = mesh_utils.create_device_mesh((num_devices,))
    mesh = Mesh(device_mesh, axis_names=("batch",))
    data_sharding = NamedSharding(mesh, P("batch"))
    param_sharding = NamedSharding(mesh, P())

    # ensure parameters are replicated across all cores
    params = jax.device_put(params, param_sharding)

    # Config
    batch_size = num_devices * 2

    # do stuff
    keys = jax.random.split(key, num_devices * num_batches)
    with mesh:
        params, losses = jax.lax.scan(
            partial(train_loop, data_sharding, batch_size),
            params,
            keys,
        )
    print("XLA retuned control")
    print(losses)
    # Save params to GCS (defaults to project bucket if env not set)
    from save_params import save_params

    save_params(params)

if __name__ == "__main__":
    num_batches = int(os.environ.get("NUM_BATCHES", "100"))
    main(num_batches=num_batches)
