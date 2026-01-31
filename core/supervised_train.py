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
import jax.numpy as jnp
from core.gemma_forward import forward
from utils.inspect_weights import load_weights_as_dict
import optax
from core.gemma_forward import Params
from utils.sft_data import get_training_sample
from functools import partial
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils


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

    keys = jax.random.split(key, batch_size)
    train_data = jax.vmap(partial(get_training_sample, seq_length))(keys)

    train_data = jax.lax.with_sharding_constraint(train_data, data_sharding)

    loss, grads = jax.value_and_grad(loss_batched, argnums=1)(train_data, params)
    return SGD(params, grads, lr), loss


def train_loop(data_sharding, batch_size, params, key) -> tuple[Params, jax.Array]:
    new_params, loss = train(key, batch_size, params, 1024, 0.01, data_sharding)

    return new_params, loss


def param_sharding_fn(params: Params, mesh) -> dict[str, NamedSharding]:
    sharding_dict = {}

    for key in params.keys():
        if key == "model.embed_tokens.weight":
            sharding_dict[key] = NamedSharding(mesh, PartitionSpec("model", None))
        else:
            sharding_dict[key] = NamedSharding(mesh, PartitionSpec())  # replicate

    return sharding_dict


# TESTING
def main(num_batches=100):
    print("--- supervised_train.main() started ---")
    # keys and parameters
    key = jax.random.key(42)
    print("Loading weights...")
    params = load_weights_as_dict("data/gemma-1b/model_stacked_pt.safetensors")
    print("Weights loaded.")

    # Distributed training
    num_devices: int = jax.device_count()
    model_axis_size = min(num_devices, 2)
    batch_axis_size = num_devices // model_axis_size
    device_mesh = mesh_utils.create_device_mesh((batch_axis_size, model_axis_size))
    mesh = Mesh(device_mesh, axis_names=("batch", "model"))
    data_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    param_shardings = param_sharding_fn(params, mesh)
    params = jax.tree.map(
        lambda M, sharding: jax.device_put(M, sharding),
        params,
        param_shardings,
    )

    # Config
    batch_size = num_devices * 2

    # do stuff
    keys = jax.random.split(key, num_devices * num_batches)

    params, losses = jax.lax.scan(
        partial(train_loop, data_sharding, batch_size),
        params,
        keys,
    )

    print("XLA returned control")
    print(losses)

    # Save params to GCS (defaults to project bucket if env not set)
    # from utils.save_params import save_params

    # save_params(params)

    return params
