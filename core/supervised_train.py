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
import optax
from beartype import beartype
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped
from core.forward_parralel import forward_parralel
from core.forward_parralel import Params
from utils.sft_data import get_training_sample
from functools import partial

LEARNING_RATE = 0.01


def loss_fn(xs: Int[Array, "seq"], params: Params) -> Float[Array, ""]:
    predictions = forward_parralel(xs, params)
    input_len = xs.shape[0]
    predictions = predictions[: input_len - 1]  # remove padding
    labels = xs[1:]
    loss = optax.softmax_cross_entropy_with_integer_labels(predictions, labels)
    return jnp.mean(loss)


def train(
    key: PRNGKeyArray,
    batch_size: int,
    params: Params,
    seq_length: int,
    data_sharding,
    optimizer_state: optax.OptState,
) -> tuple[Params, Float[Array, ""], optax.OptState]:
    def loss_batched(xss: Int[Array, "batch seq"], params: Params) -> Float[Array, ""]:
        return jnp.mean(jax.vmap(loss_fn, in_axes=(0, None))(xss, params))

    keys = jax.random.split(key, batch_size)
    train_data = jax.vmap(partial(get_training_sample, seq_length))(keys)

    train_data = jax.lax.with_sharding_constraint(train_data, data_sharding)

    loss, grads = jax.value_and_grad(loss_batched, argnums=1)(train_data, params)
    updates, new_optimizer_state = optax.contrib.muon(learning_rate=LEARNING_RATE).update(
        grads, optimizer_state, params
    )
    new_params = optax.apply_updates(params, updates)
    return new_params, loss, new_optimizer_state


@jaxtyped(typechecker=beartype)
def train_loop(
    data_sharding,
    batch_size: int,
    carry: tuple[Params, optax.OptState],
    key: PRNGKeyArray,
) -> tuple[tuple[Params, optax.OptState], Float[Array, ""]]:
    params, optimizer_state = carry
    new_params, loss, new_optimizer_state = train(
        key,
        batch_size,
        params,
        1024,
        data_sharding,
        optimizer_state,
    )

    return (new_params, new_optimizer_state), loss
