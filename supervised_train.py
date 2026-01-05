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
from gemma_forward import forward
from inspect_weights import load_weights_as_dict
import optax
from gemma_forward import Params
from sft_data import get_training_batch


def loss_fn(xs, params) -> jax.Array:
    predictions = forward(xs, params)
    input_len = xs.shape[0]
    predictions = predictions[: input_len - 1]  # remove padding
    labels = xs[1:]
    loss = optax.softmax_cross_entropy_with_integer_labels(predictions, labels)
    return jnp.mean(loss)


def SGD(params, grads, lr) -> Params:
    return jax.tree_util.tree_map(lambda param, grad: param - lr * grad, params, grads)


def train(key, batch_size, params, seq_length, lr) -> Params:
    def loss_batched(xss, params) -> jax.Array:
        return jnp.mean(jax.vmap(loss_fn, in_axes=(0, None))(xss, params))

    train_data = get_training_batch(key, batch_size, seq_length)
    loss, grads = jax.value_and_grad(loss_batched, argnums=1)(train_data, params)
    print(loss)
    return SGD(params, grads, lr)


# TESTING
def main():
    key = jax.random.key(42)
    params = load_weights_as_dict("model_stacked_pt.safetensors")

    print(train(key, 2, params, 4, 0.01))


if __name__ == "__main__":
    main()
