import jax
import jax.numpy as jnp
from nn_utils import forward
from inspect_weights import load_weights_as_dict


def sample(logits: jax.Array) -> jax.Array:
    return jnp.argmax(logits)


def main():
    vocab_size = 262144
    # token_ids = jnp.array([2, 81115])  # <BOS> Hell
    token_ids = jnp.array([2, 153637, 532, 622])  # Cats and D
    xs = jax.nn.one_hot(token_ids, vocab_size)

    params = load_weights_as_dict("model_stacked.safetensors")
    xs = forward(xs, params)

    predicted = jax.vmap(sample)(xs)
    print(predicted)


if __name__ == "__main__":
    main()
