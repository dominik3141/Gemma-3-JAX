import jax
import jax.numpy as jnp
from nn_utils import forward
from inspect_weights import load_weights_as_dict


def main():
    vocab_size = 262144
    token_ids = jnp.array([123, 231])
    xs = jax.nn.one_hot(token_ids, vocab_size)

    params = load_weights_as_dict("model_stacked.safetensors")
    xs = forward(xs, params)

    print(xs)


if __name__ == "__main__":
    main()
