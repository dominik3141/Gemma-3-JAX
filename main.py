import jax
import jax.numpy as jnp


def main():
    vocab_size = 262144
    token_ids = jnp.array([123, 231])
    xs = jax.nn.one_hot(token_ids, vocab_size)
    print(xs)


if __name__ == "__main__":
    main()
