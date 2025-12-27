import jax
import jax.numpy as jnp

Params = dict[str, jax.Array]


def forward(xs, Params) -> jax.Array:
    """
    xs is a sequence of tokens (tokenization happens outside this function).
    We are trying to predict the next token.

    Steps:
    1.  Embedding the tokens
        vocab_size -> d_model
    ITERATE (26 Blocks):
        2.  Attention
            d_model -> d_model
        3.  MLP
            d_model -> d_model
    4.  Final RMSNorm
        d_model -> d_model
    5.  Make logits
        d_model -> vocab_size
    """
    pass


def main():
    print("Hello from jax-gemma-1b!")


if __name__ == "__main__":
    main()
