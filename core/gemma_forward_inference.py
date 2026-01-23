"""
An inference optimized version of the forward function for Gemma.

We only calculate Q,K,V for a single token, the newest token in our sequence, and expect the K,V
values for each layer to have been calculated prior to the invocation of our new forward function.

I'm not yet sure whether we should reuse our current forward function in `gemma_forward.py` for the prefill
step that is supposed to provide us with the prior K,V values, or if we need a third forward function.
We would loose some compute efficency by reusing our current forward function as it is optimized
for pretraining and calculates the next token as every position, while for prefill we would only need
it to calculate the K,V values.

This is all getting a bit messy, but it looks like we can hardly avoid duplicating logic if we don't want
to accept a lot of conditional branching, which might make our code less efficent (and which generally
isn't something I would consider good style).
"""

import jax
from gemma_forward import Params


def forward_single(xs: jax.Array, params: Params, pos: int):
    """
    Our forward function calculates the next token for every token, which is what
    we want for pretraining, but if we just want to get the next token from a sequence
    of tokens we waste a lot of compute (O(n) vs O(n^2)).
    This function only calculates the new value of the token at pos based on the tokens
    prior to pos.
    """
    pass
