"""
An inference optimized version of the forward function for Gemma.

We only calculate Q,K,V for a single token, the newest token in our sequence, and expect the K,V
values for each layer to have been calculated prior to the invocation of our new forward function.

I'm not yet sure whether we should reuse our current forward function in `gemma_forward.py` for the prefill
step that is supposed to provide us with the prior K,V values, or if we need a third forward function.
We would lose some compute efficiency by reusing our current forward function as it is optimized
for pretraining and calculates the next token as every position, while for prefill we would only need
it to calculate the K,V values.

This is all getting a bit messy, but it looks like we can hardly avoid duplicating logic if we don't want
to accept a lot of conditional branching, which might make our code less efficient (and which generally
isn't something I would consider good style).

TODO:
    - KV cache needs padding, currently we jit recompile at every step
    - Extract the logic that is shared among all forward functions to a separate file (RMSNorm etc.)
"""

import jax
import jax.numpy as jnp
from core.gemma_forward import (
    Params,
    RMSNorm,
    attnHead,
    calc_qkv,
    extract_block_params,
    postAttn,
)


def Block_KV_cached(inits, scans) -> jax.Array:
    """
    Gemma block for a single token. Communication with other tokens via KV cache.
    """
    x, pos = (
        inits  # should the position really be carried? Optimally would be a closure?
    )
    block_params, is_local_attn, Ks_cached, Vs_cached = scans

    # make a copy of x to keep the residual
    x_og = x
    x = RMSNorm(x, block_params["input_layernorm.weight"])

    # Calculate comm vectors for new token
    K_new, V_new, Qs = calc_qkv(x, block_params, pos, is_local_attn)
    # add sequence dimension
    K_new = jnp.expand_dims(K_new, axis=0)
    V_new = jnp.expand_dims(V_new, axis=0)
    Qs = jnp.expand_dims(Qs, axis=0)

    # combine new comm vectors with the cached ones
    Ks = jnp.concatenate([Ks_cached, K_new])
    Vs = jnp.concatenate([Vs_cached, V_new])

    # COMMUNICATION WITH OTHER TOKENS
    # first we go one level down to parallelize over the four Qs
    Qs = jnp.transpose(Qs, (1, 0, 2))  # head dimension should be first
    pos_array = jnp.expand_dims(pos, axis=0)
    x = jax.vmap(attnHead, in_axes=(None, None, 0, None, None))(
        Ks, Vs, Qs, pos_array, is_local_attn
    )
    x = jnp.reshape(x, (4 * 256))  # concat heads and remove sequence dim

    x = postAttn(x, x_og, block_params)

    return (x, pos), (Ks, Vs)


def forward_single(
    x: jax.Array, params: Params, pos: int, Ks_cached: jax.Array, Vs_cached: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Predict the next token given only a single token together with the K,V vectors
    of all prior tokens.
    Also needs the position of x in the larger sequence so we can calculate the
    positional embedding.
    Returns predicted next token as well as updated K,V cache.
    """
    # embed the token
    x = params["model.embed_tokens.weight"][x]

    # normalize according to Gemma reference implementation
    x = jnp.sqrt(1152) * x

    # single position should be wrapped in array to not confuse later vmaps
    pos = jnp.array(pos)

    # BLOCKS
    # Create the pattern list based on the config
    # 26 layers total: (5 local, 1 global) * 4 + 2 local
    layer_types = (["local"] * 5 + ["global"]) * 4 + ["local"] * 2
    is_local_attn = jnp.array(
        [t == "local" for t in layer_types]
    )  # shape (26,), [1,1...,1,1]
    (x, _), (Ks_cached, Vs_cached) = jax.lax.scan(
        Block_KV_cached,
        (x, pos),
        (extract_block_params(params), is_local_attn, Ks_cached, Vs_cached),
    )

    # final norm
    x = RMSNorm(x, params["model.norm.weight"])

    # map to logits
    x = params["model.embed_tokens.weight"] @ x

    return x, Ks_cached, Vs_cached


def main() -> None:
    """Test function for forward_single with actual generation."""
    from utils.inspect_weights import load_weights_as_dict
    from utils.tokenize import tokenize_text, detokenize_ids

    print("Loading weights from data/model_stacked_pt.safetensors...")
    params = load_weights_as_dict("data/model_stacked_pt.safetensors")
    print("Weights loaded.")

    prompt = "The capital of France is Paris. The capital of Germany is"
    tokens = tokenize_text(prompt)

    # Add BOS token (2) if not present
    if tokens[0] != 2:
        tokens = [2] + tokens

    print(f"Prompt: '{prompt}'")
    print(f"Tokens: {tokens}")

    # Initialize cache
    num_layers = 26
    head_dim = 256

    # Initialize with size 0 in sequence dimension
    Ks_cached = jnp.zeros((num_layers, 0, head_dim), dtype=jnp.bfloat16)
    Vs_cached = jnp.zeros((num_layers, 0, head_dim), dtype=jnp.bfloat16)

    print("Processing prompt (prefill)...")
    logits = None

    # Process each token in the prompt
    for i, token in enumerate(tokens):
        token_id = jnp.array(token)
        pos = i  # Position in sequence

        logits, Ks_cached, Vs_cached = forward_single(
            token_id, params, pos, Ks_cached, Vs_cached
        )

    print("Prompt processed.")

    # Generate new tokens
    print("Generating...")
    generated_tokens = []
    max_new_tokens = 10

    curr_pos = len(tokens)

    for i in range(max_new_tokens):
        # Sample from logits (greedy)
        next_token = jnp.argmax(logits).item()
        generated_tokens.append(next_token)

        # Decode so far
        current_text = detokenize_ids(tokens + generated_tokens)
        print(f"Step {i}: token {next_token} -> {repr(current_text)}")

        # Feed back
        token_id = jnp.array(next_token)
        logits, Ks_cached, Vs_cached = forward_single(
            token_id, params, curr_pos, Ks_cached, Vs_cached
        )
        curr_pos += 1

    final_text = detokenize_ids(tokens + generated_tokens)
    print(f"Final output: {final_text}")


if __name__ == "__main__":
    main()
