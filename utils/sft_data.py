import jax
import jax.numpy as jnp
import sentencepiece as spm
from pathlib import Path

import numpy as np

# Module-level constants for pre-loaded data to ensure get_training_batch remains efficient
_TOKENIZER_PATH = "data/tokenizer.model"
_DATA_PATH = "data/shakespeare.txt"
_TOKENIZED_PATH = "data/shakespeare_tokenized.npz"


def _load_data() -> jax.Array:
    """Loads and tokenizes the Shakespeare dataset, caching to disk."""
    cache_path = Path(_TOKENIZED_PATH)

    if cache_path.exists():
        with np.load(cache_path) as data:
            return jnp.array(data["tokens"], dtype=jnp.int32)

    sp = spm.SentencePieceProcessor()
    sp.Load(_TOKENIZER_PATH)

    data_path = Path(_DATA_PATH)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {_DATA_PATH}")

    text = data_path.read_text()
    ids = sp.EncodeAsIds(text)
    tokens_np = np.array(ids, dtype=np.int32)

    # Save for future use
    np.savez_compressed(cache_path, tokens=tokens_np)

    return jnp.array(tokens_np, dtype=jnp.int32)


# Pre-tokenize the data
_TOKENS = _load_data()


def get_training_batch(key: jax.Array, batch_size: int, length: int) -> jax.Array:
    """
    Returns a batch of tokenized training data from shakespeare.txt.

    The function samples random contiguous slices and prepends a BOS token (ID 2).
    The returned array has shape (batch_size, length).
    """
    num_tokens = _TOKENS.shape[0]
    # We sample length - 1 tokens to make room for the BOS token
    slice_len = length - 1

    # Generate random start indices for each batch element
    start_indices = jax.random.randint(key, (batch_size,), 0, num_tokens - slice_len)

    def get_slice(start_idx: jax.Array) -> jax.Array:
        content = jax.lax.dynamic_slice_in_dim(_TOKENS, start_idx, slice_len)
        return jnp.concatenate([jnp.array([2], dtype=jnp.int32), content])

    return jax.vmap(get_slice)(start_indices)
