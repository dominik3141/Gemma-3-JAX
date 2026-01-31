import jax
import jax.numpy as jnp
import sentencepiece as spm
from pathlib import Path

import numpy as np

# Module-level constants for pre-loaded data to ensure get_training_batch remains efficient
_TOKENIZER_PATH = "data/gemma-3-1b/tokenizer.model"
_DATA_PATH = "data/train/shakespeare.txt"
_TOKENIZED_PATH = "data/gemma-3-1b/shakespeare_tokenized.npz"


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


def get_training_sample(length: int, key: jax.Array) -> jax.Array:
    """
    Returns a single sequence of tokenized training data from skaespeare.txt
    """
    num_tokens = _TOKENS.shape[0]

    # We sample length - 1 tokens to make room for the BOS token
    slice_len = length - 1
    BOS_token = jnp.array([2], dtype=_TOKENS.dtype)

    start_idx = jax.random.randint(key, (), 0, num_tokens - slice_len)

    content = jax.lax.dynamic_slice_in_dim(_TOKENS, start_idx, slice_len)

    return jnp.concatenate([BOS_token, content])
