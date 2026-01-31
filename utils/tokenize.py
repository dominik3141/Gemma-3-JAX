"""
Should provide a simple!, pure! function that just takes ASCII and returns tokens.
"""

from __future__ import annotations

from functools import lru_cache

import sentencepiece as spm


_TOKENIZER_PATH = "data/gemma-1b/tokenizer.model"


@lru_cache(maxsize=1)
def _load_tokenizer() -> spm.SentencePieceProcessor:
    """Loads the Gemma 3 tokenizer model once."""
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(_TOKENIZER_PATH)
    return tokenizer


def tokenize_text(text: str) -> list[int]:
    """Tokenizes ASCII text using the Gemma 3 tokenizer."""
    if not text.isascii():
        raise ValueError("Input must be ASCII-only.")
    return _load_tokenizer().EncodeAsIds(text)


def detokenize_ids(ids: list[int]) -> str:
    """Detokenizes a list of IDs using the Gemma 3 tokenizer."""
    return _load_tokenizer().DecodeIds(ids)
