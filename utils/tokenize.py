"""
Should provide a simple!, pure! function that just takes ASCII and returns tokens.

Additionally we allow for an argument to add the <user> tags or whatever Gemma uses for
conversational mode.
"""

from __future__ import annotations


def _apply_chat_template(
    text: str,
    *,
    add_chat_tags: bool,
    user_tag: str,
    model_tag: str,
) -> str:
    """Wraps text in optional chat tags."""
    return f"{user_tag}{text}{model_tag}" if add_chat_tags else text


def tokenize_text(
    text: str,
    add_chat_tags: bool = False,
) -> list[int]:
    """Tokenizes ASCII text into codepoint IDs with optional chat tags."""
    user_tag = "<start_of_turn>user\n"
    model_tag = "\n<end_of_turn>\n<start_of_turn>model\n"
    formatted = _apply_chat_template(
        text,
        add_chat_tags=add_chat_tags,
        user_tag=user_tag,
        model_tag=model_tag,
    )
    if not formatted.isascii():
        raise ValueError("Input must be ASCII-only.")
    return [ord(ch) for ch in formatted]
