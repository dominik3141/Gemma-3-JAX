import argparse

import jax
import jax.numpy as jnp
import sentencepiece as spm
from safetensors import safe_open

from gemma_forward import forward


VOCAB_SIZE = 262_144
BOS_ID = 2
EOS_ID = 1
END_OF_TURN_ID = 106  # tokenizer.piece_to_id("<end_of_turn>")
STOP_IDS_DEFAULT = (EOS_ID, END_OF_TURN_ID)


def load_weights_as_jax(path: str) -> dict[str, jax.Array]:
    params: dict[str, jax.Array] = {}
    with safe_open(path, framework="np", device="cpu") as f:
        for key in f.keys():
            params[key] = jnp.asarray(f.get_tensor(key))
    return params


def load_tokenizer(model_path: str) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp


def sample_next_token(
    logits: jax.Array, rng: jax.random.PRNGKey, temperature: float
) -> tuple[int, jax.random.PRNGKey]:
    logits = logits.astype(jnp.float32)
    if temperature <= 0:
        return int(jnp.argmax(logits)), rng

    rng, subkey = jax.random.split(rng)
    scaled_logits = logits / jnp.float32(temperature)
    token = int(jax.random.categorical(subkey, scaled_logits))
    return token, rng


def format_prompt(prompt: str, template: str) -> str:
    if template == "plain":
        return prompt
    if template == "gemma-instruct":
        return (
            "<start_of_turn>user\n" + prompt + "\n<end_of_turn>\n<start_of_turn>model\n"
        )
    raise ValueError(f"Unknown template '{template}'")


def strip_trailing_stop(token_ids: list[int], stop_ids: set[int]) -> list[int]:
    if not stop_ids:
        return token_ids
    while token_ids and token_ids[-1] in stop_ids:
        token_ids = token_ids[:-1]
    return token_ids


def generate(
    prompt: str,
    *,
    params: dict[str, jax.Array],
    tokenizer: spm.SentencePieceProcessor,
    max_new_tokens: int = 32,
    temperature: float = 0.8,
    seed: int = 0,
    stop_ids: set[int] | None = None,
    template: str = "gemma-instruct",
) -> tuple[list[int], set[int], int]:
    formatted = format_prompt(prompt, template)
    token_ids: list[int] = [BOS_ID] + tokenizer.EncodeAsIds(formatted)
    prompt_len = len(token_ids)
    stop_set = set(stop_ids) if stop_ids is not None else set(STOP_IDS_DEFAULT)
    rng = jax.random.PRNGKey(seed)

    for _ in range(max_new_tokens):
        xs = jnp.asarray(token_ids, dtype=jnp.int32)

        logits = forward(xs, params)
        next_logits = logits[len(token_ids) - 1]  # pick logits for last real token
        next_token, rng = sample_next_token(next_logits, rng, temperature)
        token_ids.append(next_token)
        if stop_set and next_token in stop_set:
            break

    return token_ids, stop_set, prompt_len


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample autoregressively from the Gemma 1B model."
    )
    parser.add_argument(
        "--prompt", type=str, default="Hello", help="Input prompt text."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="model_stacked.safetensors",
        help="Path to the stacked safetensors weights.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizer.model",
        help="Path to the SentencePiece tokenizer model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Number of tokens to sample after the prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Softmax temperature (<=0 uses greedy decoding).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for sampling.",
    )
    parser.add_argument(
        "--no-eos-stop",
        action="store_true",
        help="Do not stop early on the eos token.",
    )
    parser.add_argument(
        "--no-end-of-turn-stop",
        action="store_true",
        help="Do not stop early on <end_of_turn>.",
    )
    parser.add_argument(
        "--template",
        choices=["plain", "gemma-instruct"],
        default="gemma-instruct",
        help="Prompt formatting template.",
    )
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    params = load_weights_as_jax(args.weights)

    stop_ids: set[int] = set()
    if not args.no_eos_stop:
        stop_ids.add(EOS_ID)
    if not args.no_end_of_turn_stop:
        stop_ids.add(END_OF_TURN_ID)

    generated, used_stop_ids, prompt_len = generate(
        args.prompt,
        params=params,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        stop_ids=stop_ids,
        template=args.template,
    )

    gen_only = generated[prompt_len:]
    gen_only = strip_trailing_stop(gen_only, used_stop_ids)
    decoded = tokenizer.DecodeIds(gen_only)

    print(f"Prompt: {args.prompt}")
    print(f"Prompt token count (incl. BOS): {prompt_len}")
    print(f"Generated IDs: {gen_only}")
    print(f"Decoded generation: {decoded}")


if __name__ == "__main__":
    cli_main()
