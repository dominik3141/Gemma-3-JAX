#!/usr/bin/env python3
"""
Minimal text generation helper for Gemma 3 (27B) using the inference forward path.
Loads sharded weights from local SSD and runs prefill + greedy decoding.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import jax
import jax.numpy as jnp

from core.gemma_forward import config
from core.gemma_forward_inference import forward_single
from utils.load_sharded import load_stacked_sharded_model
from utils.tokenize_text import detokenize_ids, tokenize_text


DEFAULT_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the "
    "Assistant solves it. The assistant first thinks about the reasoning process in "
    "the mind and then provides the user with the answer. The reasoning process and "
    "answer are enclosed within <think>...</think> and <answer>...</answer> tags, "
    "respectively, i.e., <think> reasoning process here </think> <answer> answer here "
    "</answer>. The answer should only include the numerical result and nothing else. "
    "User: Calculate the square root of 2 up to three decimal places. Assistant:"
)


def _ensure_tokenizer() -> None:
    target = "data/gemma-3-1b/tokenizer.model"
    if os.path.exists(target):
        return

    alt = "data/gemma-3-27b-local/tokenizer.model"
    if os.path.exists(alt):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy(alt, target)
        print(f"[tokenizer] Copied {alt} -> {target}", flush=True)
        return

    raise FileNotFoundError(
        f"Tokenizer not found at {target}. "
        f"Expected alt at {alt} is also missing."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=64)
    parser.add_argument("--weights-dir", type=str, default="data/gemma-3-27b-local")
    parser.add_argument("--cache-size", type=int, default=0)
    parser.add_argument("--stop-on", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _ensure_tokenizer()

    tokens = tokenize_text(args.prompt)
    if tokens and tokens[0] != 2:
        tokens = [2] + tokens
    elif not tokens:
        tokens = [2]

    print("[load] weights...", flush=True)
    mesh = jax.sharding.Mesh(jax.devices(), axis_names=("model",))
    params = load_stacked_sharded_model(args.weights_dir, mesh)

    cache_size = args.cache_size or (len(tokens) + args.max_new_tokens)
    Ks = jnp.zeros(
        (config.num_layers, cache_size, config.num_key_value_heads, config.head_dim),
        dtype=jnp.bfloat16,
    )
    Vs = jnp.zeros(
        (config.num_layers, cache_size, config.num_key_value_heads, config.d_kvq),
        dtype=jnp.bfloat16,
    )

    print("[prefill] ...", flush=True)
    logits = None
    t0 = time.time()
    for i, tok in enumerate(tokens):
        logits, Ks, Vs = forward_single(jnp.array(tok), params, i, Ks, Vs)
        if args.log_every and (i + 1) % args.log_every == 0:
            dt = time.time() - t0
            print(f"[prefill] {i+1}/{len(tokens)} ({dt:.1f}s)", flush=True)

    prefill_time = time.time() - t0
    print(f"[prefill] done in {prefill_time:.2f}s", flush=True)

    print("[generate] ...", flush=True)
    generated: list[int] = []
    pos = len(tokens)
    gen_start = time.time()
    for i in range(args.max_new_tokens):
        next_token = int(jnp.argmax(logits))
        generated.append(next_token)
        logits, Ks, Vs = forward_single(jnp.array(next_token), params, pos, Ks, Vs)
        pos += 1

        if args.log_every and (i + 1) % args.log_every == 0:
            print(f"[generate] {i+1}/{args.max_new_tokens}", flush=True)

        if args.stop_on:
            text = detokenize_ids(tokens + generated)
            if args.stop_on in text:
                break

    gen_time = time.time() - gen_start
    print(f"[generate] done in {gen_time:.2f}s", flush=True)

    out = detokenize_ids(tokens + generated)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
