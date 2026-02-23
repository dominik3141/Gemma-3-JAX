"""Inference script entrypoint for batched cached forward generation."""

from dataclasses import dataclass
import logging
import time

import jax
import jax.experimental.multihost_utils as multihost_utils
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int

from core.forward_inference import allocate_kv_cache, decode, prefill
from utils.params_io_27b import DEFAULT_ORBAX_CHECKPOINT, load_params
from utils.profiling import build_shared_profile_options
from utils.tokenize_text import detokenize_ids, tokenize_text
import utils.wandb_logging as wandb_logging

ENABLE_PROFILER = True
PROFILE_LOGDIR = "gs://gemma-3-training-profiles-20260207-165411-1d9c5e-euw4"
PROFILE_SESSION_PREFIX = "gemma_inference"
PROFILE_START_BARRIER_NAME = "gemma_inference_profile_start"
PROFILE_STOP_BARRIER_NAME = "gemma_inference_profile_stop"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptBatch:
    prompts: list[str]
    prompt_tokens_batch: list[list[int]]
    prompt_lengths: list[int]
    prompt_lengths_array: Int[Array, "batch"]
    tokens_padded: Int[Array, "batch prompt_len"]
    batch_size: int
    max_prompt_tokens: int


def get_prompts() -> list[str]:
    return [
        "The capital of France is Paris. The capital of Germany is",
        "Write a Python function that returns the square root of a number without using the math library.",
        "Draft a 120-word scene from a science-fiction story where an engineer explains a repair in zero gravity.",
        "A train leaves New York at 3:15 PM traveling 180 miles at 60 miles per hour. What is the arrival time?",
    ]


def tokenize_prompt(prompt: str) -> list[int]:
    tokens = tokenize_text(prompt)
    if not tokens or tokens[0] != 2:
        tokens = [2] + tokens
    return tokens


def build_prompt_batch(prompts: list[str]) -> PromptBatch:
    prompt_tokens_batch: list[list[int]] = []
    for prompt_idx, prompt in enumerate(prompts):
        tokens = tokenize_prompt(prompt)
        prompt_tokens_batch.append(tokens)
        print(f"Prompt {prompt_idx}: '{prompt}'")
        print(f"Prompt {prompt_idx} tokens ({len(tokens)}): {tokens}")

    prompt_lengths = [len(tokens) for tokens in prompt_tokens_batch]
    batch_size = len(prompt_tokens_batch)
    max_prompt_tokens = max(prompt_lengths)

    tokens_padded_np = np.zeros((batch_size, max_prompt_tokens), dtype=np.int32)
    for idx, tokens in enumerate(prompt_tokens_batch):
        tokens_padded_np[idx, : len(tokens)] = tokens
    tokens_padded = jnp.asarray(tokens_padded_np, dtype=jnp.int32)
    prompt_lengths_array = jnp.asarray(prompt_lengths, dtype=jnp.int32)

    return PromptBatch(
        prompts=prompts,
        prompt_tokens_batch=prompt_tokens_batch,
        prompt_lengths=prompt_lengths,
        prompt_lengths_array=prompt_lengths_array,
        tokens_padded=tokens_padded,
        batch_size=batch_size,
        max_prompt_tokens=max_prompt_tokens,
    )


def print_final_outputs(
    prompt_tokens_batch: list[list[int]], generated_tokens_batch: list[list[int]]
) -> None:
    for prompt_idx, (prompt_tokens, generated_tokens) in enumerate(
        zip(prompt_tokens_batch, generated_tokens_batch)
    ):
        final_text = detokenize_ids(prompt_tokens + generated_tokens)
        print(f"Final output {prompt_idx}: {final_text}")


def report_and_log_metrics(
    *,
    prefill_elapsed: float,
    generation_elapsed: float,
    prompt_lengths: list[int],
    generated_tokens_batch: list[list[int]],
    batch_size: int,
) -> None:
    total_prefill_tokens = sum(prompt_lengths)
    total_generated_tokens = sum(len(tokens) for tokens in generated_tokens_batch)

    prefill_tokens_per_s = (
        total_prefill_tokens / prefill_elapsed if prefill_elapsed > 0 else 0.0
    )
    generation_tokens_per_s_batch = (
        total_generated_tokens / generation_elapsed if generation_elapsed > 0 else 0.0
    )
    generation_tokens_per_s_per_prompt = (
        generation_tokens_per_s_batch / batch_size if batch_size > 0 else 0.0
    )

    print(
        f"Prefill avg (batch): {prefill_tokens_per_s:.1f} tokens/s "
        f"({prefill_elapsed:.3f}s total for {total_prefill_tokens} tokens)"
    )
    print(
        f"Generation avg (batch): {generation_tokens_per_s_batch:.1f} tokens/s "
        f"({generation_elapsed:.3f}s total for {total_generated_tokens} tokens)"
    )
    print(
        f"Generation avg (per prompt): {generation_tokens_per_s_per_prompt:.1f} tokens/s "
        f"(batch_size={batch_size})"
    )

    wandb_logging.log_metrics(
        {
            "inference/tokens_per_second": float(generation_tokens_per_s_batch),
            "inference/tokens_per_second_per_prompt": float(
                generation_tokens_per_s_per_prompt
            ),
            "inference/batch_size": float(batch_size),
        },
        step=0,
    )


def start_profiler() -> bool:
    if not ENABLE_PROFILER:
        return False

    try:
        LOGGER.info(
            "Synchronizing hosts before profiler start (%s)...",
            PROFILE_START_BARRIER_NAME,
        )
        multihost_utils.sync_global_devices(PROFILE_START_BARRIER_NAME)
        profile_options, profile_session_id = build_shared_profile_options(
            PROFILE_SESSION_PREFIX
        )
        LOGGER.info(
            "Starting JAX profiler trace to %s (session_id=%s)...",
            PROFILE_LOGDIR,
            profile_session_id,
        )
        jax.profiler.start_trace(PROFILE_LOGDIR, profiler_options=profile_options)
        return True
    except Exception as exc:
        LOGGER.warning("Failed to start JAX profiler trace: %s", exc)
        return False


def stop_profiler(profiler_started: bool) -> None:
    if not ENABLE_PROFILER:
        return

    try:
        LOGGER.info(
            "Synchronizing hosts before profiler stop (%s)...",
            PROFILE_STOP_BARRIER_NAME,
        )
        multihost_utils.sync_global_devices(PROFILE_STOP_BARRIER_NAME)
    except Exception as exc:
        LOGGER.warning("Failed to synchronize hosts before stopping profiler: %s", exc)

    if not profiler_started:
        return

    try:
        jax.profiler.stop_trace()
        LOGGER.info("Stopped JAX profiler trace.")
    except Exception as exc:
        LOGGER.warning("Failed to stop JAX profiler trace: %s", exc)


def main() -> None:
    max_new_tokens = 10_000
    decode_temperature = 1.0
    seed = 42
    key = jax.random.PRNGKey(seed)
    profiler_started = False

    wandb_logging.init_wandb(
        project="gemma-27b-inference",
        config={
            "max_new_tokens": max_new_tokens,
            "decode_temperature": decode_temperature,
            "seed": seed,
            "enable_profiler": ENABLE_PROFILER,
            "profile_logdir": PROFILE_LOGDIR,
            "profile_session_prefix": PROFILE_SESSION_PREFIX,
        },
    )

    try:
        profiler_started = start_profiler()

        print("Loading weights from Orbax checkpoint...")
        mesh = jax.sharding.Mesh(jax.devices(), axis_names=("model",))
        params = load_params(DEFAULT_ORBAX_CHECKPOINT, mesh)
        print("Weights loaded.")

        prompt_batch = build_prompt_batch(get_prompts())
        kv_cache_len = prompt_batch.max_prompt_tokens + max_new_tokens + 1
        ks_cached, vs_cached = allocate_kv_cache(
            prompt_batch.batch_size, kv_cache_len, mesh
        )

        print("Compiling prefill...")
        prefill_batch = jax.vmap(
            prefill, in_axes=(None, 0, 0, 1, 1), out_axes=(0, 1, 1)
        )
        prefill_jit = jax.jit(
            prefill_batch,
            donate_argnames=("ks_cached", "vs_cached"),
        )
        prefill_compile_start = time.perf_counter()
        compiled_prefill = prefill_jit.lower(
            params,
            prompt_batch.tokens_padded,
            prompt_batch.prompt_lengths_array,
            ks_cached,
            vs_cached,
        ).compile()
        prefill_compile_elapsed = time.perf_counter() - prefill_compile_start
        print(f"Prefill JIT compilation time: {prefill_compile_elapsed:.3f}s")

        print("Processing prompts (prefill)...")
        multihost_utils.sync_global_devices("gemma_inference_prefill_start")
        prefill_start = time.perf_counter()
        logits, ks_cached, vs_cached = compiled_prefill(
            params,
            prompt_batch.tokens_padded,
            prompt_batch.prompt_lengths_array,
            ks_cached,
            vs_cached,
        )
        logits.block_until_ready()
        multihost_utils.sync_global_devices("gemma_inference_prefill_stop")
        prefill_elapsed = time.perf_counter() - prefill_start

        print("Compiling decode...")
        decode_keys = jax.random.split(key, prompt_batch.batch_size)
        decode_batch = jax.vmap(
            decode, in_axes=(None, 0, 0, 0, 1, 1, None, None), out_axes=(1, 1, 1, 1)
        )
        decode_jit = jax.jit(
            decode_batch,
            static_argnames=("max_new_tokens", "temperature"),
            donate_argnames=("ks_cached", "vs_cached"),
        )
        decode_compile_start = time.perf_counter()
        compiled_decode = decode_jit.lower(
            params,
            decode_keys,
            logits,
            prompt_batch.prompt_lengths_array,
            ks_cached,
            vs_cached,
            max_new_tokens,
            decode_temperature,
        ).compile()
        decode_compile_elapsed = time.perf_counter() - decode_compile_start
        print(f"Decode JIT compilation time: {decode_compile_elapsed:.3f}s")

        print("Prompts processed.")
        print("Generating batched completions...")
        multihost_utils.sync_global_devices("gemma_inference_generation_start")
        generation_start = time.perf_counter()
        generated_tokens_array, _, ks_cached, vs_cached = compiled_decode(
            params,
            decode_keys,
            logits,
            prompt_batch.prompt_lengths_array,
            ks_cached,
            vs_cached,
        )
        generated_tokens_array.block_until_ready()
        multihost_utils.sync_global_devices("gemma_inference_generation_stop")
        generation_elapsed = time.perf_counter() - generation_start

        generated_tokens_batch = np.asarray(generated_tokens_array).T.tolist()
        print_final_outputs(prompt_batch.prompt_tokens_batch, generated_tokens_batch)
        report_and_log_metrics(
            prefill_elapsed=prefill_elapsed,
            generation_elapsed=generation_elapsed,
            prompt_lengths=prompt_batch.prompt_lengths,
            generated_tokens_batch=generated_tokens_batch,
            batch_size=prompt_batch.batch_size,
        )
    finally:
        try:
            stop_profiler(profiler_started)
        finally:
            wandb_logging.finish_wandb()


if __name__ == "__main__":
    main()
