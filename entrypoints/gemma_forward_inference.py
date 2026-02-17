"""Inference script entrypoint for batched cached forward generation."""

from dataclasses import dataclass
import logging
import time

import jax
import jax.experimental.multihost_utils as multihost_utils
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from core.gemma_forward import config
from core.gemma_forward_inference import forward_single_impl
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
    last_prompt_tokens: Int[Array, "batch"]
    batch_size: int
    max_prompt_tokens: int


@dataclass(frozen=True)
class PrefillResult:
    logits: Float[Array, "batch vocab"]
    ks_cached: Float[Array, "layer batch cache_pos kv_head head_dim"]
    vs_cached: Float[Array, "layer batch cache_pos kv_head value_dim"]
    prefill_elapsed: float


@dataclass(frozen=True)
class DecodeResult:
    final_carry: tuple[
        Float[Array, "batch vocab"],
        Int[Array, "batch"],
        Float[Array, "layer batch cache_pos kv_head head_dim"],
        Float[Array, "layer batch cache_pos kv_head value_dim"],
    ]
    generated_tokens_array: Int[Array, "decode_len batch"]
    compile_elapsed: float
    generation_elapsed: float


def get_prompts() -> list[str]:
    return [
        "The capital of France is Paris. The capital of Germany is",
        "The capital of France is Paris. The capital of Italy is",
        "The capital of France is Paris. The capital of Spain is",
        "The capital of France is Paris. The capital of Portugal is",
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
    last_prompt_tokens = tokens_padded[
        jnp.arange(batch_size, dtype=jnp.int32), prompt_lengths_array - 1
    ]

    return PromptBatch(
        prompts=prompts,
        prompt_tokens_batch=prompt_tokens_batch,
        prompt_lengths=prompt_lengths,
        prompt_lengths_array=prompt_lengths_array,
        tokens_padded=tokens_padded,
        last_prompt_tokens=last_prompt_tokens,
        batch_size=batch_size,
        max_prompt_tokens=max_prompt_tokens,
    )


def forward_single_batch(
    token_ids: Int[Array, "batch"],
    params,
    pos: Int[Array, "batch"],
    ks_cached: Float[Array, "layer batch cache_pos kv_head head_dim"],
    vs_cached: Float[Array, "layer batch cache_pos kv_head value_dim"],
) -> tuple[
    Float[Array, "batch vocab"],
    Float[Array, "layer batch cache_pos kv_head head_dim"],
    Float[Array, "layer batch cache_pos kv_head value_dim"],
]:
    # Keep carry as [layers, batch, ...] to match decode scan layout.
    return jax.vmap(
        forward_single_impl, in_axes=(0, None, 0, 1, 1), out_axes=(0, 1, 1)
    )(token_ids, params, pos, ks_cached, vs_cached)


def run_prefill(
    params,
    prompt_tokens: Int[Array, "batch prompt_len"],
    prompt_lengths: Int[Array, "batch"],
    prompt_last_tokens: Int[Array, "batch"],
    init_ks_cached: Float[Array, "layer batch cache_pos kv_head head_dim"],
    init_vs_cached: Float[Array, "layer batch cache_pos kv_head value_dim"],
) -> tuple[
    Float[Array, "batch vocab"],
    Float[Array, "layer batch cache_pos kv_head head_dim"],
    Float[Array, "layer batch cache_pos kv_head value_dim"],
]:
    pos0 = jnp.zeros((prompt_tokens.shape[0],), dtype=jnp.int32)
    logits, ks_cached, vs_cached = forward_single_batch(
        prompt_tokens[:, 0],
        params,
        pos0,
        init_ks_cached,
        init_vs_cached,
    )

    def prefill_step(carry, t):
        logits, ks_cached, vs_cached = carry
        is_active = t < prompt_lengths
        pos = jnp.where(is_active, t, prompt_lengths - 1).astype(jnp.int32)
        token_ids = jnp.where(
            is_active, prompt_tokens[:, t], prompt_last_tokens
        ).astype(jnp.int32)
        logits, ks_cached, vs_cached = forward_single_batch(
            token_ids, params, pos, ks_cached, vs_cached
        )
        return (logits, ks_cached, vs_cached), None

    timesteps = jnp.arange(1, prompt_tokens.shape[1], dtype=jnp.int32)
    (logits, ks_cached, vs_cached), _ = jax.lax.scan(
        prefill_step,
        (logits, ks_cached, vs_cached),
        timesteps,
    )
    return logits, ks_cached, vs_cached


def allocate_kv_cache(
    batch_size: int, kv_cache_len: int
) -> tuple[
    Float[Array, "layer batch cache_pos kv_head head_dim"],
    Float[Array, "layer batch cache_pos kv_head value_dim"],
]:
    ks_cached = jnp.zeros(
        (
            config.num_layers,
            batch_size,
            kv_cache_len,
            config.num_key_value_heads,
            config.head_dim,
        ),
        dtype=jnp.bfloat16,
    )
    vs_cached = jnp.zeros(
        (
            config.num_layers,
            batch_size,
            kv_cache_len,
            config.num_key_value_heads,
            config.d_kvq,
        ),
        dtype=jnp.bfloat16,
    )
    return ks_cached, vs_cached


def prefill(
    params,
    prompt_batch: PromptBatch,
    kv_cache_len: int,
) -> PrefillResult:
    init_ks_cached, init_vs_cached = allocate_kv_cache(
        prompt_batch.batch_size, kv_cache_len
    )
    run_prefill_jit = jax.jit(run_prefill, donate_argnums=(4, 5))

    prefill_start = time.perf_counter()
    logits, ks_cached, vs_cached = run_prefill_jit(
        params,
        prompt_batch.tokens_padded,
        prompt_batch.prompt_lengths_array,
        prompt_batch.last_prompt_tokens,
        init_ks_cached,
        init_vs_cached,
    )
    logits.block_until_ready()
    prefill_elapsed = time.perf_counter() - prefill_start
    return PrefillResult(logits, ks_cached, vs_cached, prefill_elapsed)


def scan_body(
    params,
    carry: tuple[
        Float[Array, "batch vocab"],
        Int[Array, "batch"],
        Float[Array, "layer batch cache_pos kv_head head_dim"],
        Float[Array, "layer batch cache_pos kv_head value_dim"],
    ],
    _,
) -> tuple[
    tuple[
        Float[Array, "batch vocab"],
        Int[Array, "batch"],
        Float[Array, "layer batch cache_pos kv_head head_dim"],
        Float[Array, "layer batch cache_pos kv_head value_dim"],
    ],
    Int[Array, "batch"],
]:
    logits, curr_pos, ks_cached, vs_cached = carry
    next_tokens = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    new_logits, new_ks, new_vs = forward_single_batch(
        next_tokens, params, curr_pos, ks_cached, vs_cached
    )
    return (new_logits, curr_pos + 1, new_ks, new_vs), next_tokens


def make_run_scan(max_new_tokens: int):
    def run_scan(
        params,
        init_carry: tuple[
            Float[Array, "batch vocab"],
            Int[Array, "batch"],
            Float[Array, "layer batch cache_pos kv_head head_dim"],
            Float[Array, "layer batch cache_pos kv_head value_dim"],
        ],
    ) -> tuple[
        tuple[
            Float[Array, "batch vocab"],
            Int[Array, "batch"],
            Float[Array, "layer batch cache_pos kv_head head_dim"],
            Float[Array, "layer batch cache_pos kv_head value_dim"],
        ],
        Int[Array, "decode_len batch"],
    ]:
        return jax.lax.scan(
            lambda carry, xs: scan_body(params, carry, xs),
            init_carry,
            None,
            length=max_new_tokens,
        )

    return run_scan


def decode(
    params,
    init_carry: tuple[
        Float[Array, "batch vocab"],
        Int[Array, "batch"],
        Float[Array, "layer batch cache_pos kv_head head_dim"],
        Float[Array, "layer batch cache_pos kv_head value_dim"],
    ],
    max_new_tokens: int,
) -> DecodeResult:
    run_scan = make_run_scan(max_new_tokens)

    compile_start = time.perf_counter()
    run_scan_jit = jax.jit(run_scan, donate_argnums=(1,))
    lowered = run_scan_jit.lower(params, init_carry)
    compiled_scan = lowered.compile()
    compile_elapsed = time.perf_counter() - compile_start
    print(f"JIT compilation time: {compile_elapsed:.3f}s")

    generation_start = time.perf_counter()
    final_carry, generated_tokens_array = compiled_scan(params, init_carry)
    generated_tokens_array.block_until_ready()
    generation_elapsed = time.perf_counter() - generation_start

    return DecodeResult(
        final_carry=final_carry,
        generated_tokens_array=generated_tokens_array,
        compile_elapsed=compile_elapsed,
        generation_elapsed=generation_elapsed,
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
    max_new_tokens = 400
    profiler_started = False

    wandb_logging.init_wandb(
        project="gemma-27b-inference",
        config={
            "max_new_tokens": max_new_tokens,
            "enable_profiler": ENABLE_PROFILER,
            "profile_logdir": PROFILE_LOGDIR,
            "profile_session_prefix": PROFILE_SESSION_PREFIX,
        },
    )

    try:
        profiler_started = start_profiler()

        print("Loading weights from Orbax checkpoint...")
        params = load_params(DEFAULT_ORBAX_CHECKPOINT)
        print("Weights loaded.")

        prompt_batch = build_prompt_batch(get_prompts())
        kv_cache_len = 1024 + max_new_tokens
        assert max_new_tokens + prompt_batch.max_prompt_tokens < kv_cache_len

        print("Processing prompts (prefill)...")
        prefill_result = prefill(params, prompt_batch, kv_cache_len)

        print("Prompts processed.")
        print("Generating batched completions...")
        init_carry = (
            prefill_result.logits,
            prompt_batch.prompt_lengths_array,
            prefill_result.ks_cached,
            prefill_result.vs_cached,
        )
        decode_result = decode(params, init_carry, max_new_tokens)

        generated_tokens_batch = np.asarray(
            decode_result.generated_tokens_array
        ).T.tolist()
        print_final_outputs(prompt_batch.prompt_tokens_batch, generated_tokens_batch)
        report_and_log_metrics(
            prefill_elapsed=prefill_result.prefill_elapsed,
            generation_elapsed=decode_result.generation_elapsed,
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
