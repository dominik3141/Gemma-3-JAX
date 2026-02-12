"""Inference script entrypoint for batched cached forward generation."""

import logging
import os
import subprocess
import time

import jax
import jax.experimental.multihost_utils as multihost_utils
import jax.numpy as jnp
import numpy as np

from core.gemma_forward import config
from core.gemma_forward_inference import forward_single
from utils.params_io_27b import DEFAULT_ORBAX_CHECKPOINT, load_params
from utils.tokenize_text import detokenize_ids, tokenize_text
import utils.wandb_logging as wandb_logging

ENABLE_PROFILER = True
PROFILE_GCS_BUCKET = "gs://gemma-3-training-profiles-20260207-165411-1d9c5e-euw4"
PROFILE_TRACE_DIR = "artifacts/profile"
# Default behavior keeps one profile artifact set per run.
# Set to True if you want profile uploads from every host.
UPLOAD_ALL_HOST_PROFILES = True
LOGGER = logging.getLogger(__name__)


def _profile_run_id() -> str:
    local_run_id = np.int64(time.time())
    shared_run_id = multihost_utils.broadcast_one_to_all(local_run_id)
    return str(int(np.asarray(shared_run_id).item()))


def _log_completed_process_output(result: subprocess.CompletedProcess[str]) -> None:
    for line in result.stdout.splitlines():
        if line.strip():
            LOGGER.info("[gsutil stdout] %s", line)
    for line in result.stderr.splitlines():
        if line.strip():
            LOGGER.warning("[gsutil stderr] %s", line)


def _maybe_upload_profile_artifacts(profile_run_id: str) -> None:
    try:
        process_index = jax.process_index()
        if UPLOAD_ALL_HOST_PROFILES:
            destination = (
                f"{PROFILE_GCS_BUCKET}/inference/{profile_run_id}/host_{process_index}/"
            )
        elif process_index == 0:
            destination = f"{PROFILE_GCS_BUCKET}/inference/{profile_run_id}/"
        else:
            LOGGER.info(
                "Skipping profile upload on host_%s; host_0 uploads for this run.",
                process_index,
            )
            destination = None

        if destination is not None and os.path.isdir(PROFILE_TRACE_DIR):
            LOGGER.info("Uploading profiler artifacts to %s...", destination)
            result = subprocess.run(
                [
                    "gsutil",
                    "-m",
                    "cp",
                    "-c",
                    "-r",
                    PROFILE_TRACE_DIR,
                    destination,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            _log_completed_process_output(result)
            if result.returncode == 0:
                LOGGER.info("Profiler upload complete.")
            else:
                LOGGER.warning(
                    "Profiler upload finished with non-zero exit code (%s).",
                    result.returncode,
                )
        elif destination is not None:
            LOGGER.info(
                "No profiler artifacts found at %s; skipping upload.", PROFILE_TRACE_DIR
            )
    except Exception as exc:
        LOGGER.warning(
            "Failed to upload profiler artifacts (best effort, continuing): %s",
            exc,
        )


def main() -> None:
    max_new_tokens = 400
    profile_run_id = _profile_run_id() if ENABLE_PROFILER else None
    profiler_started = False

    wandb_logging.init_wandb(
        project="gemma-27b-inference",
        config={
            "max_new_tokens": max_new_tokens,
            "enable_profiler": ENABLE_PROFILER,
            "profile_trace_dir": PROFILE_TRACE_DIR,
        },
    )
    try:
        if ENABLE_PROFILER:
            assert profile_run_id is not None
            LOGGER.info("Starting JAX profiler trace...")
            LOGGER.info("Profile run id: %s", profile_run_id)
            try:
                jax.profiler.start_trace(PROFILE_TRACE_DIR)
                profiler_started = True
            except Exception as exc:
                LOGGER.warning("Failed to start JAX profiler trace: %s", exc)

        print("Loading weights from Orbax checkpoint...")
        with jax.profiler.TraceAnnotation("inference/load_params"):
            params = load_params(DEFAULT_ORBAX_CHECKPOINT)
        print("Weights loaded.")

        prompts = [
            "The capital of France is Paris. The capital of Germany is",
            "The capital of France is Paris. The capital of Italy is",
            "The capital of France is Paris. The capital of Spain is",
            "The capital of France is Paris. The capital of Portugal is",
        ]

        prompt_tokens_batch: list[list[int]] = []
        for prompt_idx, prompt in enumerate(prompts):
            tokens = tokenize_text(prompt)
            if not tokens or tokens[0] != 2:
                tokens = [2] + tokens
            prompt_tokens_batch.append(tokens)
            print(f"Prompt {prompt_idx}: '{prompt}'")
            print(f"Prompt {prompt_idx} tokens ({len(tokens)}): {tokens}")

        batch_size = len(prompt_tokens_batch)
        max_prompt_tokens = max(len(tokens) for tokens in prompt_tokens_batch)

        kv_cache_len = 1024 + max_new_tokens
        assert max_new_tokens + max_prompt_tokens < kv_cache_len

        print("Processing prompts (prefill)...")

        prefill_logits_per_prompt: list[jax.Array] = []
        Ks_cached_per_prompt: list[jax.Array] = []
        Vs_cached_per_prompt: list[jax.Array] = []
        prompt_lengths: list[int] = []

        prefill_start = time.perf_counter()
        with jax.profiler.TraceAnnotation("inference/prefill"):
            for prompt_tokens in prompt_tokens_batch:
                Ks_cached = jnp.zeros(
                    (
                        config.num_layers,
                        kv_cache_len,
                        config.num_key_value_heads,
                        config.head_dim,
                    ),
                    dtype=jnp.bfloat16,
                )
                Vs_cached = jnp.zeros(
                    (
                        config.num_layers,
                        kv_cache_len,
                        config.num_key_value_heads,
                        config.d_kvq,
                    ),
                    dtype=jnp.bfloat16,
                )

                logits = None
                for pos, token in enumerate(prompt_tokens):
                    token_id = jnp.array(token, dtype=jnp.int32)
                    logits, Ks_cached, Vs_cached = forward_single(
                        token_id, params, pos, Ks_cached, Vs_cached
                    )
                assert logits is not None
                logits.block_until_ready()

                prefill_logits_per_prompt.append(logits)
                Ks_cached_per_prompt.append(Ks_cached)
                Vs_cached_per_prompt.append(Vs_cached)
                prompt_lengths.append(len(prompt_tokens))
        prefill_elapsed = time.perf_counter() - prefill_start

        logits = jnp.stack(prefill_logits_per_prompt, axis=0)
        Ks_cached = jnp.stack(Ks_cached_per_prompt, axis=0)
        Vs_cached = jnp.stack(Vs_cached_per_prompt, axis=0)

        print("Prompts processed.")
        print("Generating batched completions...")

        curr_pos = jnp.array(prompt_lengths, dtype=jnp.int32)

        def forward_single_batch(token_ids, params, pos, Ks_cached, Vs_cached):
            return jax.vmap(forward_single, in_axes=(0, None, 0, 0, 0))(
                token_ids, params, pos, Ks_cached, Vs_cached
            )

        def scan_body(params, carry, _):
            logits, curr_pos, Ks, Vs = carry
            next_tokens = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            new_logits, new_Ks, new_Vs = forward_single_batch(
                next_tokens, params, curr_pos, Ks, Vs
            )
            return (new_logits, curr_pos + 1, new_Ks, new_Vs), next_tokens

        def run_scan(params, init_carry):
            return jax.lax.scan(
                lambda carry, xs: scan_body(params, carry, xs),
                init_carry,
                None,
                length=max_new_tokens,
            )

        # Initial carry
        init_carry = (logits, curr_pos, Ks_cached, Vs_cached)

        # Measure compilation time for jitted scan
        compile_start = time.perf_counter()
        with jax.profiler.TraceAnnotation("inference/compile"):
            run_scan_jit = jax.jit(run_scan)
            # Explicitly lower and compile to avoid running the computation
            lowered = run_scan_jit.lower(params, init_carry)
            compiled_scan = lowered.compile()
        compile_elapsed = time.perf_counter() - compile_start
        print(f"JIT compilation time: {compile_elapsed:.3f}s")

        generation_start = time.perf_counter()
        with jax.profiler.TraceAnnotation("inference/generate"):
            final_carry, generated_tokens_array = compiled_scan(params, init_carry)
            logits, curr_pos, Ks_cached, Vs_cached = final_carry
            generated_tokens_array.block_until_ready()
        generation_elapsed = time.perf_counter() - generation_start

        generated_tokens_batch = np.asarray(generated_tokens_array).T.tolist()

        for prompt_idx, (prompt_tokens, generated_tokens) in enumerate(
            zip(prompt_tokens_batch, generated_tokens_batch)
        ):
            final_text = detokenize_ids(prompt_tokens + generated_tokens)
            print(f"Final output {prompt_idx}: {final_text}")

        total_prefill_tokens = sum(prompt_lengths)
        total_generated_tokens = sum(len(tokens) for tokens in generated_tokens_batch)

        prefill_tokens_per_s = (
            total_prefill_tokens / prefill_elapsed if prefill_elapsed > 0 else 0.0
        )
        generation_tokens_per_s_batch = (
            total_generated_tokens / generation_elapsed
            if generation_elapsed > 0
            else 0.0
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
    finally:
        try:
            if profiler_started:
                try:
                    jax.profiler.stop_trace()
                    LOGGER.info("Stopped JAX profiler trace.")
                except Exception as exc:
                    LOGGER.warning("Failed to stop JAX profiler trace: %s", exc)

                if profile_run_id is not None:
                    _maybe_upload_profile_artifacts(profile_run_id)
        finally:
            wandb_logging.finish_wandb()


if __name__ == "__main__":
    main()
