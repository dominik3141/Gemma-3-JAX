"""Inference script entrypoint for single-token cached forward generation."""

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
    max_new_tokens = 2000
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

        prompt = "The capital of France is Paris. The capital of Germany is"
        tokens = tokenize_text(prompt)

        if tokens[0] != 2:
            tokens = [2] + tokens

        print(f"Prompt: '{prompt}'")
        print(f"Tokens: {tokens}")

        kv_cache_len = 1024 + max_new_tokens
        assert max_new_tokens + len(tokens) < kv_cache_len

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

        print("Processing prompt (prefill)...")

        logits = None
        prefill_start = time.perf_counter()
        with jax.profiler.TraceAnnotation("inference/prefill"):
            for i, token in enumerate(tokens):
                token_id = jnp.array(token)
                logits, Ks_cached, Vs_cached = forward_single(
                    token_id, params, i, Ks_cached, Vs_cached
                )
                logits.block_until_ready()
        prefill_elapsed = time.perf_counter() - prefill_start

        print("Prompt processed.")
        print("Generating:")

        generated_tokens: list[int] = []
        curr_pos = len(tokens)

        def token_callback(token_id):
            token_val = int(token_id)
            token_text = detokenize_ids([token_val])
            if token_text:
                print(token_text, end="", flush=True)
            return jnp.array(token_val, dtype=jnp.int32)

        def scan_body(params, carry, _):
            logits, curr_pos, Ks, Vs = carry
            next_token = jnp.argmax(logits).astype(jnp.int32)

            # Print via callback
            jax.experimental.io_callback(
                token_callback,
                jax.ShapeDtypeStruct((), jnp.int32),
                next_token,
                ordered=False,
            )

            new_logits, new_Ks, new_Vs = forward_single(
                next_token, params, curr_pos, Ks, Vs
            )
            return (new_logits, curr_pos + 1, new_Ks, new_Vs), next_token

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

        generated_tokens = generated_tokens_array.tolist()
        print()  # Ensure newline after streaming

        final_text = detokenize_ids(tokens + generated_tokens)
        print(f"Final output: {final_text}")

        prefill_tokens_per_s = (
            len(tokens) / prefill_elapsed if prefill_elapsed > 0 else 0.0
        )
        generation_tokens_per_s = (
            len(generated_tokens) / generation_elapsed
            if generation_elapsed > 0 and generated_tokens
            else 0.0
        )
        print(
            f"Prefill avg: {prefill_tokens_per_s:.1f} tokens/s "
            f"({prefill_elapsed:.3f}s total for {len(tokens)} tokens)"
        )
        print(
            f"Generation avg: {generation_tokens_per_s:.1f} tokens/s "
            f"({generation_elapsed:.3f}s total for {len(generated_tokens)} tokens)"
        )
        wandb_logging.log_metrics(
            {"inference/tokens_per_second": float(generation_tokens_per_s)},
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
