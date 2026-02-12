import logging
import os
import subprocess
import time

import jax
import jax.experimental.multihost_utils as multihost_utils
import numpy as np
import optax

import utils.wandb_logging as wandb_logging
from core.rl import (
    BETA,
    EPSILON,
    GROUP_SIZE,
    LEARNING_RATE,
    MAX_RESPONSE_LENGTH,
    MAX_ROOT,
    MIN_ROOT,
    NUM_GROUPS_PER_UPDATE,
    SAMPLE_TEMP,
    train_loop,
)
from utils.params_io_27b import DEFAULT_ORBAX_CHECKPOINT, load_params, save_params

REFERENCE_PARAMS_UPDATE_INTERVAL = 400  # as suggested by the R1 paper
CHECKPOINT_INTERVAL = 25
ENABLE_PROFILER = False
PROFILE_STOP_STEP = 5
PROFILE_GCS_BUCKET = "gs://gemma-3-training-profiles-20260207-165411-1d9c5e"
# Default behavior keeps one profile artifact set per run.
# Set to True if you want profile uploads from every host.
UPLOAD_ALL_HOST_PROFILES = False
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
            destination = f"{PROFILE_GCS_BUCKET}/{profile_run_id}/host_{process_index}/"
        elif process_index == 0:
            destination = f"{PROFILE_GCS_BUCKET}/{profile_run_id}/"
        else:
            LOGGER.info(
                "Skipping profile upload on host_%s; host_0 uploads for this run.",
                process_index,
            )
            destination = None

        if destination is not None and os.path.isdir("artifacts"):
            LOGGER.info("Uploading artifacts to %s...", destination)
            result = subprocess.run(
                [
                    "gsutil",
                    "-m",
                    "cp",
                    "-c",
                    "-r",
                    "artifacts",
                    destination,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            _log_completed_process_output(result)
            if result.returncode == 0:
                LOGGER.info("Upload complete.")
            else:
                LOGGER.warning(
                    "Profile upload finished with non-zero exit code (%s); continuing training.",
                    result.returncode,
                )
        elif destination is not None:
            LOGGER.info("No local artifacts directory found; skipping profile upload.")
    except Exception as exc:
        LOGGER.warning(
            "Failed to upload artifacts (best effort, continuing training): %s",
            exc,
        )


def main() -> None:
    key = jax.random.PRNGKey(42)
    params = load_params(
        DEFAULT_ORBAX_CHECKPOINT,
        mesh_factory=lambda: jax.sharding.Mesh(jax.devices(), axis_names=("model",)),
    )

    optimizer_state = optax.adam(LEARNING_RATE).init(params)
    params_ref = params
    i = 0

    wandb_logging.init_wandb(
        project="gemma-27b-r1-zero",
        config={
            "max_root": MAX_ROOT,
            "min_root": MIN_ROOT,
            "sample_temp": SAMPLE_TEMP,
            "group_size": GROUP_SIZE,
            "max_response_length": MAX_RESPONSE_LENGTH,
            "epsilon": EPSILON,
            "beta": BETA,
            "num_groups_per_update": NUM_GROUPS_PER_UPDATE,
            "learning_rate": LEARNING_RATE,
        },
    )

    profile_run_id = _profile_run_id() if ENABLE_PROFILER else None

    try:
        if ENABLE_PROFILER:
            assert profile_run_id is not None
            LOGGER.info("Starting JAX profiler trace...")
            LOGGER.info("Profile run id: %s", profile_run_id)
            jax.profiler.start_trace("artifacts/profile")

        while True:
            wandb_logging.set_step(i)

            if ENABLE_PROFILER and i == PROFILE_STOP_STEP:
                try:
                    jax.profiler.stop_trace()
                    LOGGER.info("Stopped JAX profiler trace.")
                except Exception as exc:
                    LOGGER.warning("Failed to stop JAX profiler trace: %s", exc)

                assert profile_run_id is not None
                _maybe_upload_profile_artifacts(profile_run_id)

            params, loss, format_pct, correct_pct, optimizer_state = train_loop(
                key, params, params_ref, optimizer_state
            )
            key, _ = jax.random.split(key)
            loss, format_pct, correct_pct = jax.device_get(
                (loss, format_pct, correct_pct)
            )

            LOGGER.info(
                "%s, Loss: %s, Format: %.2f%%, Correct: %.2f%%",
                i,
                loss,
                format_pct * 100,
                correct_pct * 100,
            )
            wandb_logging.log_metrics(
                {
                    "train/loss": float(loss),
                    "train/format_pct": float(format_pct * 100.0),
                    "train/correct_pct": float(correct_pct * 100.0),
                },
                step=i,
            )

            i += 1

            if i % CHECKPOINT_INTERVAL == 0:
                try:
                    save_params(params)
                    LOGGER.info("Saved parameters")
                except Exception as exc:
                    LOGGER.warning("Failed to save parameters: %s", exc)

            if i % REFERENCE_PARAMS_UPDATE_INTERVAL == 0:
                params_ref = params
                LOGGER.info("Updated reference parameters")
    finally:
        try:
            save_params(params)
            LOGGER.info("Uploaded final parameters")
        except Exception as exc:
            LOGGER.warning("Failed to upload final parameters: %s", exc)
        finally:
            wandb_logging.finish_wandb()


if __name__ == "__main__":
    main()
