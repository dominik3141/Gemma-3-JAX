import logging

import jax
import jax.experimental.multihost_utils as multihost_utils
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
from utils.profiling import build_shared_profile_options

REFERENCE_PARAMS_UPDATE_INTERVAL = 400  # as suggested by the R1 paper
CHECKPOINT_INTERVAL = 25
ENABLE_PROFILER = True
PROFILE_STOP_STEP = 5
PROFILE_LOGDIR = "gs://gemma-3-training-profiles-20260207-165411-1d9c5e-euw4"
PROFILE_SESSION_PREFIX = "gemma_rl"
PROFILE_START_BARRIER_NAME = "gemma_rl_profile_start"
PROFILE_STOP_BARRIER_NAME = "gemma_rl_profile_stop"
LOGGER = logging.getLogger(__name__)


def start_profiler() -> bool:
    if not ENABLE_PROFILER:
        return False

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


def stop_profiler() -> None:
    LOGGER.info(
        "Synchronizing hosts before profiler stop (%s)...",
        PROFILE_STOP_BARRIER_NAME,
    )
    multihost_utils.sync_global_devices(PROFILE_STOP_BARRIER_NAME)
    jax.profiler.stop_trace()
    LOGGER.info("Stopped JAX profiler trace.")


def main() -> None:
    key = jax.random.PRNGKey(42)
    mesh = jax.sharding.Mesh(jax.devices(), axis_names=("model",))
    params = load_params(DEFAULT_ORBAX_CHECKPOINT, mesh)

    optimizer_state = optax.contrib.muon(learning_rate=LEARNING_RATE).init(params)
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
            "enable_profiler": ENABLE_PROFILER,
            "profile_logdir": PROFILE_LOGDIR,
            "profile_stop_step": PROFILE_STOP_STEP,
            "profile_session_prefix": PROFILE_SESSION_PREFIX,
        },
    )

    profiler_started = start_profiler()

    try:
        while True:
            wandb_logging.set_step(i)

            if profiler_started and i == PROFILE_STOP_STEP:
                stop_profiler()
                profiler_started = False

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
                save_params(params)
                LOGGER.info("Saved parameters")

            if i % REFERENCE_PARAMS_UPDATE_INTERVAL == 0:
                params_ref = params
                LOGGER.info("Updated reference parameters")
    finally:
        try:
            if profiler_started:
                stop_profiler()
            save_params(params)
            LOGGER.info("Uploaded final parameters")
        finally:
            wandb_logging.finish_wandb()


if __name__ == "__main__":
    main()
