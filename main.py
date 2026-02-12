from dotenv import load_dotenv

load_dotenv()

import logging
import os
from typing import Final

# Configure HLO dumping
# Must be done before JAX is initialized/used
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_dump_to=artifacts/hlo"
    " --xla_dump_hlo_as_text"
    " --xla_dump_hlo_as_html"
    " --xla_dump_hlo_as_proto"
)

# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"
# print("Forcing the use of 16 devices.")

WANDB_SECRET_PROJECT: Final[str] = os.environ.get(
    "WANDB_SECRET_PROJECT", "default-482802"
)
WANDB_SECRET_NAME: Final[str] = os.environ.get("WANDB_SECRET_NAME", "wandb-api-key")
LOGGER = logging.getLogger(__name__)


def init_dist():
    import jax

    try:
        jax.distributed.initialize()
    except (ValueError, RuntimeError):
        LOGGER.info("Single host mode")

    # Distributed training
    num_devices: int = jax.device_count()
    LOGGER.info("Number of devices: %s", num_devices)
    LOGGER.info("Devices: %s", jax.devices())
    LOGGER.info("Local devices: %s", jax.local_devices())
    LOGGER.info("Backend: %s", jax.default_backend())


def configure_wandb_api_key_for_worker() -> bool:
    import jax

    process_index = jax.process_index()

    if process_index != 0:
        # Keep credentials only on worker 0.
        os.environ.pop("WANDB_API_KEY", None)
        return False

    if os.environ.get("WANDB_API_KEY"):
        return True

    secret_path = (
        f"projects/{WANDB_SECRET_PROJECT}/secrets/{WANDB_SECRET_NAME}/versions/latest"
    )

    try:
        from google.cloud import secretmanager
    except Exception as exc:
        LOGGER.warning(
            "google-cloud-secret-manager unavailable in current environment; "
            "could not load WANDB_API_KEY: %s",
            exc,
        )
        return False

    try:
        client = secretmanager.SecretManagerServiceClient()
        response = client.access_secret_version(request={"name": secret_path})
        api_key = response.payload.data.decode("utf-8").strip()
        if not api_key:
            raise RuntimeError("Secret payload is empty")

        os.environ["WANDB_API_KEY"] = api_key
        LOGGER.info(
            "Loaded WANDB_API_KEY from Secret Manager (%s/%s).",
            WANDB_SECRET_PROJECT,
            WANDB_SECRET_NAME,
        )
        return True
    except Exception as exc:
        LOGGER.warning(
            "Failed to load WANDB_API_KEY from Secret Manager (%s/%s): %s",
            WANDB_SECRET_PROJECT,
            WANDB_SECRET_NAME,
            exc,
        )
        return False


if __name__ == "__main__":
    from utils.gcp import init_logging

    init_logging(log_name="gemma-3-training")
    LOGGER.info("Logging initialized")

    # must be called before any importing anything that might use JAX
    init_dist()
    configure_wandb_api_key_for_worker()

    from entrypoints.gemma_forward_inference import main

    main()
