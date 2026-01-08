"""GCP infrastructure utilities."""


def is_gcp_machine() -> bool:
    """Detect if running on a GCP machine by checking metadata server availability."""
    import socket

    try:
        # Try to connect to GCP metadata server (only available on GCP VMs)
        socket.create_connection(("metadata.google.internal", 80), timeout=0.5)
        return True
    except (socket.error, OSError, TimeoutError):
        return False


def init_gcp_logging():
    """Initialize GCP Cloud Logging as an additional backend without replacing stdout."""
    if not is_gcp_machine():
        print("Not on GCP machine, skipping cloud logging initialization")
        return

    import google.cloud.logging
    from google.cloud.logging.handlers import CloudLoggingHandler
    import google.auth
    from google.oauth2 import service_account
    import logging
    import sys
    from pathlib import Path

    # Load credentials from the ops directory
    credentials_path = (
        Path(__file__).parent.parent / "ops" / "gemma-tpu-writer-key.json"
    )
    credentials = service_account.Credentials.from_service_account_file(
        str(credentials_path)
    )

    # Initialize GCP Client and Handler with custom labels
    client = google.cloud.logging.Client(credentials=credentials)
    handler = CloudLoggingHandler(client, labels={"project": "jax-gemma"})

    # Add GCP handler to the root logger to send logs without replacing existing ones
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)

    # Use a dedicated logger for stdout capture to avoid feedback loops
    print_logger = logging.getLogger("stdout_gcp")
    print_logger.addHandler(handler)
    print_logger.propagate = False

    original_stdout_write = sys.stdout.write

    def tee_write(msg: str):
        if msg.strip():
            print_logger.info(msg.strip())
        return original_stdout_write(msg)

    sys.stdout.write = tee_write
