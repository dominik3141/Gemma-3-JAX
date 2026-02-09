"""GCP infrastructure utilities."""

from dataclasses import dataclass
import logging
import queue
import threading


def is_gcp_machine() -> bool:
    """Detect if running on a GCP machine by checking metadata server availability."""
    import socket

    try:
        # Try to connect to GCP metadata server (only available on GCP VMs)
        socket.create_connection(("metadata.google.internal", 80), timeout=0.5)
        return True
    except (socket.error, OSError, TimeoutError):
        return False


@dataclass
class _AsyncLoggerState:
    logger: logging.Logger
    queue: queue.Queue
    thread: threading.Thread
    dropped: int = 0


_ASYNC_LOGGER_LOCK = threading.Lock()
_ASYNC_LOGGERS: dict[str, _AsyncLoggerState] = {}


def _ensure_logging_configured() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO)


def _async_log_worker(
    name: str, log_queue: queue.Queue, logger: logging.Logger
) -> None:
    while True:
        message = log_queue.get()
        if message is None:
            log_queue.task_done()
            return
        try:
            logger.info(message)
        except Exception as exc:
            # Avoid recursive logging; keep failures from crashing the worker.
            print(f"Async logger {name} failed: {exc}")
        finally:
            log_queue.task_done()


def _ensure_async_logger(name: str, queue_size: int) -> _AsyncLoggerState:
    with _ASYNC_LOGGER_LOCK:
        state = _ASYNC_LOGGERS.get(name)
        if state is not None:
            return state

        _ensure_logging_configured()
        logger = logging.getLogger(name)
        log_queue = queue.Queue(maxsize=queue_size)
        thread = threading.Thread(
            target=_async_log_worker, args=(name, log_queue, logger), daemon=True
        )
        thread.start()

        state = _AsyncLoggerState(
            logger=logger,
            queue=log_queue,
            thread=thread,
            dropped=0,
        )
        _ASYNC_LOGGERS[name] = state
        return state


def log_text_async(name: str, message: str, queue_size: int = 256) -> None:
    if not message:
        return

    state = _ensure_async_logger(name, queue_size)
    try:
        state.queue.put_nowait(message)
    except queue.Full:
        # Drop on overload to avoid blocking training.
        state.dropped += 1


def init_gcp_logging():
    """Initialize GCP Cloud Logging as an additional backend without replacing stdout."""
    if not is_gcp_machine():
        print("Not on GCP machine, skipping cloud logging initialization")
        return

    import google.cloud.logging
    from google.cloud.logging.handlers import CloudLoggingHandler
    import logging
    import sys

    # Initialize GCP Client and Handler with custom labels using ADC metadata auth.
    client = google.cloud.logging.Client()
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
