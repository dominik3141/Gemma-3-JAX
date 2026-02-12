"""GCP infrastructure and logging utilities."""

from dataclasses import dataclass
import logging
import os
import queue
import sys
import threading
from typing import TextIO


_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_LOG_LEVEL = logging.INFO

_ORIGINAL_STDOUT: TextIO = sys.stdout
_ORIGINAL_STDERR: TextIO = sys.stderr
_ORIGINAL_EXCEPTHOOK = sys.excepthook
_ORIGINAL_THREAD_EXCEPTHOOK = threading.excepthook

_LOGGING_INITIALIZED = False
_STD_STREAMS_CAPTURED = False
_WARNINGS_CAPTURED = False
_EXCEPTION_HOOKS_INSTALLED = False


def _should_enable_gcp_logging() -> bool:
    if os.environ.get("GEMMA_DISABLE_GCP_LOGGING"):
        return False
    return True


class _StreamToLogger:
    """Tee stream writes to the original stream and to a logger."""

    def __init__(self, stream: TextIO, logger: logging.Logger, level: int) -> None:
        self._stream = stream
        self._logger = logger
        self._level = level
        self._line_buffer = ""

    def write(self, message: str) -> int:
        if not isinstance(message, str):
            message = str(message)

        written = self._stream.write(message)
        self._line_buffer += message

        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            line = line.rstrip()
            if line:
                self._logger.log(self._level, line)
        return written

    def flush(self) -> None:
        self._stream.flush()
        leftover = self._line_buffer.strip()
        if leftover:
            self._logger.log(self._level, leftover)
        self._line_buffer = ""

    @property
    def encoding(self):  # pragma: no cover - passthrough
        return getattr(self._stream, "encoding", None)

    @property
    def errors(self):  # pragma: no cover - passthrough
        return getattr(self._stream, "errors", None)

    def fileno(self) -> int:  # pragma: no cover - passthrough
        return self._stream.fileno()

    def isatty(self) -> bool:  # pragma: no cover - passthrough
        return self._stream.isatty()

    def writable(self) -> bool:  # pragma: no cover - passthrough
        return True

    def __getattr__(self, name: str):  # pragma: no cover - passthrough
        return getattr(self._stream, name)


@dataclass
class _AsyncLoggerState:
    logger: logging.Logger
    queue: queue.Queue
    thread: threading.Thread
    dropped: int = 0


_ASYNC_LOGGER_LOCK = threading.Lock()
_ASYNC_LOGGERS: dict[str, _AsyncLoggerState] = {}


def _ensure_console_handler(root_logger: logging.Logger) -> None:
    if any(
        getattr(handler, "_gemma_console_handler", False)
        for handler in root_logger.handlers
    ):
        return

    handler = logging.StreamHandler(stream=_ORIGINAL_STDERR)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.setLevel(_LOG_LEVEL)
    handler._gemma_console_handler = True  # type: ignore[attr-defined]
    root_logger.addHandler(handler)


def _ensure_gcp_handler(
    root_logger: logging.Logger,
    log_name: str,
    labels: dict[str, str],
    enable_gcp: bool,
) -> None:
    if any(
        getattr(handler, "_gemma_gcp_handler", False)
        for handler in root_logger.handlers
    ):
        return
    if not enable_gcp:
        return

    try:
        import google.cloud.logging
        from google.cloud.logging.handlers import CloudLoggingHandler
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "google-cloud-logging unavailable; Cloud Logging disabled: %s", exc
        )
        return

    try:
        client = google.cloud.logging.Client()
        handler = CloudLoggingHandler(client, name=log_name, labels=labels)
        handler.setLevel(_LOG_LEVEL)
        handler._gemma_gcp_handler = True  # type: ignore[attr-defined]
        root_logger.addHandler(handler)
        logging.getLogger(__name__).info(
            "Cloud Logging enabled with log name '%s'", log_name
        )
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Failed to initialize Cloud Logging handler: %s", exc
        )


def _ensure_warning_capture() -> None:
    global _WARNINGS_CAPTURED
    if _WARNINGS_CAPTURED:
        return

    logging.captureWarnings(True)
    _WARNINGS_CAPTURED = True


def _ensure_exception_hooks() -> None:
    global _EXCEPTION_HOOKS_INSTALLED
    if _EXCEPTION_HOOKS_INSTALLED:
        return

    def _log_uncaught_exception(exc_type, exc_value, exc_traceback) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            _ORIGINAL_EXCEPTHOOK(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger("uncaught").error(
            "Unhandled exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        _ORIGINAL_EXCEPTHOOK(exc_type, exc_value, exc_traceback)

    def _log_uncaught_thread_exception(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is KeyboardInterrupt:
            _ORIGINAL_THREAD_EXCEPTHOOK(args)
            return
        logging.getLogger("uncaught.thread").error(
            "Unhandled exception in thread %s",
            args.thread.name if args.thread is not None else "unknown",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        _ORIGINAL_THREAD_EXCEPTHOOK(args)

    sys.excepthook = _log_uncaught_exception
    threading.excepthook = _log_uncaught_thread_exception
    _EXCEPTION_HOOKS_INSTALLED = True


def _ensure_std_stream_capture() -> None:
    global _STD_STREAMS_CAPTURED
    if _STD_STREAMS_CAPTURED:
        return

    root_logger = logging.getLogger()
    stdout_logger = logging.getLogger("stdout")
    stderr_logger = logging.getLogger("stderr")

    for capture_logger, level in (
        (stdout_logger, logging.INFO),
        (stderr_logger, logging.ERROR),
    ):
        capture_logger.handlers.clear()
        capture_logger.setLevel(level)
        capture_logger.propagate = False
        capture_logger.addHandler(logging.NullHandler())
        for handler in root_logger.handlers:
            if getattr(handler, "_gemma_console_handler", False):
                continue
            capture_logger.addHandler(handler)

    sys.stdout = _StreamToLogger(_ORIGINAL_STDOUT, stdout_logger, logging.INFO)
    sys.stderr = _StreamToLogger(_ORIGINAL_STDERR, stderr_logger, logging.ERROR)
    _STD_STREAMS_CAPTURED = True


def _ensure_logging_configured() -> None:
    root_logger = logging.getLogger()
    _ensure_console_handler(root_logger)
    root_logger.setLevel(_LOG_LEVEL)


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
        except Exception:
            # Avoid recursive logging and keep worker alive under logging failures.
            _ORIGINAL_STDERR.write(f"Async logger {name} failed\n")
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


def init_logging(
    log_name: str = "gemma-3-training",
    labels: dict[str, str] | None = None,
) -> None:
    """Initialize terminal + Cloud Logging and capture std streams/exceptions.

    GCP handler behavior:
    - Cloud Logging is enabled by default.
    - Set GEMMA_DISABLE_GCP_LOGGING to disable Cloud Logging.
    """
    global _LOGGING_INITIALIZED

    if labels is None:
        labels = {"project": "jax-gemma"}

    root_logger = logging.getLogger()
    if not _LOGGING_INITIALIZED:
        _ensure_console_handler(root_logger)
        root_logger.setLevel(_LOG_LEVEL)
        _LOGGING_INITIALIZED = True

    _ensure_gcp_handler(
        root_logger,
        log_name=log_name,
        labels=labels,
        enable_gcp=_should_enable_gcp_logging(),
    )
    _ensure_warning_capture()
    _ensure_exception_hooks()
    _ensure_std_stream_capture()


def init_gcp_logging() -> None:
    """Backwards-compatible wrapper for older call sites."""
    init_logging()
