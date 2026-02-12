from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping
import os

import jax


@dataclass
class WandbState:
    run: Any | None
    enabled: bool
    process_index: int
    step: int


_STATE = WandbState(run=None, enabled=False, process_index=-1, step=0)
LOGGER = logging.getLogger(__name__)


def _disable_state(process_index: int | None = None) -> None:
    global _STATE
    if process_index is None:
        process_index = _STATE.process_index
    _STATE = WandbState(run=None, enabled=False, process_index=process_index, step=0)


def _is_enabled() -> bool:
    return _STATE.enabled and _STATE.run is not None


def init_wandb(
    project: str = "gemma-27b-r1-zero",
    entity: str | None = None,
    run_name: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> Any | None:
    global _STATE
    process_index = jax.process_index()

    if process_index != 0:
        _disable_state(process_index=process_index)
        return None

    if _STATE.enabled and _STATE.run is not None:
        return _STATE.run

    if not os.environ.get("WANDB_API_KEY"):
        LOGGER.info("WANDB_API_KEY not set; W&B disabled.")
        _disable_state(process_index=process_index)
        return None

    try:
        import wandb
    except Exception as exc:
        LOGGER.warning("Failed to import wandb: %s", exc)
        _disable_state(process_index=process_index)
        return None

    try:
        settings = wandb.Settings(start_method="thread")
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config,
            settings=settings,
        )
    except Exception as exc:
        LOGGER.warning("W&B init failed: %s", exc)
        _disable_state(process_index=process_index)
        return None

    _STATE = WandbState(run=run, enabled=True, process_index=process_index, step=0)
    return run


def set_step(step: int) -> None:
    _STATE.step = int(step)


def log_metrics(metrics: Mapping[str, Any], step: int | None = None) -> None:
    if not _is_enabled():
        return
    if not metrics:
        return

    use_step = _STATE.step if step is None else step
    try:
        import wandb

        wandb.log(dict(metrics), step=use_step)
    except Exception as exc:
        LOGGER.warning("W&B log failed: %s", exc)


def log_sample(sample: Mapping[str, Any], step: int | None = None) -> None:
    if not _is_enabled():
        return
    if not sample:
        return

    use_step = _STATE.step if step is None else step
    try:
        import wandb

        data = dict(sample)
        if "step" not in data:
            data["step"] = use_step
        columns = list(data.keys())
        row = [data[col] for col in columns]
        table = wandb.Table(columns=columns, data=[row])
        wandb.log({"samples": table}, step=use_step)
    except Exception as exc:
        LOGGER.warning("W&B sample log failed: %s", exc)


def finish_wandb() -> None:
    if not _is_enabled():
        return

    try:
        if _STATE.run is not None:
            _STATE.run.finish()
    except Exception as exc:
        LOGGER.warning("W&B finish failed: %s", exc)
    finally:
        _disable_state()
