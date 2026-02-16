from __future__ import annotations

import time

import jax
from jax.experimental import multihost_utils
import numpy as np


def build_shared_profile_options(
    session_prefix: str,
) -> tuple[jax.profiler.ProfileOptions, str]:
    """Build profiler options with one session_id shared across all hosts."""
    is_source = jax.process_index() == 0
    seed = np.array([time.time_ns() if is_source else 0], dtype=np.int64)
    shared_seed = multihost_utils.broadcast_one_to_all(seed, is_source=is_source)
    session_id = f"{session_prefix}_{int(shared_seed[0])}"

    options = jax.profiler.ProfileOptions()
    options.session_id = session_id
    return options, session_id
