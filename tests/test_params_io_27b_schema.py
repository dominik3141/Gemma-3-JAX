import pathlib
import sys

import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from utils.params_io_27b import EXPECTED_TARGET_SPECS, SHARDING_PLAN


def test_sharding_plan_has_same_keys_as_targets() -> None:
    missing_specs = sorted(set(EXPECTED_TARGET_SPECS) - set(SHARDING_PLAN))
    extra_specs = sorted(set(SHARDING_PLAN) - set(EXPECTED_TARGET_SPECS))
    assert not missing_specs, f"Missing sharding specs for keys: {missing_specs}"
    assert not extra_specs, f"Unexpected sharding specs for keys: {extra_specs}"


def test_target_rank_matches_partition_spec_rank() -> None:
    for key, target in EXPECTED_TARGET_SPECS.items():
        spec = SHARDING_PLAN[key]
        assert len(target.shape) == len(tuple(spec)), (
            f"Spec rank mismatch for {key}: "
            f"shape rank {len(target.shape)} != spec rank {len(tuple(spec))}"
        )


def test_target_dtypes_are_bfloat16() -> None:
    for key, target in EXPECTED_TARGET_SPECS.items():
        assert target.dtype == jnp.bfloat16, (
            f"{key} has dtype {target.dtype}, expected bfloat16"
        )
