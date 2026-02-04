import argparse
import os
import pathlib
import sys

from utils.load_sharded_host import load_stacked_sharded_model_host


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TPU-safe RL entrypoint (host load, then TPU init)."
    )
    parser.add_argument("--weights-dir", default="data/gemma-3-27b")
    parser.add_argument("--max-layers", type=int, default=None)
    return parser.parse_args()


def _configure_tpu_watchdog(timeout_seconds: int = 600, disable: bool = True) -> None:
    flags = []
    if disable:
        flags.append("--megascale_enable_watchdog=false")
        flags.append("--threadmanager_overseer_enable_watchdog=false")
    if timeout_seconds > 0:
        timeout_value = f"{timeout_seconds}s"
        flags.append(
            f"--megascale_callback_registry_watchdog_timeout={timeout_value}"
        )
        flags.append(f"--megascale_graph_executor_watchdog_timeout={timeout_value}")
        flags.append(f"--threadmanager_overseer_watchdog_s={timeout_seconds}")

    if not flags:
        return

    flag_str = " ".join(flags)
    for env_key in ("TPU_INIT_ARGS", "LIBTPU_INIT_ARGS"):
        existing = os.environ.get(env_key, "")
        if flag_str in existing:
            continue
        os.environ[env_key] = (existing + " " + flag_str).strip()


def _configure_tpu_health_check(timeout_ms: int = 600_000) -> None:
    flag = f"--xla_tpu_health_check_timeout_threshold_ms={timeout_ms}"
    libtpu_path = (
        pathlib.Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "libtpu"
        / "libtpu.so"
    )
    if not libtpu_path.exists():
        return
    try:
        data = libtpu_path.read_bytes()
    except OSError:
        return
    if flag.encode() not in data:
        return
    existing = os.environ.get("LIBTPU_INIT_ARGS", "")
    if flag in existing:
        return
    os.environ["LIBTPU_INIT_ARGS"] = (existing + " " + flag).strip()


def _device_put_with_sharding(
    host_params: dict,
    sharding_specs: dict,
):
    import jax

    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices, axis_names=("model",))

    params = {}
    for key, array in host_params.items():
        spec_tuple = sharding_specs[key]
        sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(*spec_tuple)
        )
        params[key] = jax.device_put(array, sharding)
        host_params[key] = None
    host_params.clear()
    return params, mesh


def main() -> None:
    args = _parse_args()

    _configure_tpu_watchdog(600, disable=True)
    _configure_tpu_health_check(600_000)

    print("Loading weights on host...", flush=True)
    host_params, sharding_specs = load_stacked_sharded_model_host(
        args.weights_dir, max_layers=args.max_layers
    )
    layer_key = next(
        k for k in host_params.keys() if "layers_stacked" in k
    )
    num_layers = host_params[layer_key].shape[0]
    print("Host load complete. Initializing TPU...", flush=True)

    import jax
    import optax
    from core import rl as rl_mod
    from core import gemma_forward, gemma_forward_inference

    gemma_forward.config.num_layers = num_layers
    gemma_forward_inference.config.num_layers = num_layers

    print("Transferring weights to TPU...", flush=True)
    params, _mesh = _device_put_with_sharding(host_params, sharding_specs)
    print("Device transfer complete. Starting training...", flush=True)

    key = jax.random.PRNGKey(42)
    optimizer_state = optax.adam(rl_mod.LEARNING_RATE).init(params)

    params_ref = params
    i = 0
    try:
        while True:
            params, loss, format_pct, correct_pct, optimizer_state = (
                rl_mod.train_loop_host_rewards(key, params, params_ref, optimizer_state)
            )
            key, _ = jax.random.split(key)
            print(
                f"{i}, Loss: {loss}, Format: {format_pct * 100:.2f}%, "
                f"Correct: {correct_pct * 100:.2f}%"
            )
            i += 1

            if i % 100 == 0:
                rl_mod.save_params(params, upload_to_gcs=False)
                print("Saved parameters")

            if i % 400 == 0:
                params_ref = params
                print("Updated reference parameters")
    finally:
        try:
            rl_mod.save_params(params, upload_to_gcs=True)
            print("Uploaded final parameters")
        except Exception as exc:
            print(f"Failed to upload final parameters: {exc}")


if __name__ == "__main__":
    main()
