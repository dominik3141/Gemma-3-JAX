import argparse

from utils.load_sharded_host import load_stacked_sharded_model_host


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TPU-safe RL entrypoint (host load, then TPU init)."
    )
    parser.add_argument("--weights-dir", default="data/gemma-3-27b")
    parser.add_argument("--max-layers", type=int, default=None)
    return parser.parse_args()


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

    host_params, sharding_specs = load_stacked_sharded_model_host(
        args.weights_dir, max_layers=args.max_layers
    )

    import jax
    import optax
    from core import rl as rl_mod

    params, _mesh = _device_put_with_sharding(host_params, sharding_specs)

    key = jax.random.PRNGKey(42)
    optimizer_state = optax.adam(rl_mod.LEARNING_RATE).init(params)

    params_ref = params
    i = 0
    try:
        while True:
            params, loss, format_pct, correct_pct, optimizer_state = rl_mod.train_loop(
                key, params, params_ref, optimizer_state
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
