from dotenv import load_dotenv

load_dotenv()

import jax


def init_dist():
    try:
        jax.distributed.initialize()
    except (ValueError, RuntimeError):
        print("Single host mode")

    # Distributed training
    num_devices: int = jax.device_count()
    print(f"Number of devices: {num_devices}")
    print(f"Devices: {jax.devices()}")
    print(f"Local devices: {jax.local_devices()}")
    print(f"Backend: {jax.default_backend()}")


if __name__ == "__main__":
    from utils.gcp import init_gcp_logging

    init_gcp_logging()

    # must be called before any importing anything that might use JAX
    init_dist()

    from core.supervised_train import main

    main()
