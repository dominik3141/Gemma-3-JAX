from dotenv import load_dotenv

load_dotenv()

import os

# Configure HLO dumping
# Must be done before JAX is initialized/used
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_dump_to=artifacts/hlo"
    " --xla_dump_hlo_as_text"
    " --xla_dump_hlo_as_html"
    " --xla_dump_hlo_as_proto"
)

import jax

# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"
# print("Forcing the use of 16 devices.")


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

    from core.rl import main

    main()

    # lowered = jax.jit(main).lower()
    # hlo_text = lowered.compile().as_text()

    # print(hlo_text)
