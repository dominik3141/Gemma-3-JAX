from dotenv import load_dotenv

load_dotenv()

from core.supervised_train import main
import jax


def init_dist():
    # init distributed training communications (blocking)
    jax.distributed.initialize()

    # Distributed training
    num_devices: int = jax.device_count()
    print(f"Number of devices: {num_devices}")
    print(f"Devices: {jax.devices()}")
    print(f"Local devices: {jax.local_devices()}")
    print(f"Backend: {jax.default_backend()}")


if __name__ == "__main__":
    init_dist()

    main()
