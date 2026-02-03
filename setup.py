#!/usr/bin/env python3
"""
Zero-dependency setup script for JAX Gemma (1B/27B).
Runs on Compute Engine environments.

Responsibilities:
1. Configure Git (for dev/debug).
2. Install `uv`.
3. Sync dependencies (`uv sync`).
4. Download model weights from GCS via `gcloud storage`.
5. Launch `main.py`.
"""

import os
import subprocess
import shutil
import argparse
import sys

# --- Configuration ---
GIT_USER_NAME = "Dominik Farr"
GIT_USER_EMAIL = "dominik.farr@icloud.com"
MODEL_CONFIG = {
    "1b": {
        "bucket": "gemma_tmp_12342378236hf",
        "mount_dir": "data/gemma-3-1b",
        "sentinel": "model_stacked_pt.safetensors",
    },
    "27b": {
        "bucket": "gemma-3-weights-231d4b",
        "mount_dir": "data/gemma-3-27b",
        "only_dir": "gemma-weights",
        "sentinel": "model.safetensors.index.json",
    },
}


def run(cmd, check=True, shell=False):
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    subprocess.run(cmd, check=check, shell=shell)


def setup_git():
    print("--- Setting up Git ---")
    run(["git", "config", "--global", "user.name", GIT_USER_NAME], check=False)
    run(["git", "config", "--global", "user.email", GIT_USER_EMAIL], check=False)


def install_uv():
    if shutil.which("uv"):
        print("--- uv already installed ---")
    else:
        print("--- Installing uv ---")
        cmd = "curl -LsSf https://astral.sh/uv/install.sh | sh"
        run(cmd, shell=True)

    # Add to PATH for this session
    home = os.path.expanduser("~")
    for path in [".local/bin", ".cargo/bin"]:
        uv_bin = os.path.join(home, path)
        if os.path.exists(uv_bin) and uv_bin not in os.environ["PATH"]:
            os.environ["PATH"] = uv_bin + os.pathsep + os.environ["PATH"]


def get_metadata(attribute):
    try:
        result = subprocess.run(
            [
                "curl",
                "-s",
                "-H",
                "Metadata-Flavor: Google",
                f"http://metadata.google.internal/computeMetadata/v1/instance/{attribute}",
            ],
            capture_output=True,
            text=True,
            timeout=1,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def sync_dependencies():
    print("--- Syncing dependencies with uv ---")

    cmd = ["uv", "sync", "--frozen", "--no-dev"]

    accel_type = get_metadata("attributes/accelerator-type")

    is_tpu = False
    is_gpu = False

    if accel_type and ("tpu" in accel_type.lower() or "litepod" in accel_type.lower()):
        print(f"TPU detected via metadata: {accel_type}")
        is_tpu = True
    elif shutil.which("nvidia-smi"):
        print("GPU detected via nvidia-smi")
        is_gpu = True

    if not is_tpu and not is_gpu and os.environ.get("TPU_NAME"):
        is_tpu = True
        print("TPU detected via TPU_NAME env var")

    if is_tpu:
        print("Enabling 'tpu' extra")
        cmd.extend(["--extra", "tpu"])
    elif is_gpu:
        print("Enabling 'cuda' extra")
        cmd.extend(["--extra", "cuda"])
    else:
        print("No accelerator detected: CPU only")

    run(cmd)


def ensure_gcloud_storage():
    if shutil.which("gcloud"):
        return
    print("ERROR: gcloud CLI not found on PATH.")
    print("Install the Google Cloud SDK and authenticate, then re-run setup.")
    sys.exit(1)


def download_weights(model_size: str) -> None:
    config = MODEL_CONFIG[model_size]
    mount_dir = config["mount_dir"]
    sentinel_path = os.path.join(mount_dir, config["sentinel"])

    if os.path.exists(sentinel_path):
        print(f"--- Weights already present at {sentinel_path} ---")
        return

    ensure_gcloud_storage()

    bucket = config["bucket"]
    only_dir = config.get("only_dir")

    if only_dir:
        # Copy contents so weights land directly under mount_dir.
        source = f"gs://{bucket}/{only_dir}/*"
    else:
        source = f"gs://{bucket}/"

    label = "1B" if model_size == "1b" else "27B"
    print(f"--- Downloading {label} weights from {source} ---")

    os.makedirs(mount_dir, exist_ok=True)
    run(["gcloud", "storage", "cp", "-r", source, mount_dir])

    if not os.path.exists(sentinel_path):
        print(f"ERROR: Expected weights file missing after download: {sentinel_path}")
        sys.exit(1)


def create_env_file():
    key_path = os.path.join(os.getcwd(), "ops", "gemma-tpu-writer-key.json")
    if os.path.exists(key_path):
        print(f"--- Configuring .env with key: {key_path} ---")
        with open(".env", "w") as f:
            f.write(f"GOOGLE_APPLICATION_CREDENTIALS={key_path}\n")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path


def main():
    parser = argparse.ArgumentParser(description="Setup JAX environment")
    parser.add_argument(
        "model_size",
        choices=sorted(MODEL_CONFIG.keys()),
        help="Model size to download weights for (1b or 27b).",
    )
    parser.add_argument(
        "--run-main", action="store_true", help="Launch main.py after setup"
    )
    args, unknown_args = parser.parse_known_args()

    create_env_file()

    setup_git()
    install_uv()
    sync_dependencies()
    download_weights(args.model_size)

    if args.run_main:
        print("--- Launching main.py ---")
        env = os.environ.copy()
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        is_gpu = shutil.which("nvidia-smi") is not None
        is_tpu = (
            get_metadata("attributes/accelerator-type")
            or os.environ.get("TPU_NAME")
            or os.path.exists("/dev/accel0")
        )

        if not is_gpu and not is_tpu:
            print("Forcing JAX to CPU mode")
            env["JAX_PLATFORMS"] = "cpu"

        cmd = ["uv", "run", "python", "-m", "main"] + unknown_args

        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env)
    else:
        print("--- Setup Complete ---")
        print("To run training: uv run python -m main")


if __name__ == "__main__":
    main()
