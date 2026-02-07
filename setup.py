#!/usr/bin/env python3
"""
Zero-dependency setup script for JAX Gemma.
Runs on Compute Engine environments.

Responsibilities:
1. Configure Git (for dev/debug).
2. Install `uv`.
3. Sync dependencies (`uv sync`).
4. Download tokenizer files from GCS.
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
TOKENIZER_ROOT = "gs://gemma-3-weights-231d4b/gemma-weights"
TOKENIZER_DIR = "data/gemma-3-27b"
TOKENIZER_FILES = [
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
]
TOKENIZER_SENTINEL = os.path.join(TOKENIZER_DIR, "tokenizer.model")
WANDB_SECRET_PROJECT = "default-482802"
WANDB_SECRET_NAME = "wandb-api-key"


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


def ensure_gsutil() -> None:
    if shutil.which("gsutil"):
        # Fix for AttributeError: module 'lib' has no attribute 'X509_V_FLAG_NOTIFY_POLICY'
        # in gsutil on some Ubuntu/TPU VM images.
        print("--- Patching cryptography/pyOpenSSL for gsutil ---")
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--user",
                "--upgrade",
                "cryptography",
                "pyOpenSSL",
            ]
        )
        return
    print("ERROR: gsutil not found on PATH.")
    print("Install the Google Cloud SDK and authenticate, then re-run setup.")
    sys.exit(1)


def download_tokenizer() -> None:
    if os.path.exists(TOKENIZER_SENTINEL):
        print(f"--- Tokenizer already present at {TOKENIZER_SENTINEL} ---")
        return

    ensure_gsutil()

    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    for filename in TOKENIZER_FILES:
        source = f"{TOKENIZER_ROOT}/{filename}"
        dest = os.path.join(TOKENIZER_DIR, filename)
        print(f"--- Downloading tokenizer: {source} ---")
        run(["gsutil", "cp", source, dest])

    if not os.path.exists(TOKENIZER_SENTINEL):
        print(f"ERROR: Expected tokenizer file missing: {TOKENIZER_SENTINEL}")
        sys.exit(1)


def upsert_env_file(updates: dict[str, str]) -> None:
    env_path = ".env"
    existing_lines: list[str] = []
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            existing_lines = f.readlines()

    new_lines: list[str] = []
    seen_keys: set[str] = set()

    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            new_lines.append(line)
            continue

        key_part, _ = line.split("=", 1)
        key_part = key_part.strip()
        prefix = ""
        if key_part.startswith("export "):
            prefix = "export "
            key = key_part[len("export ") :].strip()
        else:
            key = key_part

        if key in updates:
            new_lines.append(f"{prefix}{key}={updates[key]}\n")
            seen_keys.add(key)
        else:
            new_lines.append(line)

    for key, value in updates.items():
        if key not in seen_keys:
            new_lines.append(f"{key}={value}\n")

    with open(env_path, "w") as f:
        f.writelines(new_lines)


def create_env_file():
    key_path = os.path.join(os.getcwd(), "ops", "gemma-tpu-writer-key.json")
    if os.path.exists(key_path):
        print(f"--- Configuring .env with key: {key_path} ---")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        upsert_env_file({"GOOGLE_APPLICATION_CREDENTIALS": key_path})


def ensure_wandb_api_key() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True

    try:
        from google.cloud import secretmanager
    except Exception:
        print(
            "google-cloud-secret-manager is not available yet; will retry after dependencies are installed."
        )
        return False

    secret_path = (
        f"projects/{WANDB_SECRET_PROJECT}/secrets/{WANDB_SECRET_NAME}/versions/latest"
    )
    try:
        client = secretmanager.SecretManagerServiceClient()
        response = client.access_secret_version(request={"name": secret_path})
        api_key = response.payload.data.decode("utf-8").strip()
        if not api_key:
            raise RuntimeError("Secret payload is empty")

        os.environ["WANDB_API_KEY"] = api_key
        upsert_env_file({"WANDB_API_KEY": api_key})
        print(
            f"--- Loaded WANDB_API_KEY from Secret Manager ({WANDB_SECRET_PROJECT}/{WANDB_SECRET_NAME}) ---"
        )
        return True
    except Exception as exc:
        print(f"Failed to load WANDB_API_KEY from Secret Manager: {exc}")
        print("If this is the first time, create the secret and grant access:")
        print("  gcloud secrets create wandb-api-key --data-file=-")
        print("  # paste WANDB key, then Ctrl-D")
        print("  gcloud secrets add-iam-policy-binding wandb-api-key \\")
        print(
            "    --member=serviceAccount:gemma-tpu-writer@default-482802.iam.gserviceaccount.com \\"
        )
        print("    --role=roles/secretmanager.secretAccessor \\")
        print("    --project=default-482802")
        return False


def ensure_system_deps() -> None:
    """Setup runs under system Python, so install deps needed to fetch W&B credentials."""
    missing: list[str] = []
    try:
        from google.cloud import secretmanager  # noqa: F401
    except Exception:
        missing.append("google-cloud-secret-manager")
    try:
        import wandb  # noqa: F401
    except Exception:
        missing.append("wandb")

    if not missing:
        return

    print(f"--- Installing system deps: {', '.join(missing)} ---")
    run([sys.executable, "-m", "pip", "install", "--user", *missing])


def enable_hugepages() -> None:
    print("--- Enabling Transparent Hugepages ---")
    try:
        # This requires sudo and is specific to Linux environments like TPU VMs.
        cmd = "sudo sh -c 'echo always > /sys/kernel/mm/transparent_hugepage/enabled'"
        subprocess.run(cmd, shell=True, check=False)
    except Exception as e:
        print(f"Warning: Could not enable hugepages: {e}")


def main():
    parser = argparse.ArgumentParser(description="Setup JAX environment")
    parser.add_argument(
        "--run-main", action="store_true", help="Launch main.py after setup"
    )
    args, unknown_args = parser.parse_known_args()

    # Setup runs outside the venv, so ensure W&B/Secret Manager deps exist first.
    ensure_system_deps()
    enable_hugepages()
    create_env_file()
    wandb_ready = ensure_wandb_api_key()

    setup_git()
    install_uv()
    sync_dependencies()
    if not wandb_ready and not os.environ.get("WANDB_API_KEY"):
        ensure_wandb_api_key()
    download_tokenizer()

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

        cmd = ["~/.local/bin/uv", "run", "python", "-m", "main"] + unknown_args

        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env)
    else:
        print("--- Setup Complete ---")
        print("To run training: ~/.local/bin/uv run python -m main")


if __name__ == "__main__":
    main()
