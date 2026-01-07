#!/usr/bin/env python3
"""
Zero-dependency setup script for JAX Gemma 1B.
Runs on both Dev (Compute Engine) and Prod (Vertex AI) environments.

Responsibilities:
1. Configure Git (for dev/debug).
2. Install `uv`.
3. Sync dependencies (`uv sync`).
4. Download model weights from GCS.
5. Launch `main.py`.
"""

import os
import subprocess
import sys
import shutil
import argparse

# --- Configuration ---
GIT_USER_NAME = "Dominik Farr"
GIT_USER_EMAIL = "dominik.farr@icloud.com"
WEIGHTS_BUCKET = "gemma-tpu-weights-us-west4-482802"
WEIGHTS_FILE = "model_stacked_pt.safetensors"
WEIGHTS_LOCAL_PATH = "data/model_stacked_pt.safetensors"

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
        # Use curl to get metadata
        result = subprocess.run(
            ["curl", "-s", "-H", "Metadata-Flavor: Google", f"http://metadata.google.internal/computeMetadata/v1/instance/{attribute}"],
            capture_output=True,
            text=True,
            timeout=1
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None

def sync_dependencies():
    print("--- Syncing dependencies with uv ---")
    
    cmd = ["uv", "sync", "--frozen", "--no-dev"]
    
    # Hardware Detection Logic
    is_gpu = shutil.which("nvidia-smi") is not None
    is_tpu = False
    
    # 1. Check Metadata Server (Definitive for GCP TPUs)
    accel_type = get_metadata("attributes/accelerator-type")
    
    if accel_type and ("tpu" in accel_type.lower() or "litepod" in accel_type.lower()):
        print(f"TPU detected via metadata: {accel_type}")
        is_tpu = True
    elif shutil.which("nvidia-smi"):
        # 2. Check local GPU
        print("GPU detected via nvidia-smi")
        is_gpu = True
        
    # Check 3: Env Var (Legacy/Vertex)
    if not is_tpu and not is_gpu and os.environ.get("TPU_NAME"):
        is_tpu = True
        print("TPU detected via TPU_NAME env var")
    
    # Apply extras
    if is_tpu:
        print("Enabling 'tpu' extra")
        cmd.extend(["--extra", "tpu"])
    elif is_gpu:
        print("Enabling 'cuda' extra")
        cmd.extend(["--extra", "cuda"])
    else:
        print("No accelerator detected: CPU only")
        
    run(cmd)

def download_weights():
    if os.path.exists(WEIGHTS_LOCAL_PATH):
        print(f"--- Weights already exist at {WEIGHTS_LOCAL_PATH} ---")
        return

    print(f"--- Downloading weights from gs://{WEIGHTS_BUCKET} ---")
    
    # Python logic to find and download
    download_script = f"""
import os
from google.cloud import storage

client = storage.Client()
bucket = client.bucket('{WEIGHTS_BUCKET}')

# Try exact match first
blob = bucket.blob('{WEIGHTS_FILE}')
if not blob.exists():
    print(f"'{WEIGHTS_FILE}' not found, searching for latest match...")
    blobs = list(client.list_blobs('{WEIGHTS_BUCKET}', prefix='model_stacked_pt'))
    # Sort by time updated
    blobs.sort(key=lambda x: x.updated, reverse=True)
    if not blobs:
        print("No weights found in bucket!")
        exit(1)
    blob = blobs[0]
    print(f"Found latest: {{blob.name}}")

print(f"Downloading {{blob.name}} to {WEIGHTS_LOCAL_PATH}...")
blob.download_to_filename('{WEIGHTS_LOCAL_PATH}')
"""
    run(["uv", "run", "python", "-c", download_script])

def main():
    parser = argparse.ArgumentParser(description="Setup JAX environment")
    parser.add_argument("--run-main", action="store_true", help="Launch main.py after setup")
    # Capture unknown args to pass to main.py if needed
    args, unknown_args = parser.parse_known_args()

    setup_git()
    install_uv()
    sync_dependencies()
    download_weights()
    
    if args.run_main:
        print("--- Launching main.py ---")
        env = os.environ.copy()
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        
        # Disable accelerator plugins if not present to save memory and avoid errors
        # Re-check hardware using the same logic (or reuse detection if we refactored)
        # For safety, check simple signals again
        is_gpu = shutil.which("nvidia-smi") is not None
        is_tpu = get_metadata("attributes/accelerator-type") or os.environ.get("TPU_NAME") or os.path.exists("/dev/accel0")
        
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