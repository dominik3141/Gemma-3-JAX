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

# --- Configuration ---
GIT_USER_NAME = "Dominik Farr"
GIT_USER_EMAIL = "dominik.farr@icloud.com"
WEIGHTS_BUCKET = "gemma-tpu-weights-us-west4-482802"
WEIGHTS_FILE = "model_stacked_pt.safetensors"
WEIGHTS_LOCAL_PATH = "model_stacked_pt.safetensors"

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

def sync_dependencies():
    print("--- Syncing dependencies with uv ---")
    
    cmd = ["uv", "sync", "--frozen", "--no-dev"]
    
    # Determine hardware and add extras
    if shutil.which("nvidia-smi"):
        print("GPU detected: enabling 'cuda' extra")
        cmd.extend(["--extra", "cuda"])
    elif os.environ.get("TPU_NAME") or os.path.exists("/dev/accel0"):
        print("TPU detected: enabling 'tpu' extra")
        cmd.extend(["--extra", "tpu"])
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
    setup_git()
    install_uv()
    sync_dependencies()
    download_weights()
    
    print("--- Launching main.py ---")
    # Pass all arguments forwarded to this script
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    # Disable accelerator plugins if not present to save memory and avoid errors
    if not shutil.which("nvidia-smi") and not (os.environ.get("TPU_NAME") or os.path.exists("/dev/accel0")):
        print("Forcing JAX to CPU mode")
        env["JAX_PLATFORMS"] = "cpu"
    
    cmd = ["uv", "run", "python", "-m", "main"] + sys.argv[1:]
    
    # Use subprocess.run with env
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

if __name__ == "__main__":
    main()
