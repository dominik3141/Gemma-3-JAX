#!/usr/bin/env python3
"""
Dev Workflow Sync Script.
Syncs current directory to a remote GCE VM (TPU or GPU) for interactive debugging.
Always tunnels through IAP (no external IPs required).
"""

import argparse
import os
import shlex
import shutil
import subprocess
import time

DEFAULT_SYNC_GCS_PREFIX = "gs://gemma-dev-sync-482802-euw4/dev-sync"


def run(cmd):
    print(f"Running: {shlex.join(cmd)}")
    subprocess.run(cmd, check=True)


def _normalize_gcs_prefix(gcs_prefix: str) -> str:
    if not gcs_prefix.startswith("gs://"):
        raise ValueError(f"Expected gs:// prefix, got: {gcs_prefix}")
    return gcs_prefix.rstrip("/")


def _upload_artifact(local_tar: str, gcs_prefix: str) -> str:
    artifact_name = f"dev_sync_{int(time.time())}.tar.gz"
    artifact_uri = f"{_normalize_gcs_prefix(gcs_prefix)}/{artifact_name}"

    if shutil.which("gcloud"):
        run(["gcloud", "storage", "cp", local_tar, artifact_uri])
    elif shutil.which("gsutil"):
        run(["gsutil", "cp", local_tar, artifact_uri])
    else:
        raise RuntimeError("Neither gcloud nor gsutil found on PATH for GCS upload.")

    return artifact_uri


def _delete_artifact(artifact_uri: str) -> None:
    if shutil.which("gcloud"):
        run(["gcloud", "storage", "rm", artifact_uri])
    elif shutil.which("gsutil"):
        run(["gsutil", "rm", artifact_uri])
    else:
        print(
            f"WARNING: Could not delete {artifact_uri} because neither gcloud nor gsutil is on PATH."
        )


def _download_and_extract_command(artifact_uri: str) -> str:
    quoted_uri = shlex.quote(artifact_uri)
    return (
        "set -eu; "
        "rm -rf ~/app && mkdir -p ~/app; "
        "if command -v gsutil >/dev/null 2>&1; then "
        f"gsutil cp {quoted_uri} ~/dev_sync.tar.gz; "
        "elif command -v gcloud >/dev/null 2>&1; then "
        f"gcloud storage cp {quoted_uri} ~/dev_sync.tar.gz; "
        "else "
        "echo 'Missing remote dependency: neither gsutil nor gcloud is installed on this VM.' >&2; "
        "exit 1; "
        "fi; "
        "tar --warning=no-unknown-keyword -xzf ~/dev_sync.tar.gz -C ~/app; "
        "rm ~/dev_sync.tar.gz"
    )


def _run_remote_command(vm_name: str, zone: str, is_tpu: bool, command: str) -> None:
    if is_tpu:
        run(
            [
                "gcloud",
                "alpha",
                "compute",
                "tpus",
                "tpu-vm",
                "ssh",
                vm_name,
                f"--zone={zone}",
                "--worker=all",
                "--tunnel-through-iap",
                "--command",
                command,
            ]
        )
    else:
        run(
            [
                "gcloud",
                "compute",
                "ssh",
                vm_name,
                f"--zone={zone}",
                "--tunnel-through-iap",
                "--command",
                command,
            ]
        )


def sync_code(vm_name, zone, is_tpu):
    print(f"--- Syncing code to {vm_name} ({zone}) ---")

    # Exclude heavy/unnecessary files
    excludes = [
        ".venv",
        "__pycache__",
        "*.safetensors",  # Don't re-upload weights if they exist locally
        "*.pt",
        "*.ckpt",
        "source_*.tar.gz",
        "dev_sync.tar.gz",
        "data",  # Don't sync data directory
        "jax-trace",  # JAX trace files (2.7GB!)
        "HLO_dumps",  # HLO dumps
        ".ruff_cache",  # Ruff cache
        ".DS_Store",  # macOS metadata
    ]

    # Build one local tar artifact with excludes and fan it out via GCS.
    # We set COPYFILE_DISABLE=1 to prevent macOS from including ._ metadata files (AppleDouble)
    env = os.environ.copy()
    env["COPYFILE_DISABLE"] = "1"

    subprocess.run(
        ["tar", "-czf", "dev_sync.tar.gz"]
        + [f"--exclude={e}" for e in excludes]
        + ["."],
        env=env,
        check=True,
    )

    artifact_uri = _upload_artifact("dev_sync.tar.gz", DEFAULT_SYNC_GCS_PREFIX)
    print(f"--- Uploaded sync artifact to {artifact_uri} ---")

    try:
        download_cmd = _download_and_extract_command(artifact_uri)
        print("--- Distributing archive to TPU workers (this can take 1-2 minutes) ---")
        _run_remote_command(vm_name, zone, is_tpu, download_cmd)
    finally:
        try:
            _delete_artifact(artifact_uri)
            print(f"--- Deleted sync artifact {artifact_uri} ---")
        except Exception as exc:
            print(f"WARNING: Failed to delete sync artifact {artifact_uri}: {exc}")
        subprocess.run(["rm", "-f", "dev_sync.tar.gz"], check=False)

    print("--- Running setup.py on remote ---")
    _run_remote_command(vm_name, zone, is_tpu, "cd ~/app && python setup.py")

    print("--- Sync + setup complete ---")
    if is_tpu:
        print(
            f"To connect: gcloud alpha compute tpus tpu-vm ssh {vm_name} --zone={zone} --tunnel-through-iap"
        )
    else:
        print(
            f"To connect: gcloud compute ssh {vm_name} --zone={zone} --tunnel-through-iap"
        )
    print("Run: cd ~/app && ~/.local/bin/uv run python -m main")


def main():
    parser = argparse.ArgumentParser(
        description="Sync code to a VM via IAP (no external IPs required)."
    )
    parser.add_argument("--vm", required=True, help="Name of the VM")
    parser.add_argument("--zone", required=True, help="Zone of the VM")
    parser.add_argument(
        "--tpu",
        action="store_true",
        help="Is this a TPU VM? (uses gcloud alpha + IAP)",
    )
    args = parser.parse_args()

    sync_code(args.vm, args.zone, args.tpu)


if __name__ == "__main__":
    main()
