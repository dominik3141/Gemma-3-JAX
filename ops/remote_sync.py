#!/usr/bin/env python3
"""
Dev Workflow Sync Script.
Syncs current directory to a remote GCE VM (TPU or GPU) for interactive debugging.
Always tunnels through IAP (no external IPs required).
"""

import argparse
import subprocess
import os


def run(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


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

    # Simple fallback: Tar locally with excludes, SCP tar, Untar remotely.
    # We set COPYFILE_DISABLE=1 to prevent macOS from including ._ metadata files (AppleDouble)
    env = os.environ.copy()
    env["COPYFILE_DISABLE"] = "1"

    subprocess.run(
        ["tar", "-czf", "dev_sync.tar.gz"]
        + [f"--exclude={e}" for e in excludes]
        + ["."],
        env=env,
    )

    if is_tpu:
        # Full Clean: Nuke ~/app and recreate
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
                "rm -rf ~/app && mkdir -p ~/app",
            ]
        )
        # TPU VM SCP
        run(
            [
                "gcloud",
                "alpha",
                "compute",
                "tpus",
                "tpu-vm",
                "scp",
                "dev_sync.tar.gz",
                "--worker=all",
                f"{vm_name}:~/dev_sync.tar.gz",
                f"--zone={zone}",
                "--tunnel-through-iap",
            ]
        )
        # Untar
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
                "tar --warning=no-unknown-keyword -xzf ~/dev_sync.tar.gz -C ~/app && rm ~/dev_sync.tar.gz",
            ]
        )
    else:
        # Full Clean: Nuke ~/app and recreate
        run(
            [
                "gcloud",
                "compute",
                "ssh",
                vm_name,
                f"--zone={zone}",
                "--tunnel-through-iap",
                "--command",
                "rm -rf ~/app && mkdir -p ~/app",
            ]
        )
        # Standard VM SCP
        run(
            [
                "gcloud",
                "compute",
                "scp",
                "dev_sync.tar.gz",
                f"{vm_name}:~/dev_sync.tar.gz",
                f"--zone={zone}",
                "--tunnel-through-iap",
            ]
        )
        run(
            [
                "gcloud",
                "compute",
                "ssh",
                vm_name,
                f"--zone={zone}",
                "--tunnel-through-iap",
                "--command",
                "tar --warning=no-unknown-keyword -xzf ~/dev_sync.tar.gz -C ~/app && rm ~/dev_sync.tar.gz",
            ]
        )

    subprocess.run(["rm", "dev_sync.tar.gz"])
    print("--- Sync Complete ---")
    if is_tpu:
        print(
            f"To connect: gcloud alpha compute tpus tpu-vm ssh {vm_name} --zone={zone} --tunnel-through-iap"
        )
    else:
        print(
            f"To connect: gcloud compute ssh {vm_name} --zone={zone} --tunnel-through-iap"
        )
    print("1. Setup:   cd ~/app && python3 setup.py")
    print("2. Run:     uv run python -m main")


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
