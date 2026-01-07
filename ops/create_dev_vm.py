#!/usr/bin/env python3
"""
Provision a Development VM (GCE) with T4 GPU.
Uses Deep Learning VM Image for pre-installed Drivers/CUDA.
"""

import argparse
import subprocess
import sys
import time

# Configuration
DEFAULT_VM_NAME = "gemma-dev-cpu"
DEFAULT_ZONE = "us-west4-a"
MACHINE_TYPE = "n1-highmem-8"  # More memory to avoid OOM
# CPU Only Configuration
IMAGE_PROJECT = "ubuntu-os-cloud"
IMAGE_FAMILY = "ubuntu-2204-lts"
SERVICE_ACCOUNT = "gemma-tpu-writer@default-482802.iam.gserviceaccount.com"


def run_checked(cmd):
    print("Running: ", " ".join(cmd))
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(f"Command failed ({res.returncode}):\n{res.stderr}\n")
        # Don't exit if it fails because it might already exist
        if "already exists" in res.stderr:
            print("VM already exists, proceeding...")
            return
        sys.exit(res.returncode)
    return res.stdout.strip()


def create_vm(vm_name, zone):
    print(f"--- Creating VM {vm_name} in {zone} ---")

    cmd = [
        "gcloud",
        "compute",
        "instances",
        "create",
        vm_name,
        f"--zone={zone}",
        f"--machine-type={MACHINE_TYPE}",
        f"--image-family={IMAGE_FAMILY}",
        f"--image-project={IMAGE_PROJECT}",
        f"--service-account={SERVICE_ACCOUNT}",
        "--scopes=https://www.googleapis.com/auth/cloud-platform",
        "--boot-disk-size=100GB",
        "--boot-disk-type=pd-ssd",
    ]

    run_checked(cmd)

    print("--- Waiting for VM to be ready ---")
    time.sleep(10)
    print(f"VM {vm_name} created.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=DEFAULT_VM_NAME)
    parser.add_argument("--zone", default=DEFAULT_ZONE)
    args = parser.parse_args()

    create_vm(args.name, args.zone)

    print("\n--- Setup Complete ---")
    print(
        f"1. Sync code:  python3 ops/remote_sync.py --vm {args.name} --zone {args.zone}"
    )
    print(f"2. Connect:    gcloud compute ssh {args.name} --zone {args.zone}")
    print("3. Init env:   cd ~/app && python3 setup.py")


if __name__ == "__main__":
    main()
