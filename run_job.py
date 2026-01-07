#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
import json
import tempfile
import os

DEFAULT_REGION = "us-west4"
DEFAULT_NUM_BATCHES = 100
DEFAULT_TIMEOUT_HOURS = 1
# Default to TPU v5e single chip.
DEFAULT_TPU_TYPE = "TPU_V5_LITEPOD"
DEFAULT_ACCELERATOR_COUNT = 1
DEFAULT_TPU_TOPOLOGY = "1x1"
DEFAULT_MACHINE_TYPE = "ct5lp-hightpu-1t"  # host VM type for TPU v5e single chip
DEFAULT_IMAGE = "us-west4-docker.pkg.dev/default-482802/gemma-tpu/jax-gemma-tpu:latest"
DEFAULT_SA = "gemma-tpu-writer@default-482802.iam.gserviceaccount.com"
DEFAULT_LOCAL_CREDS = "gemma-tpu-writer-key.json"
DEFAULT_GPU_TYPE = "NVIDIA_TESLA_T4"
DEFAULT_GPU_MACHINE_TYPE = "n1-standard-4"


def run_checked(cmd: list[str]) -> str:
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(f"Command failed ({res.returncode}): {' '.join(cmd)}\n{res.stderr}\n")
        sys.exit(res.returncode)
    return res.stdout.strip()


def create_vertex_job(args) -> str:
    machine_spec = {
        "machineType": getattr(args, "machine_type", DEFAULT_MACHINE_TYPE),
    }

    if args.use_gpu:
        machine_spec["machineType"] = DEFAULT_GPU_MACHINE_TYPE
        machine_spec["acceleratorType"] = DEFAULT_GPU_TYPE
        machine_spec["acceleratorCount"] = 1
    else:
        # TPU Configuration
        machine_spec["tpuTopology"] = getattr(args, "tpu_topology", DEFAULT_TPU_TOPOLOGY)

    config = {
        "workerPoolSpecs": [
            {
                "machineSpec": machine_spec,
                "replicaCount": 1,
                "containerSpec": {
                    "imageUri": args.image,
                    "env": [
                        {"name": "NUM_BATCHES", "value": str(args.num_batches)},
                    ],
                },
            }
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    try:
        display_name = f"gemma-sft-{int(time.time())}"
        cmd = [
            "gcloud",
            "ai",
            "custom-jobs",
            "create",
            f"--region={args.region}",
            f"--display-name={display_name}",
            f"--service-account={args.service_account}",
            f"--config={config_path}",
            "--format=value(name)",
        ]
        return run_checked(cmd)
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)



def stream_logs(args, job_name: str) -> None:
    cmd = [
        "gcloud",
        "ai",
        "custom-jobs",
        "stream-logs",
        job_name,
        f"--region={args.region}",
    ]
    subprocess.run(cmd, check=False)


def cancel_job(args, job_name: str) -> None:
    cmd = [
        "gcloud",
        "ai",
        "custom-jobs",
        "cancel",
        job_name,
        f"--region={args.region}",
    ]
    subprocess.run(cmd, check=False)


def run_local(args) -> None:
    # Always mount the local SA key for bucket writes.
    from pathlib import Path

    key_path = Path(DEFAULT_LOCAL_CREDS)
    if not key_path.exists():
        sys.stderr.write(
            f"Missing service account key at {key_path}. Place the key there before running --local.\n"
        )
        sys.exit(1)

    cmd = [
        "docker",
        "run",
        "--rm",
        "--platform=linux/amd64",
        "-v",
        f"{key_path}:/var/secrets/key.json:ro",
        "-e",
        f"NUM_BATCHES={args.num_batches}",
        "-e",
        "XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "-e",
        "GOOGLE_APPLICATION_CREDENTIALS=/var/secrets/key.json",
        DEFAULT_IMAGE,
    ]
    subprocess.run(cmd, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemma TPU training locally or on Vertex.")
    parser.add_argument("--num-batches", type=int, default=DEFAULT_NUM_BATCHES)
    parser.add_argument("--timeout-hours", type=int, default=DEFAULT_TIMEOUT_HOURS, help="Timeout hint (not enforced client-side).")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--machine-type", default=DEFAULT_MACHINE_TYPE)
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--accelerator-count", type=int, default=DEFAULT_ACCELERATOR_COUNT)
    parser.add_argument("--tpu-topology", default=DEFAULT_TPU_TOPOLOGY)
    parser.add_argument("--use-gpu", action="store_true", help="Use NVIDIA T4 GPU instead of TPU.")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--service-account", default=DEFAULT_SA)
    parser.add_argument("--local", action="store_true", help="Run locally via docker instead of Vertex.")
    args = parser.parse_args()

    if args.local:
        run_local(args)
        return

    job_name = create_vertex_job(args)
    print(f"Created job: {job_name}")
    print(f"Streaming logs (server-side timeout {args.timeout_hours}h)...")

    stream_logs(args, job_name)


if __name__ == "__main__":
    main()
