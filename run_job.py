#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time

DEFAULT_REGION = "us-west4"
DEFAULT_NUM_BATCHES = 100
DEFAULT_TIMEOUT_HOURS = 1  # server-side timeout
DEFAULT_ACCEL_TYPE = "tpu-v5e-podslice"
DEFAULT_ACCEL_COUNT = 4
DEFAULT_IMAGE = "us-west4-docker.pkg.dev/default-482802/gemma-tpu/jax-gemma-tpu:latest"
DEFAULT_SA = "gemma-tpu-writer@default-482802.iam.gserviceaccount.com"
DEFAULT_LOCAL_CREDS = "gemma-tpu-writer-key.json"


def run_checked(cmd: list[str]) -> str:
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(f"Command failed ({res.returncode}): {' '.join(cmd)}\n{res.stderr}\n")
        sys.exit(res.returncode)
    return res.stdout.strip()


def create_vertex_job(args) -> str:
    timeout_seconds = args.timeout_hours * 3600
    spec = (
        f"machine-type=cloud-tpu,"
        f"accelerator-type={args.accelerator_type},"
        f"accelerator-count={args.accelerator_count},"
        f"replica-count=1,"
        f"container-image-uri={args.image},"
        f"env=NUM_BATCHES={args.num_batches}"
    )
    display_name = f"gemma-sft-{int(time.time())}"
    cmd = [
        "gcloud",
        "ai",
        "custom-jobs",
        "create",
        f"--region={args.region}",
        f"--display-name={display_name}",
        f"--service-account={args.service_account}",
        f"--worker-pool-spec={spec}",
        f"--scheduling-timeout={timeout_seconds}s",
        "--format=value(name)",
    ]
    return run_checked(cmd)


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
    parser.add_argument("--timeout-hours", type=int, default=DEFAULT_TIMEOUT_HOURS, help="Server-side timeout (hours) for Vertex jobs.")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--accelerator-type", default=DEFAULT_ACCEL_TYPE)
    parser.add_argument("--accelerator-count", type=int, default=DEFAULT_ACCEL_COUNT)
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
