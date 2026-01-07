#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
import json
import tempfile
import os

DEFAULT_REGION = "us-west4"
BUCKET_NAME = "gemma-tpu-weights-us-west4-482802"

# GPU Config
DEFAULT_GPU_TYPE = "NVIDIA_TESLA_T4"
DEFAULT_GPU_MACHINE_TYPE = "n1-standard-4"
# Using a generic Python image, setup.py handles the rest
DEFAULT_GPU_IMAGE = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest" 

# TPU Config
DEFAULT_TPU_TYPE = "TPU_V5_LITEPOD"
DEFAULT_TPU_MACHINE_TYPE = "ct5lp-hightpu-1t"
# We can use a base python image for TPU too, but we need one with python installed.
# The pytorch image above works for python. Or we can use a python:3.10 base.
# Let's stick to the pytorch one as it has useful system libs.
DEFAULT_TPU_IMAGE = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest" 


def run_checked(cmd: list[str]) -> str:
    print(f"Running: {" ".join(cmd)}")
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(f"Command failed ({res.returncode}):\n{res.stderr}\n")
        sys.exit(res.returncode)
    return res.stdout.strip()

def create_tarball():
    timestamp = int(time.time())
    tar_filename = f"source_{timestamp}.tar.gz"
    blob_name = f"source/{tar_filename}"
    
    print(f"Creating tarball {tar_filename}...")
    subprocess.run(
        [
            "tar",
            "--exclude=.git",
            "--exclude=.venv",
            "--exclude=__pycache__",
            "--exclude=*.tar.gz",
            "--exclude=*.safetensors",
            "--exclude=*.pt",
            "--exclude=*.ckpt",
            "-czf",
            tar_filename,
            ".",
        ],
        check=True,
    )
    
    print(f"Uploading to gs://{BUCKET_NAME}/{blob_name}...")
    run_checked(["gcloud", "storage", "cp", tar_filename, f"gs://{BUCKET_NAME}/{blob_name}"])
    os.remove(tar_filename)
    
    return tar_filename

def create_vertex_job(args, tar_filename) -> str:
    display_name = f"gemma-train-{int(time.time())}"
    
    # Command to run on the remote machine:
    # 1. Install deps to download source
    # 2. Download source tarball
    # 3. Untar
    # 4. Run setup.py (which installs uv, syncs deps, downloads weights, runs main.py)
    
    # We use a simple python script embedded in the command to download the source
    download_source_cmd = (
        f"pip install google-cloud-storage && "
        f"python3 -c 'from google.cloud import storage; "
        f"client=storage.Client(); "
        f"bucket=client.bucket(\"{BUCKET_NAME}\"); "
        f"blob=bucket.blob(\"source/{tar_filename}\"); "
        f"blob.download_to_filename(\"source.tar.gz\")' && "
        f"tar -xf source.tar.gz && "
        f"python3 setup.py"
    )

    machine_spec = {
        "machineType": getattr(args, "machine_type", DEFAULT_TPU_MACHINE_TYPE),
    }
    
    container_image = DEFAULT_TPU_IMAGE

    if args.use_gpu:
        machine_spec["machineType"] = getattr(args, "machine_type", DEFAULT_GPU_MACHINE_TYPE)
        machine_spec["acceleratorType"] = DEFAULT_GPU_TYPE
        machine_spec["acceleratorCount"] = 1
        container_image = DEFAULT_GPU_IMAGE
    else:
        # TPU
        machine_spec["tpuTopology"] = "1x1" 

    config = {
        "workerPoolSpecs": [
            {
                "machineSpec": machine_spec,
                "replicaCount": 1,
                "containerSpec": {
                    "imageUri": container_image,
                    "command": ["/bin/bash", "-c", download_source_cmd],
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

def main():
    parser = argparse.ArgumentParser(description="Deploy to Vertex AI (TPU or GPU)")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU instead of TPU")
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--region", default=DEFAULT_REGION)
    # Allow overriding machine type
    parser.add_argument("--machine-type", default=None) 
    
    # Defaults
    parser.add_argument("--service-account", default="gemma-tpu-writer@default-482802.iam.gserviceaccount.com")
    
    args = parser.parse_args()
    
    # Set default machine type if not provided
    if args.machine_type is None:
        args.machine_type = DEFAULT_GPU_MACHINE_TYPE if args.use_gpu else DEFAULT_TPU_MACHINE_TYPE

    tar_file = create_tarball()
    job_name = create_vertex_job(args, tar_file)
    print(f"Job created: {job_name}")
    print(f"Stream logs with: gcloud ai custom-jobs stream-logs {job_name} --region={args.region}")

if __name__ == "__main__":
    main()
