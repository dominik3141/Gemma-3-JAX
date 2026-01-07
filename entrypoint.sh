#!/usr/bin/env bash
set -euo pipefail

# Everything lives in /app: pull repo, sync deps, run training.
REPO_URL="https://github.com/dominik3141/Gemma-3-JAX.git"
BRANCH="main"
WORKDIR="/app"

# If running in Google's pre-built container with local-package-path, code is mounted/copied.
# We might need to handle WORKDIR differently or just assume we are in the right place.
# Google mounts it at a specific path, but usually sets PWD.
# Let's ensure uv is installed (for pre-built containers)
if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    pip install uv
fi

if [[ -d "${WORKDIR}/.git" ]]; then
    git -C "${WORKDIR}" fetch --prune
    git -C "${WORKDIR}" checkout "${BRANCH}"
    git -C "${WORKDIR}" pull --ff-only origin "${BRANCH}"
else
    # Only clone if we are NOT in the pre-built flow (which likely doesn't have .git but has files)
    # Actually, for local-package-path, we don't need to clone.
    # We can check if main.py exists.
    if [[ ! -f "supervised_train.py" ]]; then
         git clone --depth=1 --branch "${BRANCH}" "${REPO_URL}" "${WORKDIR}"
         cd "${WORKDIR}"
    fi
fi

# Ensure we are in the directory with uv.lock
if [[ -f "uv.lock" ]]; then
    uv sync --frozen --no-dev
else
    uv sync --no-dev
fi

# Detect hardware and install appropriate JAX
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected, installing jax[cuda12_pip]"
    uv pip install "jax[cuda12_pip]" -C "jaxlib:--allow_prestable"
elif [[ -n "${TPU_NAME:-}" ]] || [[ -c /dev/accel0 ]]; then
    echo "TPU detected, ensuring jax[tpu] is installed"
    uv pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
else
    echo "No accelerator detected, using default JAX"
fi

exec python -m supervised_train
