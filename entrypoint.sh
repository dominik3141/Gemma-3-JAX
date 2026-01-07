#!/usr/bin/env bash
set -euo pipefail

# Everything lives in /app: pull repo, sync deps, run training.
REPO_URL="https://github.com/dominik3141/Gemma-3-JAX.git"
BRANCH="main"
WORKDIR="/app"

if [[ -d "${WORKDIR}/.git" ]]; then
    git -C "${WORKDIR}" fetch --prune
    git -C "${WORKDIR}" checkout "${BRANCH}"
    git -C "${WORKDIR}" pull --ff-only origin "${BRANCH}"
else
    git clone --depth=1 --branch "${BRANCH}" "${REPO_URL}" "${WORKDIR}"
fi

cd "${WORKDIR}"
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
