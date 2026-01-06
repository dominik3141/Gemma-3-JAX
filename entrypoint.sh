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

exec python -m supervised_train
