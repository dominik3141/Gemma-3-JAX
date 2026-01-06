FROM python:3.14-slim

# Non-interactive apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Where the project lives inside the image
ENV APP_DIR=/app \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:/root/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1

WORKDIR ${APP_DIR}

RUN set -euo pipefail \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       curl wget git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (single binary)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Dependency install first to maximize layer caching
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --compile-bytecode

# Copy the remainder of the repo
COPY . ${APP_DIR}

RUN chmod +x ${APP_DIR}/entrypoint.sh

# Default command can be overridden by Vertex worker pool spec
CMD ["bash", "/app/entrypoint.sh", "python", "-m", "supervised_train"]
