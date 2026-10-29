# syntax=docker/dockerfile:1
# Solia ai-service — FastAPI + PyTorch, serving solia_ai.api:app on port 8000.
# Build context = repo root (shared by every infra/docker/*.Dockerfile):
#   docker build -f infra/docker/ai-service.Dockerfile -t solia-ai-service .
# TORCH_VARIANT picks the torch build: `cpu` (default: CI, any machine) or `cu130` (NVIDIA GPU, see
# docker-compose.gpu.yml). Training uses CUDA when torch sees a GPU and falls back to the CPU otherwise.

# ---- build: locked deps into a venv (cached on the locks only), then the package ----
FROM python:3.11-slim AS build
ARG TORCH_VARIANT=cpu
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /src

# Dependency layer: only the locks are copied, so neither a .py edit nor a dev/tooling change in
# pyproject.toml reinstalls torch. A lock pins every runtime package exactly (transitives included;
# tests/test_requirements_lock.py keeps both in sync with pyproject.toml and each other), torch as
# its +cpu (requirements.lock) or +cu130 (requirements-cu130.lock) build, which only that PyTorch
# index serves: the CPU image can't pull CUDA wheels in. --no-deps installs exactly the lock, nothing
# re-resolved; `pip check` proves it is complete.
# The pip cache mount keeps downloaded wheels across builds without landing in a layer.
COPY services/ai-service/requirements*.lock ./
RUN --mount=type=cache,target=/root/.cache/pip \
    case "$TORCH_VARIANT" in \
        cpu) lock=requirements.lock ;; \
        cu130) lock=requirements-cu130.lock ;; \
        *) echo "unknown TORCH_VARIANT '$TORCH_VARIANT' (cpu | cu130)" >&2; exit 1 ;; \
    esac \
 && pip install --no-deps --extra-index-url "https://download.pytorch.org/whl/$TORCH_VARIANT" \
        -r "$lock" \
 && pip check

# The package itself (with its config.yaml), without re-resolving dependencies.
COPY services/ai-service/pyproject.toml .
COPY services/ai-service/solia_ai/ solia_ai/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps . \
 && python -c "import importlib.resources as r; assert r.files('solia_ai').joinpath('config.yaml').is_file()"

# ---- runtime: the venv only (no sources, tests, build tools or pip cache), non-root ----
FROM python:3.11-slim AS runtime
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models

# Model registry checkpoints live in /app/models; compose mounts the `models` named volume there
# (no VOLUME here: it would leave an anonymous volume behind every bare `docker run`). A fresh
# named volume inherits this directory's ownership; a bind mount or a volume first created by a
# root container doesn't, and the service logs a warning with the chown fix at startup.
RUN useradd --system --uid 10001 --home-dir /app --shell /usr/sbin/nologin solia \
 && mkdir -p /app/models \
 && chown -R solia:solia /app
COPY --from=build /opt/venv /opt/venv

WORKDIR /app
USER solia
EXPOSE 8000

# No curl in the image: Python stdlib probes /health, which never touches the database or the
# champion model, so the container is healthy before any model is promoted. During the start
# period it probes every 2s, so dependants don't wait a full interval for uvicorn's few seconds.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --start-interval=2s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["uvicorn", "solia_ai.api:app", "--host", "0.0.0.0", "--port", "8000"]
