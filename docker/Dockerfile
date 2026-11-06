# syntax=docker/dockerfile:1.7
# Multi-stage image (§ARCH 3.2): base → deps → model → runtime → dev.
# Only the `deps` and `model` stages touch the network, and only at build time.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

# ---------------------------------------------------------------------------- base
FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid 10001 --home-dir /home/app --create-home \
            --shell /usr/sbin/nologin app
WORKDIR /app

# ---------------------------------------------------------------------------- deps
FROM base AS deps
COPY requirements.lock /tmp/requirements.lock
# Hash-checked, no resolver: the lock is the complete closure (verified by `pip check`).
# torch's C++ headers and its own test suite are unused here (no extension builds): ~170 MB.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    PIP_NO_CACHE_DIR=0 pip install --require-hashes --no-deps -r /tmp/requirements.lock \
 && rm -rf /usr/local/lib/python3.12/site-packages/torch/include \
           /usr/local/lib/python3.12/site-packages/torch/test \
 && pip check

# ---------------------------------------------------------------------------- model
FROM deps AS model
ARG MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
ARG MODEL_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41
RUN HF_HOME=/tmp/hf python - <<EOF
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("${MODEL_NAME}", revision="${MODEL_REVISION}", device="cpu")
model.save("/models/all-MiniLM-L6-v2")
with open("/models/all-MiniLM-L6-v2/REVISION", "w", encoding="utf-8") as fh:
    fh.write("${MODEL_REVISION}\n")
EOF
# BUG-001: save() writes model.safetensors as 0600 root; the runtime user (10001) must read it.
RUN rm -rf /tmp/hf && chmod -R a+rX,go-w /models

# ---------------------------------------------------------------------------- runtime
FROM deps AS runtime
ARG MODEL_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41
COPY --from=model /models /models
COPY pyproject.toml dataset.py ./
COPY src/ src/
RUN pip install --no-deps --no-build-isolation . \
 && rm -rf build src/*.egg-info \
 && mkdir -p /data/logs && chmod 1777 /data/logs
# (A fresh named volume mounted at /data/logs inherits this directory's mode.)
COPY kb/ kb/
COPY config/ config/
COPY scripts/ scripts/
COPY eval/ eval/
# Runtime environment (§ARCH 5). Egress-off flags are also defaulted in credagent/__init__.py.
ENV MOCK_LLM=true \
    LLM_PROVIDER=mock \
    CREWAI_DISABLE_TELEMETRY=true \
    OTEL_SDK_DISABLED=true \
    CREWAI_TRACING_ENABLED=false \
    ANONYMIZED_TELEMETRY=False \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    EMBED_MODEL_PATH=/models/all-MiniLM-L6-v2 \
    EMBED_MODEL_REVISION=${MODEL_REVISION} \
    CREWAI_STORAGE_DIR=/tmp/crewai \
    HOME=/home/app
USER 10001:10001
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]

# ---------------------------------------------------------------------------- dev
FROM runtime AS dev
USER root
COPY requirements-dev.lock /tmp/requirements-dev.lock
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    PIP_NO_CACHE_DIR=0 pip install --require-hashes --no-deps -r /tmp/requirements-dev.lock \
 && pip check
# The repo is bind-mounted read-only at /app in the `tests` service; this PYTHONPATH makes the
# mounted sources win over the copy installed in site-packages (edit → test without rebuild).
ENV PYTHONPATH=/app/src:/app
USER 10001:10001
