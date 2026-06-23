# =============================================================================
# Power Demand Forecasting - production image
#
# Single-stage, slim Python image that serves the FastAPI backend (which also
# mounts the static dashboard) on port 8000. The trained model artifact
# (models/demand_model.pkl) is copied in with the source, so the API and
# dashboard work IMMEDIATELY with no retraining step at container start.
#
# Build:  docker build -t power-demand-forecasting .
# Run:    docker run --rm -p 8000:8000 power-demand-forecasting
# Open:   http://localhost:8000
# =============================================================================
FROM python:3.11-slim

# --- Runtime behaviour -------------------------------------------------------
#   PYTHONUNBUFFERED       -> logs are flushed immediately (good for `docker logs`)
#   PYTHONDONTWRITEBYTECODE-> no .pyc clutter in the image
#   PIP_NO_CACHE_DIR       -> smaller image (no pip wheel cache)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- System dependencies -----------------------------------------------------
# LightGBM links against libgomp (OpenMP) at RUNTIME. The slim base image does
# not ship it, so we must install libgomp1 or `import lightgbm` will fail with
# "libgomp.so.1: cannot open shared object file". No compiler/toolchain is
# needed: every wheel in requirements.txt (lightgbm, numpy, pandas, ...) ships
# prebuilt manylinux binaries, so we deliberately keep the image lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies (cached layer) --------------------------------------
# Copy ONLY requirements first so this layer is cached and not rebuilt every
# time application source changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- Application source -------------------------------------------------------
# Copies src/, backend/, frontend/, data/ and the committed models/ artifacts.
# .dockerignore keeps caches, notebooks checkpoints, venvs, etc. out.
COPY . .

# Document the port the app listens on.
EXPOSE 8000

# --- Start the server --------------------------------------------------------
# backend.main:app mounts the dashboard at / and the JSON API under /api.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
