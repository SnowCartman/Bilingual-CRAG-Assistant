# syntax=docker/dockerfile:1.7
#
# Backend image for the Bilingual CRAG Assistant.
# Built from the repository root as build-context.
#
#   docker build -f backend.Dockerfile -t bilingual-crag-backend .
#
# Runtime port is configurable via $PORT (default 8000) so the same image
# works under local Compose AND any container platform that injects $PORT.

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps needed by fastembed (onnxruntime expects libgomp + libstdc++).
# Everything else (numpy, requests, etc.) ships pure-Python or as manylinux wheels.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-backend.txt ./requirements.txt
RUN pip install -r requirements.txt

# Copy only the runtime code. The config module computes
#   REPO_ROOT = parents[2]
# from /app/app/config.py -> /app -> / -- so the .env it looks for is /.env,
# which doesn't exist in the container. That's intentional: secrets come via
# environment variables injected by Compose, not via a baked-in .env file.
COPY app ./app

ENV PORT=8000
EXPOSE 8000

# /bin/sh -c so $PORT expansion works at container start, not at build time.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
