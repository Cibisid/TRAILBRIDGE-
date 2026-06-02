# TrialBridge — Backend Dockerfile
# Multi-stage build: development has hot-reload, production is lean and secure.

# -----------------------------------------------
# Stage 1: Base — shared between dev and prod
# -----------------------------------------------
FROM python:3.11-slim AS base

# Prevents Python from writing .pyc files (cleaner container)
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout/stderr (logs appear immediately)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed for ML libraries and PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Copy requirements first — Docker caches this layer
# Only rebuilds when requirements.txt changes, not on every code change
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# -----------------------------------------------
# Stage 2: Development — adds hot reload
# -----------------------------------------------
FROM base AS development

# Install development-only tools
COPY backend/requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .

EXPOSE 8000

# Development runs with uvicorn --reload
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# -----------------------------------------------
# Stage 3: Production — minimal, no dev tools
# -----------------------------------------------
FROM base AS production

# Create non-root user for security
# Never run production containers as root
RUN groupadd -r trialbridge && useradd -r -g trialbridge trialbridge

COPY . .

# Set correct ownership
RUN chown -R trialbridge:trialbridge /app

USER trialbridge

EXPOSE 8000

# Production uses gunicorn with uvicorn workers for true parallelism
# 4 workers = handles 4 concurrent requests simultaneously
CMD ["gunicorn", "backend.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
