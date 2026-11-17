# syntax=docker/dockerfile:1
# Multi-stage production Dockerfile for Quant Research Platform

# ── Stage 1: Build stage ──────────────────────────────────────
FROM python:3.12-slim-bookworm AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements/constraints.txt /build/constraints.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /build/constraints.txt

# ── Stage 2: Runtime stage ────────────────────────────────────
FROM python:3.12-slim-bookworm AS runtime

LABEL maintainer="Quant AI Research Platform" \
      version="1.0.0" \
      description="Quantitative Research and Production Trading Platform"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    tzdata \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Kolkata \
    MPLCONFIGDIR=/tmp \
    JOBLIB_TEMP_FOLDER=/tmp

# Create non-root system user and group (uid 10001, gid 10001)
RUN groupadd -g 10001 quant && \
    useradd -u 10001 -g quant -m -d /home/quant -s /bin/bash quant

WORKDIR /app

# Copy installed venv from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application codebase
COPY --chown=quant:quant . /app

# Create required runtime state directories with correct permissions
RUN mkdir -p /app/data /app/models /app/logs /app/data/backtests /app/data/pipeline_runs && \
    chmod +x /app/docker/entrypoint.sh && \
    chown -R quant:quant /app/data /app/models /app/logs

USER 10001:10001

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["api"]
