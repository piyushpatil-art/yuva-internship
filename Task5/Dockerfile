# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Student Performance Prediction API
# Base image: Python 3.11 slim for minimal attack surface and fast builds
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata labels
LABEL maintainer="Yuva Internship AI Trainee"
LABEL description="Student Performance Prediction API (XGBoost + FastAPI)"
LABEL version="1.0.0"

# ── Environment variables ─────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app \
    MODEL_PATH=/app/model/xgb_model.joblib \
    SCALER_PATH=/app/model/scaler.joblib \
    META_PATH=/app/model/metadata.json

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR $APP_HOME

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy requirements first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ──────────────────────────────────────────────────────
COPY app/       ./app/
COPY model/     ./model/

# ── Create directories for logs and monitoring outputs ─────────────────────────
RUN mkdir -p logs monitoring

# ── Train model at build time (generates model artefacts) ─────────────────────
# In production, artefacts would be mounted from a model registry (e.g. MLflow/S3)
RUN python model/train_model.py

# ── Non-root user for security ─────────────────────────────────────────────────
RUN groupadd --gid 1001 appgroup && \
    useradd  --uid 1001 --gid appgroup --no-create-home appuser && \
    chown -R appuser:appgroup $APP_HOME

USER appuser

# ── Expose API port ────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ───────────────────────────────────────────────────────────────
# Docker will mark the container unhealthy if /health returns non-200
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start command ──────────────────────────────────────────────────────────────
# --workers 2: 2 Uvicorn worker processes for concurrency
# --timeout-keep-alive 30: keep-alive timeout
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info", \
     "--timeout-keep-alive", "30"]
