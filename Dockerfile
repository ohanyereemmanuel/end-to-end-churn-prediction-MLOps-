# Dockerfile
# Industry concept: Multi-stage builds keep the final image lean.
# Build stage installs everything; runtime stage copies only what's needed.
# This matters for cloud costs (smaller image = faster cold starts, lower storage).

FROM python:3.10-slim AS builder

WORKDIR /app

# Install dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# Only prod requirements (no pytest, no notebooks) in the final image
RUN pip install --no-cache-dir \
    pandas numpy scikit-learn xgboost imbalanced-learn \
    shap mlflow joblib pyyaml \
    fastapi uvicorn pydantic

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy only the venv from the builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the code needed at runtime
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
COPY config.yaml .

# Create a non-root user (security best practice)
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check so orchestrators (K8s, ECS) know when the service is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
