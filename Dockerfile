# ── Stage 1: Build dependencies ─────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /tmp

COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt


# ── Stage 2: Production image ──────────────────────────────────────────────
FROM python:3.13-slim

# Don't buffer Python output (logs appear in real time)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY config.py log_config.py ./
COPY app/ ./app/
COPY api/ ./api/
COPY models/ ./models/
COPY data/processed/ ./data/processed/
COPY src/ ./src/

# Create logs directory
RUN mkdir -p logs

# Expose the API port
EXPOSE 8000

# Run the API with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
