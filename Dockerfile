FROM python:3.12-slim

ENV PORT=7860 \
    HOST=0.0.0.0 \
    LOG_LEVEL=info \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (appuser)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code with non-root ownership
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Hugging Face Spaces & Container orchestrators expose port 7860 by default
EXPOSE 7860

# Healthcheck directive pointing to /healthz
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:${PORT}/healthz || exit 1

# Run FastAPI server
CMD ["python", "server.py"]
