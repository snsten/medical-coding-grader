FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency manifest first for layer caching
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "openenv-core[core]>=0.2.2" && \
    pip install --no-cache-dir openai

# Copy application code
COPY models.py .
COPY server/ ./server/
COPY data/ ./data/

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Expose port
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
