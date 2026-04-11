# Root-level Dockerfile — used by the openenv validation script (docker build <repo_root>).
# Build context is the project root, so all files (models.py, data/, server/) are available.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency manifest first for layer caching
COPY pyproject.toml ./

# Install runtime dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "openenv-core[core]>=0.2.2" "openai>=1.0.0" "uvicorn[standard]>=0.29.0" "gradio>=4.0.0"

# Copy application code
COPY models.py ./
COPY server/ ./server/
COPY data/ ./data/

# Expose project root so all relative imports resolve
ENV PYTHONPATH="/app:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
