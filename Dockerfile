FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and .env file
COPY whisper_api.py .
COPY app/ ./app/
COPY .env .

# Expose port (default 8000, but will use value from .env)
EXPOSE 8000

# Health check (uses default port 8000)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/health').read()" || exit 1

# Run with gunicorn (reads all config from .env via python-dotenv)
CMD ["sh", "-c", "gunicorn whisper_api:app --workers ${WORKERS:-2} --worker-class uvicorn.workers.UvicornWorker --bind ${HOST:-0.0.0.0}:${PORT:-8000} --timeout ${REQUEST_TIMEOUT:-600} --graceful-timeout 30 --max-requests 1000 --max-requests-jitter 100 --access-logfile - --error-logfile - --log-level info"]
