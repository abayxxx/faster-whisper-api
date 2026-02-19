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

# Copy application code
COPY whisper_api.py .

# Expose port
EXPOSE 8000

# Set environment variables with defaults
ENV HOST=0.0.0.0
ENV PORT=8000
ENV MODEL_SIZE=small
ENV ENABLE_DIARIZATION=false
ENV CPU_THREADS=4

# Protection limits
ENV MAX_CONCURRENT_REQUESTS=2
ENV REQUEST_TIMEOUT=600
ENV RATE_LIMIT=20/minute
ENV MAX_FILE_SIZE_MB=100

# Gemini AI for summarization
ENV GEMINI_API_KEY=
ENV GEMINI_MODEL=gemini-1.5-flash
ENV JOB_EXPIRY_SECONDS=3600

# Gunicorn workers (calculate based on your server)
# Formula: min(CPU_cores / 2, RAM_GB / 4)
ENV WORKERS=2

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health').read()" || exit 1

# Run with gunicorn + uvicorn workers (production setup)
CMD gunicorn whisper_api:app \
    --workers ${WORKERS} \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind ${HOST}:${PORT} \
    --timeout ${REQUEST_TIMEOUT} \
    --graceful-timeout 30 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --access-logfile - \
    --error-logfile - \
    --log-level info
