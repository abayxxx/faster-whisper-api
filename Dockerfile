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
ENV WORKERS=1
ENV CPU_THREADS=4

# Run the application with gunicorn in production
CMD ["sh", "-c", "gunicorn whisper_api:app --workers ${WORKERS} --worker-class uvicorn.workers.UvicornWorker --bind ${HOST}:${PORT} --timeout 300 --access-logfile - --error-logfile -"]
