# Whisper Transcription API

## Why FastAPI Instead of Flask?

**FastAPI is better for this use case because:**

1. **Async/Await Support**: Native async support for better performance with I/O operations
2. **Automatic API Documentation**: Built-in Swagger UI at `/docs` and ReDoc at `/redoc`
3. **Type Safety**: Pydantic models provide automatic validation and serialization
4. **Better Performance**: Generally faster than Flask, especially for API services
5. **Modern Python**: Uses Python 3.6+ type hints for better code quality
6. **Easy Go Integration**: Clear request/response models make Go client implementation straightforward

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables (optional):
```bash
export HF_TOKEN="your_huggingface_token"
export MODEL_SIZE="small"  # or "medium", "large"
export ENABLE_DIARIZATION="false"  # or "true"
export PORT=8000
export HOST="0.0.0.0"
```

3. Run the API:
```bash
python whisper_api.py
```

Or with uvicorn directly:
```bash
uvicorn whisper_api:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model": "small",
  "diarization_enabled": false
}
```

### Transcribe Audio
```bash
POST /transcribe
```

Parameters (multipart/form-data):
- `audio`: Audio file (required)
- `language`: Language code like 'en', 'id' (optional, auto-detect if not specified)
- `enable_diarization`: true/false (optional, default: false)
- `num_speakers`: Number of speakers (optional, default: 2)
- `clean_audio_flag`: true/false (optional, default: true)

Response:
```json
{
  "success": true,
  "metadata": {
    "audio_length": 45.23,
    "language": "en",
    "processing_time": 12.5,
    "diarization_enabled": false
  },
  "full_transcript": "Complete transcript text...",
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "text": "Hello world",
      "speaker": "SPEAKER_00"
    }
  ]
}
```

## Testing with cURL

```bash
# Health check
curl http://localhost:8000/health

# Transcribe without diarization
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@path/to/audio.wav" \
  -F "language=en"

# Transcribe with diarization
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@path/to/audio.wav" \
  -F "language=en" \
  -F "enable_diarization=true" \
  -F "num_speakers=2"
```

## Go Client Usage

See `example_go_client.go` for a complete example.

```go
import "your-project/transcription"

// Check if API is healthy
health, err := transcription.CheckHealth("http://localhost:8000")
if err != nil {
    log.Fatal(err)
}

// Transcribe audio
result, err := transcription.TranscribeAudio(
    "http://localhost:8000",
    "audio.wav",
    "en",           // language
    false,          // enable_diarization
    2,              // num_speakers
)
if err != nil {
    log.Fatal(err)
}

fmt.Println(result.FullTranscript)
```

## Interactive API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

You can test the API directly from the browser!

## Production Deployment

For production, use a production ASGI server:

```bash
# Install gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn whisper_api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

Or use Docker:

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY whisper_api.py .

CMD ["uvicorn", "whisper_api:app", "--host", "0.0.0.0", "--port", "8000"]
```
