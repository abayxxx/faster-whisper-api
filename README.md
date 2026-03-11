# 🎙️ Faster Whisper API

> Production-ready audio transcription API with speaker diarization, AI polishing, and summarization powered by OpenAI Whisper and Google Gemini.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ Features

- 🎤 **Audio Transcription** - Fast, accurate transcription using OpenAI Whisper
- 👥 **Speaker Diarization** - Identify and label different speakers (optional)
- 🤖 **AI Polishing** - Clean transcripts with Gemini AI (remove filler words, fix grammar)
- 📝 **Text Summarization** - Generate summaries and next-step suggestions
- ☁️ **S3 & URL Support** - Process audio from S3 buckets or HTTP/HTTPS URLs
- ⚡ **Async Processing** - Job-based system with real-time status updates
- 🔒 **Production Ready** - Rate limiting, API authentication, concurrent request control
- 🐳 **Docker Support** - Easy deployment with Docker and Docker Compose

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#️-configuration)
- [API Endpoints](#-api-endpoints)
- [Usage Examples](#-usage-examples)
- [Project Structure](#-project-structure)
- [Docker Deployment](#-docker-deployment)
- [Development](#-development)
- [Documentation](#-documentation)
- [License](#-license)

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- FFmpeg (for audio processing)

### 1. Clone the repository

```bash
git clone https://github.com/abayxxx/faster-whisper-api.git
cd faster-whisper-api
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 4. Run the server

```bash
python whisper_api.py
```

The API will be available at `http://localhost:8000`

**API Documentation:** http://localhost:8000/docs

## 📦 Installation

### Option 1: Python (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python whisper_api.py
```

### Option 2: Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python whisper_api.py
```

### Option 3: Docker (Production)

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
# Model Configuration
MODEL_SIZE=small                    # Options: tiny, base, small, medium, large
ENABLE_DIARIZATION=false           # Enable speaker identification
CPU_THREADS=4                      # Number of CPU threads

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=2                          # Number of Gunicorn workers

# Protection & Limits
MAX_CONCURRENT_REQUESTS=3          # Max concurrent transcriptions
REQUEST_TIMEOUT=600                # Request timeout in seconds
RATE_LIMIT=10/minute              # Per-IP rate limit
GLOBAL_RATE_LIMIT=50/minute       # Global rate limit
MAX_FILE_SIZE_MB=100              # Maximum file size
JOB_EXPIRY_SECONDS=3600           # Job cleanup time

# Security (Optional)
API_KEY=your-secret-key-here       # Enable API key authentication

# Diarization (Optional - requires HuggingFace token)
HF_TOKEN=your-huggingface-token    # For speaker diarization

# AI Features (Optional - requires Gemini API key)
GEMINI_API_KEY=your-gemini-key     # For polishing & summarization
GEMINI_MODEL=gemini-1.5-flash      # Gemini model to use

# AWS S3 (Optional - for S3 URL support)
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
AWS_REGION=us-east-1
```

### Model Sizes

| Model    | Size   | Speed   | Accuracy |
| -------- | ------ | ------- | -------- |
| `tiny`   | 75 MB  | Fastest | Lower    |
| `base`   | 145 MB | Fast    | Good     |
| `small`  | 488 MB | Medium  | Better   |
| `medium` | 1.5 GB | Slow    | High     |
| `large`  | 3.1 GB | Slowest | Highest  |

## 🌐 API Endpoints

### Health Check

```http
GET /health
```

Check API status and configuration.

### Transcribe Audio

```http
POST /transcribe
```

Transcribe audio files with optional speaker diarization and AI polishing.

**Parameters:**

- `audio` (file) - Audio file to transcribe
- `audio_url` (string) - Or URL to audio file (S3/HTTP/HTTPS)
- `language` (string, optional) - Language code (auto-detect if not specified)
- `enable_diarization` (boolean, default: false) - Enable speaker identification
- `num_speakers` (integer, default: 2) - Expected number of speakers
- `clean_audio_flag` (boolean, default: true) - Clean audio before processing
- `enable_polishing` (boolean, default: true) - Enable AI polishing with Gemini

**Returns:** Job ID for polling

### Summarize

```http
POST /summarize
```

Summarize audio or text transcript.

**Parameters:**

- `audio` (file) - Audio file to transcribe and summarize
- `audio_url` (string) - Or URL to audio file
- `text` (string) - Or text transcript to summarize
- `language` (string, optional) - Language code

**Returns:** Job ID for polling

### Get Job Status

```http
GET /jobs/{job_id}
```

Get status and result of a job.

**Returns:**

- `status`: pending, processing, completed, failed
- `result`: Job result (when completed)
- `error`: Error message (if failed)

## 💡 Usage Examples

### Basic Transcription

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@recording.wav" \
  -F "language=en"
```

Response:

```json
{
  "job_id": "abc-123-def-456",
  "status": "pending",
  "input_type": "transcription"
}
```

### Check Job Status

```bash
curl http://localhost:8000/jobs/abc-123-def-456
```

Response:

```json
{
  "job_id": "abc-123-def-456",
  "status": "completed",
  "result": {
    "success": true,
    "metadata": {
      "audio_length": 120.5,
      "language": "en",
      "processing_time": 25.3,
      "polished": true
    },
    "full_transcript": "Hello, this is a test recording...",
    "segments": [...]
  }
}
```

### With Speaker Diarization

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@meeting.wav" \
  -F "enable_diarization=true" \
  -F "num_speakers=3"
```

### Transcribe from URL

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "audio_url=https://example.com/audio.mp3" \
  -F "language=en"
```

### Transcribe from S3

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "audio_url=https://bucket.s3.amazonaws.com/audio.wav"
```

### Summarize Audio

```bash
curl -X POST http://localhost:8000/summarize \
  -F "audio=@meeting.wav" \
  -F "language=en"
```

### Summarize Text

```bash
curl -X POST http://localhost:8000/summarize \
  -F "text=Your transcript here..." \
  -F "language=en"
```

### With API Key Authentication

```bash
curl -X POST http://localhost:8000/transcribe \
  -H "X-API-Key: your-secret-key" \
  -F "audio=@recording.wav"
```

### Disable AI Polishing (Save API Costs)

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@recording.wav" \
  -F "enable_polishing=false"
```

## 📁 Project Structure

```
faster-whisper-api/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application
│   ├── api/
│   │   └── routes/
│   │       ├── health.py          # Health check endpoint
│   │       ├── transcription.py   # Transcription endpoint
│   │       ├── summarization.py   # Summarization endpoint
│   │       └── jobs.py            # Job status endpoint
│   ├── core/
│   │   ├── config.py              # Configuration (Pydantic Settings)
│   │   └── security.py            # API key & rate limiting
│   ├── models/
│   │   └── responses.py           # Pydantic response models
│   ├── services/
│   │   ├── whisper.py             # Whisper transcription
│   │   ├── diarization.py         # Speaker diarization
│   │   ├── gemini.py              # AI polishing & summarization
│   │   ├── storage.py             # S3 & URL downloads
│   │   └── audio.py               # Audio processing
│   ├── workers/
│   │   ├── transcription.py       # Background transcription job
│   │   └── summarization.py       # Background summarization job
│   └── utils/
│       └── jobs.py                # Job storage & management
├── whisper_api.py                 # Entry point (backward compatible)
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker Compose configuration
├── .env.example                   # Environment variables template
└── README.md                      # This file
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t faster-whisper-api .

# Run the container
docker run -p 8000:8000 --env-file .env faster-whisper-api
```

### Using Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

### docker-compose.yml Example

```yaml
version: "3.8"

services:
  whisper-api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

## 🛠️ Development

### Running Locally

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or use the wrapper
python whisper_api.py
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

### Code Structure

The project follows a modular architecture:

- **Services** - Business logic (transcription, AI, storage)
- **Routes** - API endpoints (thin controllers)
- **Workers** - Background job processing
- **Models** - Data validation (Pydantic)
- **Core** - Configuration & security

### Adding New Features

1. **New service** - Add to `app/services/`
2. **New endpoint** - Add to `app/api/routes/`
3. **New model** - Add to `app/models/`
4. **Background task** - Add to `app/workers/`

## 📖 Documentation

- **API Docs (Interactive):** http://localhost:8000/docs
- **API Docs (ReDoc):** http://localhost:8000/redoc
- **OpenAPI Schema:** http://localhost:8000/openapi.json

### Additional Documentation

- `HOW_TO_RUN.md` - Detailed running instructions
- `ARCHITECTURE.md` - Project architecture explained
- `REFACTORING_SUCCESS.md` - Refactoring details

## 🔒 Security

### API Key Authentication

Enable by setting `API_KEY` in `.env`:

```env
API_KEY=your-secret-key-here
```

Then include in requests:

```bash
curl -H "X-API-Key: your-secret-key" http://localhost:8000/transcribe
```

### Rate Limiting

- **Per-IP limit:** 10 requests/minute (configurable)
- **Global limit:** 50 requests/minute (configurable)
- **Concurrent requests:** 3 maximum (configurable)

### File Size Limits

- Default: 100 MB per file
- Configurable via `MAX_FILE_SIZE_MB`

## 🚦 Processing Time

| Audio Length | Transcription | With Diarization | With Polishing |
| ------------ | ------------- | ---------------- | -------------- |
| 1 minute     | ~10s          | ~15s             | ~13s           |
| 5 minutes    | ~30s          | ~45s             | ~35s           |
| 10 minutes   | ~60s          | ~90s             | ~68s           |
| 30 minutes   | ~180s         | ~270s            | ~195s          |

\*Times vary based on CPU, model size, and audio quality.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Fast Whisper implementation
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [Google Gemini](https://ai.google.dev/) - AI polishing and summarization
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework

## 📞 Support

For issues, questions, or contributions, please visit the [GitHub Issues](https://github.com/abayxxx/faster-whisper-api/issues) page.

---

**Made with ❤️ for better audio transcription**
