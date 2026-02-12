# API Authentication

## Security

This API supports optional API key authentication using the `X-API-Key` header.

### Enable Authentication

Set the `API_KEY` environment variable:

```bash
# Generate a secure API key
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Add to .env file
API_KEY=yp4x-4uAPP7NBb_6JyxiWrvExYX5BRBKtcCLDxvi_xI
```

### Disable Authentication

Remove or comment out `API_KEY` from `.env`:

```bash
# API_KEY=
```

## Using the API

### With Authentication (cURL)

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "X-API-Key: your-api-key-here" \
  -F "audio=@audio.wav" \
  -F "language=en"
```

### With Authentication (Go Client)

```go
req, _ := http.NewRequest("POST", url, body)
req.Header.Set("Content-Type", writer.FormDataContentType())
req.Header.Set("X-API-Key", "your-api-key-here")
```

### With Authentication (Python)

```python
import requests

headers = {"X-API-Key": "your-api-key-here"}
files = {"audio": open("audio.wav", "rb")}
data = {"language": "en"}

response = requests.post(
    "http://localhost:8000/transcribe",
    headers=headers,
    files=files,
    data=data
)
```

## Docker with Authentication

```bash
docker run -p 8000:8000 \
  -e API_KEY="your-secret-key" \
  -e HF_TOKEN="your-hf-token" \
  whisper-api
```

## Security Best Practices

1. **Never commit** API keys to version control
2. **Use strong keys** - minimum 32 characters, cryptographically random
3. **Rotate keys** regularly in production
4. **Use HTTPS** in production to protect keys in transit
5. **Consider rate limiting** for additional protection
