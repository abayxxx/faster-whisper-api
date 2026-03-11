from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from app.core.config import settings
from app.core.security import limiter
from app.api.routes import health, transcription, summarization, jobs


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="Whisper Transcription API",
        version="1.0.0",
        description="Fast, production-ready audio transcription with AI polishing and summarization"
    )
    
    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(transcription.router, tags=["Transcription"])
    app.include_router(summarization.router, tags=["Summarization"])
    app.include_router(jobs.router, tags=["Jobs"])
    
    return app


app = create_app()


def __main__():
    """Main entry point for running the application"""
    import uvicorn
    
    print(f"Starting Whisper API on {settings.HOST}:{settings.PORT}")
    print(f"Model: {settings.MODEL_SIZE}")
    print(f"CPU Threads: {settings.CPU_THREADS}")
    print(f"Diarization: {'Enabled' if settings.is_diarization_available else 'Disabled'}")
    print(f"Transcript Polishing: {'Enabled' if settings.is_gemini_available else 'Disabled (set GEMINI_API_KEY to enable)'}")
    print(f"Summarization: {'Enabled' if settings.is_gemini_available else 'Disabled (set GEMINI_API_KEY to enable)'}")
    print(f"Authentication: {'Enabled' if settings.is_auth_enabled else 'Disabled (set API_KEY to enable)'}")
    print(f"\n--- Protection Limits ---")
    print(f"Max Concurrent Requests: {settings.MAX_CONCURRENT_REQUESTS}")
    print(f"Rate Limit: {settings.RATE_LIMIT} per IP")
    print(f"Global Rate Limit: {settings.GLOBAL_RATE_LIMIT} total")
    print(f"Request Timeout: {settings.REQUEST_TIMEOUT}s")
    print(f"Max File Size: {settings.MAX_FILE_SIZE_MB} MB")
    print(f"Job Expiry: {settings.JOB_EXPIRY_SECONDS}s")
    print(f"\nDocs available at: http://{settings.HOST}:{settings.PORT}/docs")
    
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        timeout_keep_alive=30
    )


if __name__ == "__main__":
    __main__()
