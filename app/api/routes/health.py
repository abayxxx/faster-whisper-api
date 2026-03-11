from fastapi import APIRouter, Request
from app.models import HealthResponse
from app.core.config import settings
from app.core.security import limiter
from app.services import gemini_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model=settings.MODEL_SIZE,
        diarization_enabled=settings.is_diarization_available,
        polishing_enabled=gemini_service.is_available(),
        max_concurrent_requests=settings.MAX_CONCURRENT_REQUESTS,
        rate_limit=settings.RATE_LIMIT,
        global_rate_limit=settings.GLOBAL_RATE_LIMIT,
        max_file_size_mb=settings.MAX_FILE_SIZE_MB
    )
