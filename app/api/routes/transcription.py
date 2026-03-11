import uuid
import threading
from typing import Optional
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException, Depends
from app.models import JobResponse
from app.core.config import settings
from app.core.security import limiter, verify_api_key
from app.services import storage_service
from app.workers import process_transcription_job
from app.utils import create_job, cleanup_old_jobs

router = APIRouter()


@router.post("/transcribe", response_model=JobResponse)
@limiter.limit(settings.RATE_LIMIT)
@limiter.limit(settings.GLOBAL_RATE_LIMIT, key_func=lambda: "global")
async def transcribe(
    request: Request,
    audio: Optional[UploadFile] = File(None, description="Audio file to transcribe"),
    audio_url: Optional[str] = Form(None, description="URL to audio file (S3 or HTTP/HTTPS)"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en', 'id'). Auto-detect if not specified"),
    enable_diarization: bool = Form(False, description="Enable speaker diarization"),
    num_speakers: int = Form(2, description="Number of speakers for diarization"),
    clean_audio_flag: bool = Form(True, description="Clean audio before transcription"),
    enable_polishing: bool = Form(True, description="Enable AI polishing with Gemini (uses API tokens)"),
    _: bool = Depends(verify_api_key)
):
    """
    Transcribe audio file with optional speaker diarization and AI polishing (async)
    
    Provide either:
    - **audio**: Audio file upload (WAV, MP3, etc.)
    - **audio_url**: URL to audio file (S3 or HTTP/HTTPS)
    
    - **language**: Optional language code ('en', 'id', etc.)
    - **enable_diarization**: Enable speaker identification
    - **num_speakers**: Expected number of speakers (if diarization enabled)
    - **clean_audio_flag**: Normalize and clean audio before processing
    - **enable_polishing**: Enable AI polishing with Gemini (costs API tokens, default: true)
    
    If enable_polishing=true, transcript will be polished with Gemini AI (if configured) to:
    - Fix grammar, punctuation, and capitalization
    - Remove filler words (um, uh, etc.)
    - Correct misheard words based on context
    - Optimize for CRM/telemarketing call quality
    
    Returns job_id immediately. Poll GET /jobs/{job_id} for result.
    """
    
    # Validation
    if audio and audio_url:
        raise HTTPException(status_code=400, detail="Provide either 'audio' file or 'audio_url', not both")
    if not audio and not audio_url:
        raise HTTPException(status_code=400, detail="Must provide either 'audio' file or 'audio_url' parameter")
    
    # Get audio content and filename
    if audio:
        content = await audio.read()
        if len(content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_MB} MB"
            )
        filename = audio.filename
    else:
        content, filename = storage_service.download_file_from_url(audio_url)
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    create_job(job_id, "transcription")
    
    # Start background processing
    thread = threading.Thread(
        target=process_transcription_job,
        args=(job_id,),
        kwargs={
            "content": content,
            "filename": filename,
            "language": language,
            "enable_diarization": enable_diarization,
            "num_speakers": num_speakers,
            "clean_audio_flag": clean_audio_flag,
            "enable_polishing": enable_polishing
        },
        daemon=True
    )
    thread.start()
    
    # Cleanup old jobs
    cleanup_old_jobs()
    
    return JobResponse(
        job_id=job_id,
        status="pending",
        input_type="transcription"
    )
