import uuid
import threading
from typing import Optional
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException, Depends
from app.models import JobResponse
from app.core.config import settings
from app.core.security import limiter, verify_api_key
from app.services import storage_service, gemini_service
from app.workers import process_summarization_job
from app.utils import create_job, cleanup_old_jobs

router = APIRouter()


@router.post("/summarize", response_model=JobResponse)
@limiter.limit("5/minute")
@limiter.limit(settings.GLOBAL_RATE_LIMIT, key_func=lambda: "global")
async def summarize(
    request: Request,
    audio: Optional[UploadFile] = File(None, description="Audio file to transcribe and summarize"),
    audio_url: Optional[str] = Form(None, description="URL to audio file (S3 or HTTP/HTTPS)"),
    text: Optional[str] = Form(None, description="Text transcript to summarize"),
    language: Optional[str] = Form(None, description="Language code for audio (e.g., 'en', 'id')"),
    clean_audio_flag: bool = Form(True, description="Clean audio before transcription (for audio input only)"),
    enable_polishing: bool = Form(True, description="Enable AI polishing before summarizing (uses API tokens)"),
    _: bool = Depends(verify_api_key)
):
    """
    Summarize audio or text (async processing)
    
    Send either:
    - **audio**: Audio file upload → Returns transcript + summary (language auto-detected)
    - **audio_url**: URL to audio file (S3 or HTTP/HTTPS) → Returns transcript + summary (language auto-detected)
    - **text**: Text string → Returns summary only (specify language parameter)
    
    - **language**: Language code for output ('en', 'id', etc.). For audio, auto-detected. For text, defaults to 'id'.
    - **clean_audio_flag**: Clean and normalize audio before transcription (default: true, for audio input only)
    - **enable_polishing**: Polish transcript with Gemini AI before summarizing (default: true, costs API tokens)
    
    If enable_polishing=true, the transcript will be polished before summarization for better quality.
    You can disable it to save Gemini API costs.
    
    The summary and next steps will be generated in the same language as the input/detected language.
    
    Returns job_id immediately. Poll GET /jobs/{job_id} for result.
    """
    
    # Validation
    provided_inputs = sum([bool(audio), bool(audio_url), bool(text)])
    if provided_inputs > 1:
        raise HTTPException(status_code=400, detail="Provide only one of: 'audio' file, 'audio_url', or 'text'")
    if provided_inputs == 0:
        raise HTTPException(status_code=400, detail="Must provide either 'audio' file, 'audio_url', or 'text' parameter")
    
    # Check if Gemini is configured
    if not gemini_service.is_available():
        raise HTTPException(
            status_code=503,
            detail="Summarization service not configured. Set GEMINI_API_KEY in environment."
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Determine input type
    if audio or audio_url:
        input_type = "audio"
        
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
        
        create_job(job_id, input_type)
        
        # Start background processing
        thread = threading.Thread(
            target=process_summarization_job,
            args=(job_id, input_type),
            kwargs={
                "content": content,
                "filename": filename,
                "language": language,
                "clean_audio_flag": clean_audio_flag,
                "enable_polishing": enable_polishing
            },
            daemon=True
        )
        thread.start()
    
    else:  # text input
        input_type = "text"
        
        if len(text) < 50:
            raise HTTPException(status_code=400, detail="Text too short. Minimum 50 characters.")
        
        create_job(job_id, input_type)
        
        # Start background processing
        thread = threading.Thread(
            target=process_summarization_job,
            args=(job_id, input_type),
            kwargs={"text": text, "language": language or "id"},
            daemon=True
        )
        thread.start()
    
    # Cleanup old jobs
    cleanup_old_jobs()
    
    return JobResponse(
        job_id=job_id,
        status="pending",
        input_type=input_type
    )
