import os
import time
import tempfile
import secrets
import asyncio
import uuid
import threading
from typing import Optional, Dict, Any
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Security, Depends, Request, Body
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pydub.effects import normalize
import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.core import Segment
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Whisper Transcription API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enable CORS for Go client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN")  # Required for diarization
MODEL_SIZE = os.getenv("MODEL_SIZE", "small")
ENABLE_DIARIZATION = os.getenv("ENABLE_DIARIZATION", "false").lower() == "true"
CPU_THREADS = int(os.getenv("CPU_THREADS", "4"))
API_KEY = os.getenv("API_KEY", None)  # If not set, auth is disabled

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Protection limits
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "3"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "600"))  # 10 minutes default
RATE_LIMIT = os.getenv("RATE_LIMIT", "10/minute")  # requests per minute per IP
GLOBAL_RATE_LIMIT = os.getenv("GLOBAL_RATE_LIMIT", "50/minute")  # total requests per minute (all IPs)
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "100")) * 1024 * 1024  # MB to bytes

# Job storage configuration
JOB_EXPIRY_SECONDS = int(os.getenv("JOB_EXPIRY_SECONDS", "3600"))  # Clean up jobs after 1 hour

# --- SECURITY ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key if authentication is enabled"""
    if API_KEY is None:
        return True  # Auth disabled
    
    if api_key is None or api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    return True

# Initialize Whisper Model (loaded once at startup)
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=CPU_THREADS)

# Initialize Gemini AI
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)
else:
    gemini_model = None
    print("Warning: GEMINI_API_KEY not set. Summarization will be disabled.")

# Request queue semaphore to limit concurrent processing
request_semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)

# Job storage (in-memory)
jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()

# Initialize Diarization Pipeline if enabled
diarization_pipeline = None
if ENABLE_DIARIZATION:
    if not HF_TOKEN:
        print("Warning: ENABLE_DIARIZATION=true but HF_TOKEN not set. Diarization will be disabled.")
    else:
        try:
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=HF_TOKEN
            )
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            diarization_pipeline.to(device)
        except Exception as e:
            print(f"Warning: Diarization pipeline failed to load: {e}")

# --- RESPONSE MODELS ---
class HealthResponse(BaseModel):
    status: str
    model: str
    diarization_enabled: bool
    max_concurrent_requests: int
    rate_limit: str
    global_rate_limit: str
    max_file_size_mb: int

class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None

class TranscriptionMetadata(BaseModel):
    audio_length: float
    language: str
    processing_time: float
    diarization_enabled: bool

class TranscriptionResponse(BaseModel):
    success: bool
    metadata: TranscriptionMetadata
    full_transcript: str
    segments: list[TranscriptSegment]

class SummarizeRequest(BaseModel):
    text: str

class JobResponse(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    input_type: str  # audio or text

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    input_type: str
    progress: Optional[str] = None  # transcribing, summarizing
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    processing_time: Optional[float] = None

# --- HELPER FUNCTIONS ---
def clean_audio(input_file: str, output_file: str) -> str:
    """Clean and normalize audio file"""
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    normalized_audio = normalize(audio)
    normalized_audio.export(output_file, format="wav")
    return output_file

# --- HELPER FUNCTIONS ---
def clean_audio(input_file: str, output_file: str) -> str:
    """Clean and normalize audio file"""
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    normalized_audio = normalize(audio)
    normalized_audio.export(output_file, format="wav")
    return output_file

def get_tracks(obj):
    """Extract speaker tracks from diarization object"""
    if hasattr(obj, 'itertracks'):
        return obj
    for attr in dir(obj):
        val = getattr(obj, attr, None)
        if hasattr(val, 'itertracks'):
            return val
    return None

def cleanup_old_jobs():
    """Remove expired jobs from storage"""
    with jobs_lock:
        current_time = time.time()
        expired_jobs = [
            job_id for job_id, job_data in jobs.items()
            if current_time - job_data.get("created_at_ts", 0) > JOB_EXPIRY_SECONDS
        ]
        for job_id in expired_jobs:
            del jobs[job_id]
        if expired_jobs:
            print(f"Cleaned up {len(expired_jobs)} expired jobs")

def summarize_text_with_gemini(text: str) -> str:
    """Summarize text using Gemini API"""
    if not gemini_model:
        raise Exception("Gemini API not configured. Set GEMINI_API_KEY in .env")
    
    prompt = f"""Summarize the following transcript in a clear, concise paragraph. 
Focus on the main topics, key points, and important information discussed.

Transcript:
{text}

Summary:"""
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise Exception(f"Gemini API error: {str(e)}")

def process_transcription_job(job_id: str, **kwargs):
    """Background worker to process transcription job"""
    with request_semaphore:
        try:
            with jobs_lock:
                jobs[job_id]["status"] = "processing"
                jobs[job_id]["progress"] = "transcribing"
            
            start_time = time.time()
            
            content = kwargs.get("content")
            filename = kwargs.get("filename")
            language = kwargs.get("language")
            enable_diarization = kwargs.get("enable_diarization", False)
            num_speakers = kwargs.get("num_speakers", 2)
            clean_audio_flag = kwargs.get("clean_audio_flag", True)
            
            # Save audio to temp file
            suffix = os.path.splitext(filename)[1] if filename else '.wav'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_input:
                temp_input.write(content)
                temp_input_path = temp_input.name
            
            try:
                # Clean audio if requested
                if clean_audio_flag:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_cleaned:
                        temp_cleaned_path = temp_cleaned.name
                    clean_audio(temp_input_path, temp_cleaned_path)
                    audio_path = temp_cleaned_path
                else:
                    audio_path = temp_input_path
                    temp_cleaned_path = None
                
                # Perform diarization if enabled
                diarization = None
                if enable_diarization and diarization_pipeline:
                    waveform, sample_rate = torchaudio.load(audio_path)
                    audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}
                    diarization = diarization_pipeline(audio_in_memory, num_speakers=num_speakers)
                
                # Transcribe
                segments, info = model.transcribe(
                    audio_path,
                    language=language,
                    beam_size=5,
                    multilingual=True,
                    initial_prompt="One Core, leads, CRM, call, terasa, kalau, follow up, baik, noted, catatan, panggilan, customer, outreach, lead, izin, informasikan, terima kasih, selamat, siang, sore, pagi, malam, halo, ya, tidak, oke, baiklah, nanti, yaudah, yaa, gitu, begitu, assalamualaikum, waalaikumsalam, menghubungi, panggilan, mulai, dari, sekala, kecil, meeting, penjadwalan, penjadualan, jadwal, telepon, teleponan, teleconference, video call, follow-up, follow up, follow up, hubungi, menghubungi, customer, pelanggan, klien, client, sales, marketing, bisnis, business, outreach, jangkauan, prospek, prospect, lead, sistem CRM, tenang, kotak katik, otak, atik, laporkan, lapor, acak acak, acak, backend, hasil, berdua, acak-acakan",
                    vad_filter=True,
                    condition_on_previous_text=False,
                    no_repeat_ngram_size=3,
                )
                
                # Process results
                full_text = ""
                segment_list = []
                
                if diarization:
                    speaker_data = get_tracks(diarization)
                    for segment in segments:
                        text_seg = Segment(segment.start, segment.end)
                        best_speaker = "Unknown"
                        max_overlap = 0
                        
                        if speaker_data:
                            for turn, _, speaker in speaker_data.itertracks(yield_label=True):
                                intersection = text_seg & turn
                                if intersection:
                                    overlap = intersection.duration
                                    if overlap > max_overlap:
                                        max_overlap = overlap
                                        best_speaker = speaker
                        
                        segment_list.append({
                            "start": round(segment.start, 2),
                            "end": round(segment.end, 2),
                            "text": segment.text.strip(),
                            "speaker": best_speaker
                        })
                        full_text += segment.text + " "
                else:
                    for segment in segments:
                        segment_list.append({
                            "start": round(segment.start, 2),
                            "end": round(segment.end, 2),
                            "text": segment.text.strip()
                        })
                        full_text += segment.text + " "
                
                processing_time = time.time() - start_time
                
                # Build result
                result = {
                    "success": True,
                    "metadata": {
                        "audio_length": round(info.duration, 2),
                        "language": info.language,
                        "processing_time": round(processing_time, 2),
                        "diarization_enabled": enable_diarization and diarization is not None
                    },
                    "full_transcript": full_text.strip(),
                    "segments": segment_list
                }
                
                # Update job as completed
                with jobs_lock:
                    jobs[job_id].update({
                        "status": "completed",
                        "result": result,
                        "completed_at": datetime.utcnow().isoformat() + "Z",
                        "processing_time": round(processing_time, 2),
                        "progress": None
                    })
            
            finally:
                # Cleanup temp files
                try:
                    os.unlink(temp_input_path)
                    if temp_cleaned_path:
                        os.unlink(temp_cleaned_path)
                except:
                    pass
        
        except Exception as e:
            with jobs_lock:
                jobs[job_id].update({
                    "status": "failed",
                    "error": str(e),
                    "failed_at": datetime.utcnow().isoformat() + "Z"
                })

def process_summarization_job(job_id: str, input_type: str, **kwargs):
    """Background worker to process summarization job"""
    with request_semaphore:
        try:
            with jobs_lock:
                jobs[job_id]["status"] = "processing"
            
            start_time = time.time()
            
            if input_type == "audio":
                # Step 1: Transcribe audio
                with jobs_lock:
                    jobs[job_id]["progress"] = "transcribing"
                
                content = kwargs.get("content")
                filename = kwargs.get("filename")
                language = kwargs.get("language")
                
                # Transcribe (reuse existing logic)
                suffix = os.path.splitext(filename)[1] if filename else '.wav'
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_input:
                    temp_input.write(content)
                    temp_input_path = temp_input.name
                
                try:
                    segments, info = model.transcribe(
                        temp_input_path,
                        language=language,
                        beam_size=5,
                        vad_filter=True
                    )
                    
                    full_text = ""
                    segment_list = []
                    for segment in segments:
                        full_text += segment.text + " "
                        segment_list.append({
                            "start": round(segment.start, 2),
                            "end": round(segment.end, 2),
                            "text": segment.text.strip()
                        })
                    
                    transcript_data = {
                        "full_text": full_text.strip(),
                        "segments": segment_list,
                        "language": info.language,
                        "duration": round(info.duration, 2)
                    }
                finally:
                    os.unlink(temp_input_path)
            
            else:  # text input
                full_text = kwargs.get("text")
                transcript_data = None
            
            # Step 2: Summarize with Gemini
            with jobs_lock:
                jobs[job_id]["progress"] = "summarizing"
            
            summary = summarize_text_with_gemini(full_text)
            
            # Build result
            result = {"summary": summary}
            if transcript_data:
                result["transcript"] = transcript_data
            
            processing_time = time.time() - start_time
            
            # Update job as completed
            with jobs_lock:
                jobs[job_id].update({
                    "status": "completed",
                    "result": result,
                    "completed_at": datetime.utcnow().isoformat() + "Z",
                    "processing_time": round(processing_time, 2),
                    "progress": None
                })
        
        except Exception as e:
            with jobs_lock:
                jobs[job_id].update({
                    "status": "failed",
                    "error": str(e),
                    "failed_at": datetime.utcnow().isoformat() + "Z"
                })

def get_tracks(obj):
    """Extract speaker tracks from diarization object"""
    if hasattr(obj, 'itertracks'):
        return obj
    for attr in dir(obj):
        val = getattr(obj, attr, None)
        if hasattr(val, 'itertracks'):
            return val
    return None

# --- API ENDPOINTS ---
@app.get("/health", response_model=HealthResponse)
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model=MODEL_SIZE,
        diarization_enabled=ENABLE_DIARIZATION,
        max_concurrent_requests=MAX_CONCURRENT_REQUESTS,
        rate_limit=RATE_LIMIT,
        global_rate_limit=GLOBAL_RATE_LIMIT,
        max_file_size_mb=MAX_FILE_SIZE // (1024 * 1024)
    )

@app.post("/transcribe", response_model=JobResponse)
@limiter.limit(RATE_LIMIT)  # Per-IP rate limit
@limiter.limit(GLOBAL_RATE_LIMIT, key_func=lambda: "global")  # Global rate limit
async def transcribe(
    request: Request,
    audio: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en', 'id'). Auto-detect if not specified"),
    enable_diarization: bool = Form(False, description="Enable speaker diarization"),
    num_speakers: int = Form(2, description="Number of speakers for diarization"),
    clean_audio_flag: bool = Form(True, description="Clean audio before transcription"),
    _: bool = Depends(verify_api_key)
):
    """
    Transcribe audio file with optional speaker diarization (async)
    
    - **audio**: Audio file (WAV, MP3, etc.)
    - **language**: Optional language code ('en', 'id', etc.)
    - **enable_diarization**: Enable speaker identification
    - **num_speakers**: Expected number of speakers (if diarization enabled)
    - **clean_audio_flag**: Normalize and clean audio before processing
    
    Returns job_id immediately. Poll GET /jobs/{job_id} for result.
    
    Rate limit: {RATE_LIMIT} per IP address
    Global rate limit: {GLOBAL_RATE_LIMIT} total (all IPs)
    Max file size: {MAX_FILE_SIZE // (1024 * 1024)} MB
    Processing time: 10s-5min depending on audio length
    """
    
    # Read file content
    content = await audio.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)} MB"
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat() + "Z"
    
    # Create job
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "input_type": "transcription",
            "created_at": created_at,
            "created_at_ts": time.time()
        }
    
    # Start background processing
    thread = threading.Thread(
        target=process_transcription_job,
        args=(job_id,),
        kwargs={
            "content": content,
            "filename": audio.filename,
            "language": language,
            "enable_diarization": enable_diarization,
            "num_speakers": num_speakers,
            "clean_audio_flag": clean_audio_flag
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

@app.post("/summarize", response_model=JobResponse)
@limiter.limit("5/minute")  # Stricter limit for summarization
@limiter.limit(GLOBAL_RATE_LIMIT, key_func=lambda: "global")
async def summarize(
    request: Request,
    audio: Optional[UploadFile] = File(None, description="Audio file to transcribe and summarize"),
    text: Optional[str] = Form(None, description="Text transcript to summarize"),
    language: Optional[str] = Form(None, description="Language code for audio (e.g., 'en', 'id')"),
    _: bool = Depends(verify_api_key)
):
    """
    Summarize audio or text (async processing)
    
    Send either:
    - **audio**: Audio file → Returns transcript + summary
    - **text**: Text string → Returns summary only
    
    Returns job_id immediately. Poll GET /jobs/{job_id} for result.
    
    Rate limit: 5/minute per IP (stricter than transcribe)
    Processing time: 30s-10min depending on audio length
    """
    
    # Validation: must provide audio OR text, not both or neither
    if audio and text:
        raise HTTPException(status_code=400, detail="Provide either 'audio' or 'text', not both")
    if not audio and not text:
        raise HTTPException(status_code=400, detail="Must provide either 'audio' file or 'text' parameter")
    
    # Check if Gemini is configured
    if not gemini_model:
        raise HTTPException(
            status_code=503,
            detail="Summarization service not configured. Set GEMINI_API_KEY in environment."
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat() + "Z"
    
    # Determine input type and prepare job data
    if audio:
        input_type = "audio"
        
        # Read file content
        content = await audio.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)} MB"
            )
        
        # Create job
        with jobs_lock:
            jobs[job_id] = {
                "job_id": job_id,
                "status": "pending",
                "input_type": input_type,
                "created_at": created_at,
                "created_at_ts": time.time()
            }
        
        # Start background processing
        thread = threading.Thread(
            target=process_summarization_job,
            args=(job_id, input_type),
            kwargs={
                "content": content,
                "filename": audio.filename,
                "language": language
            },
            daemon=True
        )
        thread.start()
    
    else:  # text input
        input_type = "text"
        
        if len(text) < 50:
            raise HTTPException(status_code=400, detail="Text too short. Minimum 50 characters.")
        
        # Create job
        with jobs_lock:
            jobs[job_id] = {
                "job_id": job_id,
                "status": "pending",
                "input_type": input_type,
                "created_at": created_at,
                "created_at_ts": time.time()
            }
        
        # Start background processing
        thread = threading.Thread(
            target=process_summarization_job,
            args=(job_id, input_type),
            kwargs={"text": text},
            daemon=True
        )
        thread.start()
    
    # Cleanup old jobs asynchronously
    cleanup_old_jobs()
    
    return JobResponse(
        job_id=job_id,
        status="pending",
        input_type=input_type
    )

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
@limiter.limit("60/minute")
async def get_job_status(
    request: Request,
    job_id: str,
    _: bool = Depends(verify_api_key)
):
    """
    Get status and result of a summarization job
    
    Status values:
    - **pending**: Job created, waiting to start
    - **processing**: Job is being processed (check 'progress' field)
    - **completed**: Job finished successfully (see 'result' field)
    - **failed**: Job failed (see 'error' field)
    
    Poll this endpoint every 3-5 seconds until status is 'completed' or 'failed'
    """
    
    with jobs_lock:
        job_data = jobs.get(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    
    return JobStatusResponse(**job_data)

if __name__ == '__main__':
    import uvicorn
    
    port = int(os.getenv('PORT', 8000))
    host = os.getenv('HOST', '0.0.0.0')
    workers = int(os.getenv('WORKERS', '1'))
    
    # Calculate reasonable connection limit
    # 3x concurrent requests to allow some buffering but prevent overload
    limit_max_requests = MAX_CONCURRENT_REQUESTS * 3
    
    print(f"Starting Whisper API on {host}:{port}")
    print(f"Model: {MODEL_SIZE}")
    print(f"CPU Threads: {CPU_THREADS}")
    print(f"Diarization: {'Enabled' if ENABLE_DIARIZATION else 'Disabled'}")
    print(f"Summarization: {'Enabled' if gemini_model else 'Disabled (set GEMINI_API_KEY to enable)'}")
    print(f"Authentication: {'Enabled' if API_KEY else 'Disabled (set API_KEY to enable)'}")
    print(f"\n--- Protection Limits ---")
    print(f"Max Concurrent Requests: {MAX_CONCURRENT_REQUESTS}")
    print(f"Connection Backlog: {limit_max_requests}")
    print(f"Rate Limit: {RATE_LIMIT} per IP")
    print(f"Global Rate Limit: {GLOBAL_RATE_LIMIT} total")
    print(f"Request Timeout: {REQUEST_TIMEOUT}s")
    print(f"Max File Size: {MAX_FILE_SIZE // (1024 * 1024)} MB")
    print(f"Job Expiry: {JOB_EXPIRY_SECONDS}s")
    print(f"\nDocs available at: http://{host}:{port}/docs")
    
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        timeout_keep_alive=30
    )
