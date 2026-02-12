import os
import time
import tempfile
import secrets
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Security, Depends
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

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Whisper Transcription API", version="1.0.0")

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

# --- API ENDPOINTS ---
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model=MODEL_SIZE,
        diarization_enabled=ENABLE_DIARIZATION
    )

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    audio: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en', 'id'). Auto-detect if not specified"),
    enable_diarization: bool = Form(False, description="Enable speaker diarization"),
    num_speakers: int = Form(2, description="Number of speakers for diarization"),
    clean_audio_flag: bool = Form(True, description="Clean audio before transcription"),
    _: bool = Depends(verify_api_key)  # Underscore shows we don't use the value
):
    """
    Transcribe audio file with optional speaker diarization
    
    - **audio**: Audio file (WAV, MP3, etc.)
    - **language**: Optional language code ('en', 'id', etc.)
    - **enable_diarization**: Enable speaker identification
    - **num_speakers**: Expected number of speakers (if diarization enabled)
    - **clean_audio_flag**: Normalize and clean audio before processing
    """
    try:
        start_time = time.time()
        
        # Save uploaded file temporarily
        suffix = os.path.splitext(audio.filename)[1] if audio.filename else '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_input:
            content = await audio.read()
            temp_input.write(content)
            temp_input_path = temp_input.name
        
        # Clean audio if requested
        if clean_audio_flag:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_cleaned:
                temp_cleaned_path = temp_cleaned.name
            clean_audio(temp_input_path, temp_cleaned_path)
            audio_path = temp_cleaned_path
        else:
            audio_path = temp_input_path
        
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
        transcript_data = []
        full_text = ""
        
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
                
                transcript_data.append(TranscriptSegment(
                    start=round(segment.start, 2),
                    end=round(segment.end, 2),
                    text=segment.text.strip(),
                    speaker=best_speaker
                ))
                full_text += segment.text + " "
        else:
            for segment in segments:
                transcript_data.append(TranscriptSegment(
                    start=round(segment.start, 2),
                    end=round(segment.end, 2),
                    text=segment.text.strip()
                ))
                full_text += segment.text + " "
        
        processing_time = time.time() - start_time
        
        # Cleanup temporary files
        try:
            os.unlink(temp_input_path)
            if clean_audio_flag:
                os.unlink(temp_cleaned_path)
        except:
            pass
        
        # Return response
        return TranscriptionResponse(
            success=True,
            metadata=TranscriptionMetadata(
                audio_length=round(info.duration, 2),
                language=info.language,
                processing_time=round(processing_time, 2),
                diarization_enabled=enable_diarization and diarization is not None
            ),
            full_transcript=full_text.strip(),
            segments=transcript_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    
    port = int(os.getenv('PORT', 8000))
    host = os.getenv('HOST', '0.0.0.0')
    
    print(f"Starting Whisper API on {host}:{port}")
    print(f"Model: {MODEL_SIZE}")
    print(f"CPU Threads: {CPU_THREADS}")
    print(f"Diarization: {'Enabled' if ENABLE_DIARIZATION else 'Disabled'}")
    print(f"Authentication: {'Enabled' if API_KEY else 'Disabled (set API_KEY to enable)'}")
    print(f"Docs available at: http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port)
