from typing import Optional, List, Any, Dict
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model: str
    diarization_enabled: bool
    polishing_enabled: bool
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
    polished: bool


class TranscriptionResponse(BaseModel):
    success: bool
    metadata: TranscriptionMetadata
    full_transcript: str
    segments: List[TranscriptSegment]


class JobResponse(BaseModel):
    job_id: str
    status: str
    input_type: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    input_type: str
    progress: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    processing_time: Optional[float] = None
