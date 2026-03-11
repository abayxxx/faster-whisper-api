import torch
import torchaudio
from typing import Optional
from pyannote.audio import Pipeline
from app.core.config import settings


class DiarizationService:
    """Speaker diarization service using pyannote.audio"""
    
    def __init__(self):
        self.pipeline = None
        if settings.is_diarization_available:
            try:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=settings.HF_TOKEN
                )
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
                self.pipeline.to(device)
                print("Diarization pipeline initialized successfully")
            except Exception as e:
                print(f"Warning: Diarization pipeline failed to load: {e}")
        else:
            if settings.ENABLE_DIARIZATION and not settings.HF_TOKEN:
                print("Warning: ENABLE_DIARIZATION=true but HF_TOKEN not set. Diarization will be disabled.")
    
    def is_available(self) -> bool:
        """Check if diarization is available"""
        return self.pipeline is not None
    
    def diarize(self, audio_path: str, num_speakers: int = 2):
        """Perform speaker diarization on audio file"""
        if not self.is_available():
            return None
        
        waveform, sample_rate = torchaudio.load(audio_path)
        audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}
        return self.pipeline(audio_in_memory, num_speakers=num_speakers)
    
    @staticmethod
    def get_tracks(diarization_obj):
        """Extract speaker tracks from diarization object"""
        if hasattr(diarization_obj, 'itertracks'):
            return diarization_obj
        for attr in dir(diarization_obj):
            val = getattr(diarization_obj, attr, None)
            if hasattr(val, 'itertracks'):
                return val
        return None


diarization_service = DiarizationService()
