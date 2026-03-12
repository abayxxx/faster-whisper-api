from faster_whisper import WhisperModel
from app.core.config import settings


class WhisperService:
    """Whisper transcription service"""
    
    def __init__(self):
        self.model = WhisperModel(
            settings.MODEL_SIZE,
            device="cpu",
            compute_type="int8",
            cpu_threads=settings.CPU_THREADS
        )
        print(f"Whisper model '{settings.MODEL_SIZE}' loaded successfully")
    
    def transcribe(self, audio_path: str, language: str = None):
        """Transcribe audio file"""
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            multilingual=True,
            initial_prompt="",
            vad_filter=True,
            condition_on_previous_text=False,
            no_repeat_ngram_size=3,
        )
        return segments, info


whisper_service = WhisperService()
