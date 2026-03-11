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
            initial_prompt="One Core, leads, CRM, call, terasa, kalau, follow up, baik, noted, catatan, panggilan, customer, outreach, lead, izin, informasikan, terima kasih, selamat, siang, sore, pagi, malam, halo, ya, tidak, oke, baiklah, nanti, yaudah, yaa, gitu, begitu, assalamualaikum, waalaikumsalam, menghubungi, panggilan, mulai, dari, sekala, kecil, meeting, penjadwalan, penjadualan, jadwal, telepon, teleponan, teleconference, video call, follow-up, follow up, follow up, hubungi, menghubungi, customer, pelanggan, klien, client, sales, marketing, bisnis, business, outreach, jangkauan, prospek, prospect, lead, sistem CRM, tenang, kotak katik, otak, atik, laporkan, lapor, acak acak, acak, backend, hasil, berdua, acak-acakan",
            vad_filter=True,
            condition_on_previous_text=False,
            no_repeat_ngram_size=3,
        )
        return segments, info


whisper_service = WhisperService()
