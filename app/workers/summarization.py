import os
import time
import tempfile
import threading
from app.core.config import settings
from app.services import whisper_service, gemini_service
from app.utils import update_job

request_semaphore = threading.Semaphore(settings.MAX_CONCURRENT_REQUESTS)


def process_summarization_job(job_id: str, input_type: str, **kwargs):
    """Background worker to process summarization job"""
    with request_semaphore:
        try:
            update_job(job_id, {"status": "processing"})
            
            start_time = time.time()
            
            if input_type == "audio":
                # Step 1: Transcribe audio
                update_job(job_id, {"progress": "transcribing"})
                
                content = kwargs.get("content")
                filename = kwargs.get("filename")
                language = kwargs.get("language")
                
                suffix = os.path.splitext(filename)[1] if filename else '.wav'
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_input:
                    temp_input.write(content)
                    temp_input_path = temp_input.name
                
                try:
                    segments, info = whisper_service.transcribe(temp_input_path, language)
                    
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
                language = kwargs.get("language", "id")
            
            # Check if transcript is empty
            if not full_text or not full_text.strip():
                raise Exception("Transcript is empty. Nothing to summarize.")
            
            # Step 2: Summarize with Gemini
            update_job(job_id, {"progress": "summarizing"})
            
            if input_type == "audio":
                detected_language = info.language if hasattr(info, 'language') else (language or "id")
            else:
                detected_language = language
            
            gemini_result = gemini_service.summarize_text(full_text.strip(), detected_language)
            
            # Build result
            result = {
                "summary": gemini_result["summary"],
                "next_steps_suggestion": gemini_result["next_steps_suggestion"]
            }
            if transcript_data:
                result["transcript"] = transcript_data
            
            processing_time = time.time() - start_time
            
            # Update job as completed
            update_job(job_id, {
                "status": "completed",
                "result": result,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "processing_time": round(processing_time, 2),
                "progress": None
            })
        
        except Exception as e:
            update_job(job_id, {
                "status": "failed",
                "error": str(e),
                "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            })
