import os
import time
import tempfile
import threading
from pyannote.core import Segment
from app.core.config import settings
from app.services import whisper_service, diarization_service, gemini_service, clean_audio
from app.utils import update_job

request_semaphore = threading.Semaphore(settings.MAX_CONCURRENT_REQUESTS)


def process_transcription_job(job_id: str, **kwargs):
    """Background worker to process transcription job"""
    with request_semaphore:
        try:
            update_job(job_id, {"status": "processing", "progress": "transcribing"})
            
            start_time = time.time()
            
            content = kwargs.get("content")
            filename = kwargs.get("filename")
            language = kwargs.get("language")
            enable_diarization = kwargs.get("enable_diarization", False)
            num_speakers = kwargs.get("num_speakers", 2)
            clean_audio_flag = kwargs.get("clean_audio_flag", True)
            enable_polishing = kwargs.get("enable_polishing", True)
            
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
                if enable_diarization and diarization_service.is_available():
                    diarization = diarization_service.diarize(audio_path, num_speakers)
                
                # Transcribe
                segments, info = whisper_service.transcribe(audio_path, language)
                
                # Process results
                segment_list = []
                
                if diarization:
                    speaker_data = diarization_service.get_tracks(diarization)
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
                else:
                    for segment in segments:
                        segment_list.append({
                            "start": round(segment.start, 2),
                            "end": round(segment.end, 2),
                            "text": segment.text.strip()
                        })
                
                # Build full transcript from segments
                polished_transcript = " ".join([seg["text"] for seg in segment_list]).strip()
                
                # Polish if enabled
                polishing_applied = False
                if enable_polishing and polished_transcript and gemini_service.is_available():
                    update_job(job_id, {"progress": "polishing"})
                    
                    try:
                        detected_language = info.language if hasattr(info, 'language') else (language or "id")
                        
                        segment_texts = [seg["text"] for seg in segment_list]
                        polished_segment_texts = gemini_service.polish_segments_batch(segment_texts, detected_language)
                        
                        for i, polished_text in enumerate(polished_segment_texts):
                            if i < len(segment_list):
                                segment_list[i]["text"] = polished_text
                        
                        polished_transcript = " ".join([seg["text"] for seg in segment_list]).strip()
                        polishing_applied = True
                        
                    except Exception as e:
                        print(f"Warning: Segment polishing failed: {e}")
                
                processing_time = time.time() - start_time
                
                # Build result
                result = {
                    "success": True,
                    "metadata": {
                        "audio_length": round(info.duration, 2),
                        "language": info.language,
                        "processing_time": round(processing_time, 2),
                        "diarization_enabled": enable_diarization and diarization is not None,
                        "polished": polishing_applied
                    },
                    "full_transcript": polished_transcript,
                    "segments": segment_list
                }
                
                # Update job as completed
                update_job(job_id, {
                    "status": "completed",
                    "result": result,
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
            update_job(job_id, {
                "status": "failed",
                "error": str(e),
                "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            })
