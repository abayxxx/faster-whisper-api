from typing import List, Dict, Optional
from google import genai
from app.core.config import settings


class GeminiService:
    """Gemini AI service for transcript polishing and summarization"""
    
    def __init__(self):
        self.client = None
        if settings.is_gemini_available:
            self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
            print("Gemini AI client initialized successfully")
        else:
            print("Warning: GEMINI_API_KEY not set. Transcript polishing and summarization will be disabled.")
    
    def is_available(self) -> bool:
        """Check if Gemini service is available"""
        return self.client is not None
    
    def polish_segments_batch(self, segments_text: List[str], language: str = "id") -> List[str]:
        """Polish multiple segments in a single API call"""
        if not self.is_available() or not segments_text:
            return segments_text
        
        language_instruction = self._get_language_instruction(language)
        numbered_segments = "\n".join([f"{i+1}. {text}" for i, text in enumerate(segments_text)])
        
        prompt = f"""Polish these call transcript segments. For each segment:
- Fix grammar, punctuation, capitalization
- Remove filler words (um, uh, hmm, you know, like, gitu, etc.)
- Correct misheard words
- Keep the meaning exact{language_instruction}

Return ONLY the polished segments in the same numbered format, one per line. Do not add explanations.

Segments:
{numbered_segments}"""
        
        try:
            response = self.client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt
            )
            result_text = response.text.strip()
            
            # Parse numbered response
            polished_segments = []
            for line in result_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('*')):
                    text = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
                    polished_segments.append(text)
            
            # Ensure same number of segments
            if len(polished_segments) != len(segments_text):
                print(f"Warning: Segment count mismatch. Expected {len(segments_text)}, got {len(polished_segments)}")
                return segments_text
            
            return polished_segments
        
        except Exception as e:
            error_msg = str(e).lower()
            
            if any(keyword in error_msg for keyword in ["quota", "rate limit", "429", "resource"]):
                raise Exception(f"Gemini API quota exceeded. Please try again later or check your API limits. Error: {e}")
            elif any(keyword in error_msg for keyword in ["api key", "unauthorized", "401", "403"]):
                raise Exception(f"Gemini API authentication failed. Please check your API key. Error: {e}")
            else:
                print(f"Warning: Batch segment polishing failed: {e}")
                return segments_text
    
    def summarize_text(self, text: str, language: str = "id") -> Dict[str, str]:
        """Summarize text and generate next steps suggestion"""
        if not self.is_available():
            raise Exception("Gemini API not configured. Set GEMINI_API_KEY in .env")
        
        language_instruction = self._get_language_instruction(language)
        
        prompt = f"""Analyze the following transcript and provide:

1. A clear, concise summary paragraph focusing on main topics and key points
2. A suggestion paragraph for recommended next steps or actions based on the conversation{language_instruction}

Format your response EXACTLY as follows:

SUMMARY:
[Your summary paragraph here]

NEXT STEPS SUGGESTION:
[Your next steps suggestion paragraph here]

Transcript:
{text}"""
        
        try:
            response = self.client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt
            )
            result_text = response.text.strip()
            
            # Parse response
            summary = ""
            next_steps_suggestion = ""
            
            if "SUMMARY:" in result_text and "NEXT STEPS SUGGESTION:" in result_text:
                parts = result_text.split("NEXT STEPS SUGGESTION:")
                summary = parts[0].replace("SUMMARY:", "").strip()
                next_steps_suggestion = parts[1].strip()
            else:
                summary = result_text
                next_steps_suggestion = "Please review the transcript for further action items."
            
            return {
                "summary": summary,
                "next_steps_suggestion": next_steps_suggestion
            }
        
        except Exception as e:
            error_msg = str(e).lower()
            
            if any(keyword in error_msg for keyword in ["quota", "rate limit", "429", "resource"]):
                raise Exception(f"Gemini API quota exceeded. Please try again later or check your API limits. Error: {e}")
            elif any(keyword in error_msg for keyword in ["api key", "unauthorized", "401", "403"]):
                raise Exception(f"Gemini API authentication failed. Please check your API key. Error: {e}")
            else:
                raise Exception(f"Gemini API error: {str(e)}")
    
    def _get_language_instruction(self, language: str) -> str:
        """Get language-specific instruction for prompts"""
        if language == "id":
            return "\n\nIMPORTANT: Output MUST be in Indonesian (Bahasa Indonesia)."
        elif language == "en":
            return "\n\nIMPORTANT: Output MUST be in English."
        else:
            return f"\n\nIMPORTANT: Output MUST be in {language}."


gemini_service = GeminiService()
