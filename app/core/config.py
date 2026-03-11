import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Model Configuration
    MODEL_SIZE: str = "small"
    ENABLE_DIARIZATION: bool = False
    CPU_THREADS: int = 4
    HF_TOKEN: Optional[str] = None
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # Protection Limits
    MAX_CONCURRENT_REQUESTS: int = 3
    REQUEST_TIMEOUT: int = 600
    RATE_LIMIT: str = "10/minute"
    GLOBAL_RATE_LIMIT: str = "50/minute"
    MAX_FILE_SIZE_MB: int = 100
    JOB_EXPIRY_SECONDS: int = 3600
    
    # Security
    API_KEY: Optional[str] = None
    
    # AWS S3 Configuration
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    
    # Gemini AI Configuration
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"
    
    @property
    def MAX_FILE_SIZE(self) -> int:
        """Convert MB to bytes"""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024
    
    @property
    def is_auth_enabled(self) -> bool:
        """Check if API key authentication is enabled"""
        return self.API_KEY is not None
    
    @property
    def is_diarization_available(self) -> bool:
        """Check if diarization can be enabled"""
        return self.ENABLE_DIARIZATION and self.HF_TOKEN is not None
    
    @property
    def is_gemini_available(self) -> bool:
        """Check if Gemini AI is configured"""
        return self.GEMINI_API_KEY is not None
    
    @property
    def is_s3_available(self) -> bool:
        """Check if S3 is configured"""
        return self.AWS_ACCESS_KEY_ID is not None and self.AWS_SECRET_ACCESS_KEY is not None
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
