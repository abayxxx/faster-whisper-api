import os
import boto3
import requests
from urllib.parse import urlparse
from fastapi import HTTPException
from typing import Tuple, Optional
from app.core.config import settings


class StorageService:
    """Handle file downloads from S3 and HTTP/HTTPS URLs"""
    
    def __init__(self):
        self.s3_client = None
        if settings.is_s3_available:
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                    region_name=settings.AWS_REGION
                )
                print("AWS S3 client initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize S3 client: {e}")
        else:
            print("Warning: AWS credentials not set. S3 URL support will be disabled.")
    
    def download_file_from_url(self, url: str, timeout: int = 300) -> Tuple[bytes, str]:
        """
        Download file from URL (S3 or HTTP/HTTPS)
        Returns: (file_content, filename)
        """
        parsed_url = urlparse(url)
        
        # Try S3 download if credentials configured
        if parsed_url.hostname and 's3' in parsed_url.hostname and self.s3_client:
            try:
                return self._download_from_s3(parsed_url)
            except Exception as e:
                print(f"S3 SDK download failed, falling back to HTTPS: {e}")
        
        # Fall back to HTTP/HTTPS download
        return self._download_from_http(url, timeout)
    
    def _download_from_s3(self, parsed_url) -> Tuple[bytes, str]:
        """Download file from S3 using boto3"""
        path_parts = parsed_url.path.lstrip('/').split('/', 1)
        
        if 's3.amazonaws.com' in parsed_url.hostname:
            bucket = parsed_url.hostname.split('.')[0]
            key = parsed_url.path.lstrip('/')
        elif parsed_url.hostname.startswith('s3'):
            bucket = path_parts[0]
            key = path_parts[1] if len(path_parts) > 1 else ''
        else:
            bucket = parsed_url.hostname.split('.')[0]
            key = parsed_url.path.lstrip('/')
        
        if not key:
            raise HTTPException(status_code=400, detail="Invalid S3 URL: missing object key")
        
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read()
        filename = os.path.basename(key)
        
        return content, filename
    
    def _download_from_http(self, url: str, timeout: int) -> Tuple[bytes, str]:
        """Download file from HTTP/HTTPS URL"""
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Check content length
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > settings.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_MB} MB"
                )
            
            # Download content
            content = b''
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > settings.MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_MB} MB"
                    )
            
            # Extract filename
            filename = None
            if 'content-disposition' in response.headers:
                content_disp = response.headers['content-disposition']
                if 'filename=' in content_disp:
                    filename = content_disp.split('filename=')[1].strip('"\'')
            
            if not filename:
                filename = os.path.basename(urlparse(url).path) or 'audio.wav'
            
            return content, filename
        
        except requests.exceptions.Timeout:
            raise HTTPException(status_code=504, detail="URL download timeout")
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Failed to download from URL: {str(e)}")


storage_service = StorageService()
