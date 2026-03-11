from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address
from .config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> bool:
    """Verify API key if authentication is enabled"""
    if not settings.is_auth_enabled:
        return True
    
    if api_key is None or api_key != settings.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    return True


limiter = Limiter(key_func=get_remote_address)
