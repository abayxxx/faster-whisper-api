from fastapi import APIRouter, Request, HTTPException, Depends
from app.models import JobStatusResponse
from app.core.security import limiter, verify_api_key
from app.utils import get_job

router = APIRouter()


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
@limiter.limit("60/minute")
async def get_job_status(
    request: Request,
    job_id: str,
    _: bool = Depends(verify_api_key)
):
    """
    Get status and result of a job
    
    Status values:
    - **pending**: Job created, waiting to start
    - **processing**: Job is being processed (check 'progress' field)
    - **completed**: Job finished successfully (see 'result' field)
    - **failed**: Job failed (see 'error' field)
    
    Poll this endpoint every 3-5 seconds until status is 'completed' or 'failed'
    """
    job_data = get_job(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    
    return JobStatusResponse(**job_data)
