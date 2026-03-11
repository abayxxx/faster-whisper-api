import time
import threading
from typing import Dict, Any
from app.core.config import settings

jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()


def cleanup_old_jobs():
    """Remove expired jobs from storage"""
    with jobs_lock:
        current_time = time.time()
        expired_jobs = [
            job_id for job_id, job_data in jobs.items()
            if current_time - job_data.get("created_at_ts", 0) > settings.JOB_EXPIRY_SECONDS
        ]
        for job_id in expired_jobs:
            del jobs[job_id]
        if expired_jobs:
            print(f"Cleaned up {len(expired_jobs)} expired jobs")


def get_job(job_id: str) -> Dict[str, Any]:
    """Get job data by ID"""
    with jobs_lock:
        return jobs.get(job_id)


def create_job(job_id: str, input_type: str) -> Dict[str, Any]:
    """Create new job"""
    job_data = {
        "job_id": job_id,
        "status": "pending",
        "input_type": input_type,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "created_at_ts": time.time()
    }
    with jobs_lock:
        jobs[job_id] = job_data
    return job_data


def update_job(job_id: str, updates: Dict[str, Any]):
    """Update job data"""
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(updates)
