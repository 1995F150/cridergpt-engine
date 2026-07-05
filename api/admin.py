# API endpoint for administrative tasks
from fastapi import APIRouter

router = APIRouter()

@router.get("/admin/status")
async def get_status():
    """Returns the system status for administrators."""
    return {"status": "all systems operational"}
