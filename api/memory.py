# API endpoint for memory management
from fastapi import APIRouter

router = APIRouter()

@router.get("/memory")
async def get_memory():
    """Retrieves current memory state via the API."""
    return {"memory": "current memory content"}
