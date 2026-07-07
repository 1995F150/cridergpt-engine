import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from engine.agent import get_agent_response
from memory.memory_exporter import export_memory
from api.image import router as image_router
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CriderGPT Engine API")

# Include routers
app.include_router(image_router)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "capabilities": ["text_chat", "memory_sync", "image_generate", "image_recognize"]
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    try:
        response = get_agent_response(request.message, session_id)
        return {
            "response": response,
            "session_id": session_id
        }
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/sync")
async def sync_memory(session_id: str):
    try:
        export_memory(session_id)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error in memory sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))
