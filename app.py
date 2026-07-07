import uuid
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.chat import router as chat_router
from api.image import router as image_router
from memory.memory_exporter import export_memory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CriderGPT Engine API")

# Include routers
app.include_router(chat_router, prefix="/api", tags=["chat"])
app.include_router(image_router, prefix="/api", tags=["image"])

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "capabilities": ["text_chat", "memory_sync", "image_generate", "image_recognize"]
    }

@app.get("/api/memory/export")
async def export_mem():
    return export_memory()
