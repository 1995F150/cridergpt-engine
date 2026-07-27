"""CriderGPT Engine ASGI application."""

from __future__ import annotations

import logging

import requests
from fastapi import FastAPI

from api.chat import router as chat_router
from api.image import router as image_router
from api.video import router as video_router
from config import settings
from memory.memory_store import get_supabase

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="CriderGPT Engine", version="3.1.0")

app.include_router(chat_router, tags=["chat"])
app.include_router(image_router, tags=["image"])
app.include_router(video_router, tags=["video"])
app.include_router(chat_router, prefix="/api", tags=["chat"], include_in_schema=False)
app.include_router(image_router, prefix="/api", tags=["image"], include_in_schema=False)
app.include_router(video_router, prefix="/api", tags=["video"], include_in_schema=False)


@app.get("/")
async def root():
    return {"service": "cridergpt-engine", "status": "running", "version": app.version}


@app.get("/health")
async def health():
    ollama_ok = False
    try:
        response = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=2)
        ollama_ok = response.ok
    except requests.RequestException:
        pass

    supabase_configured = get_supabase() is not None
    ready = ollama_ok and bool(settings.api_keys)
    return {
        "status": "healthy" if ready else "degraded",
        "ready": ready,
        "dependencies": {
            "ollama": ollama_ok,
            "supabase": supabase_configured,
            "authentication": bool(settings.api_keys),
            "image_backend": settings.image_backend,
            "embedding_model": settings.embedding_model,
            "video_backend": settings.video_backend,
            "video_provider_configured": bool(settings.video_api_url),
        },
        "capabilities": [
            "text_chat",
            "core_profile",
            "project_knowledge",
            "structured_memory",
            "semantic_rag",
            "preference_learning",
            "automatic_memory_writing",
            "planning",
            "agent_routing",
            "tool_orchestration",
            "self_evaluation",
            "writing_style",
            "image_generate",
            "image_analyze",
            "video_generate",
            "video_job_status",
            "video_cancel",
            "video_webhooks",
        ],
    }
