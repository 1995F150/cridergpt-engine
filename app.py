"""CriderGPT Engine ASGI application."""

from __future__ import annotations

import logging

import requests
from fastapi import FastAPI

from api.chat import router as chat_router
from api.image import router as image_router
from api.usage import router as usage_router
from api.video import router as video_router
from config import settings
from memory.memory_store import get_supabase

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="CriderGPT Engine", version="3.2.0")

for router, tag in ((chat_router, "chat"), (image_router, "image"), (video_router, "video"), (usage_router, "usage")):
    app.include_router(router, tags=[tag])
    app.include_router(router, prefix="/api", tags=[tag], include_in_schema=False)


@app.get("/")
async def root():
    return {"service": "cridergpt-engine", "status": "running", "version": app.version}


@app.get("/health")
async def health():
    ollama_ok = False
    local_video_ok = False
    try:
        ollama_ok = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=2).ok
    except requests.RequestException:
        pass
    if settings.video_backend == "local":
        try:
            local_video_ok = requests.get(f"{settings.local_video_url.rstrip('/')}/system_stats", timeout=2).ok
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
            "local_video_model": local_video_ok,
            "local_tts": bool(settings.local_tts_url),
            "local_sound_effects": bool(settings.local_sfx_url),
            "local_music": bool(settings.local_music_url),
        },
        "capabilities": [
            "text_chat", "core_profile", "project_knowledge", "structured_memory",
            "semantic_rag", "preference_learning", "automatic_memory_writing",
            "planning", "agent_routing", "tool_orchestration", "self_evaluation",
            "writing_style", "image_generate", "image_analyze", "video_generate",
            "video_audio_generation", "video_job_status", "video_cancel",
            "unified_token_estimation", "usage_accounting",
        ],
    }
