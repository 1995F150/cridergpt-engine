"""CriderGPT Engine ASGI application."""

from __future__ import annotations

import logging
import os
import platform
import socket
import time
from datetime import datetime, timezone
from typing import Any

import requests
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from api.auth import validate_api_key
from api.chat import router as chat_router
from api.image import router as image_router
from api.usage import router as usage_router
from api.video import router as video_router
from config import settings
from memory.memory_store import get_supabase
from usage.middleware import UsageMeterMiddleware

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

STARTED_AT = datetime.now(timezone.utc)
app = FastAPI(title="CriderGPT Engine", version="3.3.0")
app.add_middleware(UsageMeterMiddleware)

for router, tag in ((chat_router, "chat"), (image_router, "image"), (video_router, "video"), (usage_router, "usage")):
    app.include_router(router, tags=[tag])
    app.include_router(router, prefix="/api", tags=[tag], include_in_schema=False)


class ConfigReloadRequest(BaseModel):
    config_version: int | None = None


def _service_health() -> dict[str, Any]:
    ollama_ok = False
    local_video_ok = False
    try:
        ollama_ok = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=2).ok
    except requests.RequestException:
        pass
    if settings.video_backend == "local":
        try:
            local_video_ok = requests.get(
                f"{settings.local_video_url.rstrip('/')}/system_stats", timeout=2
            ).ok
        except requests.RequestException:
            pass

    supabase_configured = get_supabase() is not None
    ready = ollama_ok and bool(settings.api_keys)
    return {
        "status": "online" if ready else "degraded",
        "ready": ready,
        "service": "cridergpt-engine",
        "engine_version": app.version,
        "version": app.version,
        "git_sha": os.getenv("CRIDERGPT_ENGINE_GIT_SHA") or os.getenv("GIT_SHA"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "started_at": STARTED_AT.isoformat(),
        "uptime_seconds": int((datetime.now(timezone.utc) - STARTED_AT).total_seconds()),
        "active_model": settings.ollama_model,
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
        "services": {
            "ollama": {"online": ollama_ok, "model": settings.ollama_model},
            "supabase": {"configured": supabase_configured},
            "image": {"backend": settings.image_backend},
            "video": {"backend": settings.video_backend, "online": local_video_ok},
            "narration": {"configured": bool(settings.local_tts_url)},
            "sound_effects": {"configured": bool(settings.local_sfx_url)},
            "music": {"configured": bool(settings.local_music_url)},
        },
        "capabilities": {
            "chat": True,
            "image_generation": True,
            "image_analysis": True,
            "video_generation": True,
            "rag": True,
            "structured_memory": True,
            "planning": True,
            "tool_orchestration": True,
            "usage_accounting": True,
        },
    }


def _acknowledge_config(config_version: int) -> None:
    client = get_supabase()
    if client is None:
        return
    health = _service_health()
    payload = {
        "engine_id": "primary",
        "online": bool(health["ready"]),
        "status": health["status"],
        "base_url": os.getenv("CRIDERGPT_ENGINE_PUBLIC_URL", "https://cridergpt.com/engine/api"),
        "engine_version": app.version,
        "git_sha": health.get("git_sha"),
        "hostname": health["hostname"],
        "started_at": health["started_at"],
        "last_heartbeat": datetime.now(timezone.utc).isoformat(),
        "last_health_check": datetime.now(timezone.utc).isoformat(),
        "active_model": settings.ollama_model,
        "config_version": config_version,
        "ack_config_version": config_version,
        "capabilities": health["capabilities"],
        "services": health["services"],
        "last_error": None,
        "metadata": {"reload_source": "api", "pid": os.getpid()},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        client.table("engine_runtime_status").upsert(payload).execute()
    except Exception:
        logger.exception("Could not acknowledge infrastructure configuration")


@app.get("/")
async def root():
    return {"service": "cridergpt-engine", "status": "running", "version": app.version}


@app.get("/health")
@app.get("/api/health", include_in_schema=False)
async def health():
    return _service_health()


@app.post("/config/reload")
@app.post("/api/config/reload", include_in_schema=False)
async def reload_config(
    request: ConfigReloadRequest,
    _api_key: str = Depends(validate_api_key),
):
    """Acknowledge the latest Supabase control-plane configuration.

    Runtime settings are read by request handlers from Supabase where supported;
    immutable machine addresses and credentials remain in the protected .env file.
    """
    config_version = request.config_version
    if config_version is None:
        client = get_supabase()
        if client is None:
            raise HTTPException(status_code=503, detail="Supabase is not configured")
        try:
            result = (
                client.table("ai_infrastructure_settings")
                .select("config_version")
                .order("updated_at", desc=True)
                .limit(1)
                .execute()
            )
            rows = result.data or []
            config_version = int(rows[0]["config_version"]) if rows else 0
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Could not load infrastructure configuration") from exc

    _acknowledge_config(int(config_version))
    return {
        "status": "online",
        "reloaded": True,
        "config_version": int(config_version),
        "ack_config_version": int(config_version),
        "engine_version": app.version,
        "active_model": settings.ollama_model,
        "timestamp": time.time(),
    }
