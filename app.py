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
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from api.auth import authenticate_engine_key, validate_api_key
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
app = FastAPI(title="CriderGPT Engine", version="3.4.0")
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
        "local_url": f"http://{settings.host}:{settings.port}",
        "public_url": os.getenv("CRIDERGPT_ENGINE_PUBLIC_URL", "https://cridergpt.com/engine/api"),
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
            "authentication": {
                "configured": bool(settings.api_keys),
                "accepted_methods": ["Bearer", "X-API-Key"],
                "configured_key_count": len(settings.api_keys),
            },
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
            "dashboard": True,
            "dual_authentication": True,
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
    return {
        "service": "cridergpt-engine",
        "status": "running",
        "version": app.version,
        "dashboard": "/dashboard",
        "docs": "/docs",
    }


@app.get("/health")
@app.get("/api/health", include_in_schema=False)
async def health():
    return _service_health()


@app.get("/auth/check")
@app.get("/api/auth/check", include_in_schema=False)
async def auth_check(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    auth = authenticate_engine_key(authorization, x_api_key)
    return {
        "authenticated": True,
        "authentication": auth.method,
        "configured": bool(settings.api_keys),
        "engine_version": app.version,
        "service": "cridergpt-engine",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/config/reload")
@app.post("/api/config/reload", include_in_schema=False)
async def reload_config(
    request: ConfigReloadRequest,
    _api_key: str = Depends(validate_api_key),
):
    """Acknowledge the latest Supabase control-plane configuration."""
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


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CriderGPT Engine Dashboard</title>
<style>
:root{color-scheme:dark;--bg:#09111f;--panel:#111c2e;--line:#26364f;--text:#edf4ff;--muted:#9fb0c9;--good:#37d67a;--bad:#ff6b6b;--accent:#65a7ff}
*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,Arial;background:linear-gradient(135deg,#07101d,#0d1930);color:var(--text)}
main{max-width:1180px;margin:auto;padding:28px}.top{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap}.top h1{margin:0;font-size:30px}.muted{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;margin-top:20px}.card{background:rgba(17,28,46,.94);border:1px solid var(--line);border-radius:16px;padding:18px;box-shadow:0 12px 35px rgba(0,0,0,.2)}
.card h2{font-size:15px;color:var(--muted);margin:0 0 10px}.value{font-size:22px;font-weight:700;word-break:break-word}.status{display:inline-flex;align-items:center;gap:8px}.dot{width:10px;height:10px;border-radius:99px;background:var(--bad)}.dot.good{background:var(--good)}
section{margin-top:18px}.row{display:flex;gap:10px;flex-wrap:wrap}input,textarea,button{font:inherit;border-radius:10px;border:1px solid var(--line)}input,textarea{background:#081322;color:var(--text);padding:11px;flex:1;min-width:220px}button{background:var(--accent);color:#06101e;padding:11px 16px;font-weight:700;cursor:pointer}button.secondary{background:#182840;color:var(--text)}pre{background:#07101d;border:1px solid var(--line);padding:14px;border-radius:12px;overflow:auto;min-height:80px;white-space:pre-wrap}.links a{color:#8fc1ff;margin-right:16px}.wide{grid-column:1/-1}
</style>
</head>
<body><main>
<div class="top"><div><h1>CriderGPT Engine</h1><div class="muted">Local control and diagnostics dashboard</div></div><div class="links"><a href="/docs">API Docs</a><a href="/api/health">Raw Health</a></div></div>
<div class="grid">
<div class="card"><h2>Engine status</h2><div class="value status"><span id="statusDot" class="dot"></span><span id="status">Loading</span></div></div>
<div class="card"><h2>Version</h2><div id="version" class="value">—</div></div>
<div class="card"><h2>Active model</h2><div id="model" class="value">—</div></div>
<div class="card"><h2>Uptime</h2><div id="uptime" class="value">—</div></div>
<div class="card"><h2>Ollama</h2><div id="ollama" class="value">—</div></div>
<div class="card"><h2>Supabase</h2><div id="supabase" class="value">—</div></div>
<div class="card"><h2>Authentication</h2><div id="authConfigured" class="value">—</div></div>
<div class="card"><h2>Public URL</h2><div id="publicUrl" class="value">—</div></div>
</div>
<section class="card"><h2>API authentication test</h2><div class="row"><input id="apiKey" type="password" placeholder="Paste engine API key"><button onclick="testAuth('Bearer')">Test Bearer</button><button class="secondary" onclick="testAuth('X-API-Key')">Test X-API-Key</button></div><pre id="authResult">Enter the configured engine key to test authentication.</pre></section>
<section class="card"><h2>Configuration control</h2><div class="row"><input id="configVersion" type="number" placeholder="Config version (leave blank for latest)"><button onclick="reloadConfig()">Reload configuration</button><button class="secondary" onclick="loadHealth()">Refresh health</button></div><pre id="controlResult">No action run yet.</pre></section>
<section class="card"><h2>Endpoint reference</h2><pre>Health: GET /api/health\nAuthentication: GET /api/auth/check\nChat: POST /api/chat\nReload config: POST /api/config/reload\nInteractive API: GET /docs</pre></section>
</main>
<script>
const pretty=v=>JSON.stringify(v,null,2);const keyHeaders=(method)=>{const key=document.getElementById('apiKey').value.trim();return method==='Bearer'?{'Authorization':'Bearer '+key}:{'X-API-Key':key}};
function formatUptime(s){s=Number(s||0);const d=Math.floor(s/86400),h=Math.floor((s%86400)/3600),m=Math.floor((s%3600)/60);return `${d}d ${h}h ${m}m`}
async function loadHealth(){try{const r=await fetch('/api/health');const d=await r.json();document.getElementById('status').textContent=d.status;document.getElementById('statusDot').className='dot '+(d.ready?'good':'');document.getElementById('version').textContent=d.engine_version;document.getElementById('model').textContent=d.active_model;document.getElementById('uptime').textContent=formatUptime(d.uptime_seconds);document.getElementById('ollama').textContent=d.dependencies.ollama?'Online':'Offline';document.getElementById('supabase').textContent=d.dependencies.supabase?'Configured':'Not configured';document.getElementById('authConfigured').textContent=d.dependencies.authentication?'Configured':'Not configured';document.getElementById('publicUrl').textContent=d.public_url||'—'}catch(e){document.getElementById('status').textContent='Unavailable'}}
async function testAuth(method){const out=document.getElementById('authResult');try{const r=await fetch('/api/auth/check',{headers:keyHeaders(method)});const text=await r.text();let data;try{data=JSON.parse(text)}catch{data=text}out.textContent=pretty({status:r.status,response:data})}catch(e){out.textContent=String(e)}}
async function reloadConfig(){const out=document.getElementById('controlResult');const value=document.getElementById('configVersion').value;const body=value?{config_version:Number(value)}:{};try{const r=await fetch('/api/config/reload',{method:'POST',headers:{'Content-Type':'application/json',...keyHeaders('Bearer')},body:JSON.stringify(body)});const text=await r.text();let data;try{data=JSON.parse(text)}catch{data=text}out.textContent=pretty({status:r.status,response:data});loadHealth()}catch(e){out.textContent=String(e)}}
loadHealth();setInterval(loadHealth,30000);
</script></body></html>"""


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)
