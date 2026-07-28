"""Private-network API key management routes and dashboard page."""

from __future__ import annotations

import ipaddress
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from api.key_store import create_key, list_keys, revoke_key
from config import settings

router = APIRouter()


class CreateKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    model: str = Field(default="*", max_length=120)
    max_tokens: int = Field(default=4096, ge=1, le=131072)
    requests_per_minute: int = Field(default=60, ge=1, le=10000)
    expires_in_days: int | None = Field(default=None, ge=1, le=3650)


def _private_dashboard_only(request: Request) -> None:
    host = request.client.host if request.client else ""
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Key management is available only on the private dashboard") from exc

    tailscale = address.version == 4 and address in ipaddress.ip_network("100.64.0.0/10")
    if not (address.is_loopback or address.is_private or tailscale):
        raise HTTPException(status_code=403, detail="Key management is available only on the private dashboard")


@router.get("/api/keys")
async def get_keys(request: Request) -> dict[str, Any]:
    _private_dashboard_only(request)
    return {"keys": list_keys()}


@router.post("/api/keys", status_code=status.HTTP_201_CREATED)
async def generate_key(payload: CreateKeyRequest, request: Request) -> dict[str, Any]:
    _private_dashboard_only(request)
    result = create_key(
        name=payload.name,
        model=payload.model,
        max_tokens=payload.max_tokens,
        requests_per_minute=payload.requests_per_minute,
        expires_in_days=payload.expires_in_days,
    )
    return {
        "message": "API key generated. Copy it now; the full key will not be shown again.",
        "key": result,
    }


@router.delete("/api/keys/{key_id}")
async def disable_key(key_id: str, request: Request) -> dict[str, Any]:
    _private_dashboard_only(request)
    if not revoke_key(key_id):
        raise HTTPException(status_code=404, detail="Active API key not found")
    return {"revoked": True, "id": key_id}


KEY_MANAGER_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CriderGPT API Keys</title>
<style>
:root{color-scheme:dark;--bg:#08111f;--panel:#111d30;--line:#293a55;--text:#eef5ff;--muted:#9fb0c8;--accent:#67a9ff;--danger:#ff7777}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#07101d,#0d1930);font-family:system-ui,Arial;color:var(--text)}main{max-width:1050px;margin:auto;padding:24px}.card{background:rgba(17,29,48,.96);border:1px solid var(--line);border-radius:16px;padding:18px;margin:16px 0}.row{display:flex;gap:10px;flex-wrap:wrap}input,select,button{font:inherit;border-radius:10px;border:1px solid var(--line);padding:11px}input,select{background:#071322;color:var(--text);flex:1;min-width:150px}button{background:var(--accent);color:#07101d;font-weight:750;cursor:pointer}.danger{background:var(--danger)}.muted{color:var(--muted)}pre{background:#06101c;border:1px solid var(--line);border-radius:12px;padding:14px;white-space:pre-wrap;overflow:auto}.key{display:grid;grid-template-columns:1fr auto;gap:12px;border-top:1px solid var(--line);padding:14px 0}.key:first-child{border-top:0}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:4px 8px;margin:3px;font-size:12px}a{color:#95c5ff}
</style></head><body><main>
<h1>API Key Manager</h1><p class="muted">Generate Bearer tokens and legacy X-API-Key credentials. Full keys are displayed only once.</p><p><a href="/dashboard">← Engine dashboard</a></p>
<section class="card"><h2>Generate API key</h2><div class="row"><input id="name" placeholder="Key name, e.g. Supabase chat-with-ai"><input id="model" value="*" placeholder="Allowed model or *"></div><div class="row" style="margin-top:10px"><input id="tokens" type="number" value="4096" min="1" placeholder="Maximum tokens"><input id="rpm" type="number" value="60" min="1" placeholder="Requests per minute"><input id="days" type="number" min="1" placeholder="Expires in days (optional)"><button onclick="generateKey()">Generate key</button></div><pre id="created">No key generated yet.</pre></section>
<section class="card"><div class="row" style="justify-content:space-between;align-items:center"><h2>Existing keys</h2><button onclick="loadKeys()">Refresh</button></div><div id="keys">Loading…</div></section>
<section class="card"><h2>How to use a generated key</h2><pre>Authorization: Bearer cgpt_your_generated_key

Legacy compatibility:
X-API-Key: cgpt_your_generated_key</pre></section>
<script>
const pretty=v=>JSON.stringify(v,null,2);const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function generateKey(){const out=document.getElementById('created');const body={name:document.getElementById('name').value.trim(),model:document.getElementById('model').value.trim()||'*',max_tokens:Number(document.getElementById('tokens').value||4096),requests_per_minute:Number(document.getElementById('rpm').value||60)};const days=document.getElementById('days').value;if(days)body.expires_in_days=Number(days);if(!body.name){out.textContent='Enter a name for the key.';return}try{const r=await fetch('/api/keys',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const d=await r.json();out.textContent=pretty(d);if(d.key?.api_key){document.getElementById('apiKeyCopy')?.remove();const b=document.createElement('button');b.id='apiKeyCopy';b.textContent='Copy generated key';b.onclick=()=>navigator.clipboard.writeText(d.key.api_key);out.after(b)}loadKeys()}catch(e){out.textContent=String(e)}}
async function revoke(id){if(!confirm('Revoke this API key? Applications using it will stop working.'))return;await fetch('/api/keys/'+encodeURIComponent(id),{method:'DELETE'});loadKeys()}
async function loadKeys(){const box=document.getElementById('keys');try{const r=await fetch('/api/keys');const d=await r.json();if(!r.ok){box.innerHTML='<pre>'+esc(pretty(d))+'</pre>';return}box.innerHTML=d.keys.length?d.keys.map(k=>`<div class="key"><div><strong>${esc(k.name)}</strong><div class="muted">${esc(k.prefix)}… · Created ${esc(k.created_at)}</div><div><span class="pill">Model: ${esc(k.model)}</span><span class="pill">Max tokens: ${esc(k.max_tokens)}</span><span class="pill">RPM: ${esc(k.requests_per_minute)}</span><span class="pill">${k.revoked?'Revoked':'Active'}</span></div></div><button class="danger" ${k.revoked?'disabled':''} onclick="revoke('${esc(k.id)}')">Revoke</button></div>`).join(''):'No generated keys yet.'}catch(e){box.textContent=String(e)}}loadKeys();
</script></main></body></html>"""


@router.get("/dashboard/keys", response_class=HTMLResponse, include_in_schema=False)
async def key_dashboard(request: Request) -> HTMLResponse:
    _private_dashboard_only(request)
    return HTMLResponse(KEY_MANAGER_HTML)
