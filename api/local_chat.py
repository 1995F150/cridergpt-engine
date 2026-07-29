"""Built-in local browser chat with persistent per-user memory."""

from __future__ import annotations

import html
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from engine.agent import get_agent_response
from engine.inference import InferenceUnavailable
from memory.local_chat_store import (
    append_message,
    get_profile,
    list_messages,
    update_profile,
)

router = APIRouter()


class LocalChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    user_id: str = Field(min_length=1, max_length=120)
    display_name: str | None = Field(default=None, max_length=120)
    conversation_id: str | None = Field(default=None, max_length=120)


class ProfileRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=120)
    display_name: str | None = Field(default=None, max_length=120)
    profile: dict[str, Any] = Field(default_factory=dict)


@router.get("/chat", response_class=HTMLResponse, include_in_schema=False)
async def local_chat_page() -> HTMLResponse:
    return HTMLResponse(LOCAL_CHAT_HTML)


@router.get("/local-chat/history")
async def local_chat_history(user_id: str, conversation_id: str = "default"):
    return {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "profile": get_profile(user_id),
        "messages": list_messages(user_id, conversation_id),
    }


@router.post("/local-chat/profile")
async def local_chat_profile(request: ProfileRequest):
    return update_profile(request.user_id, request.display_name, request.profile)


@router.post("/local-chat/send")
async def local_chat_send(request: LocalChatRequest):
    conversation_id = request.conversation_id or "default"
    profile = update_profile(request.user_id, request.display_name, {})
    history = list_messages(request.user_id, conversation_id, limit=30)
    profile_context = (
        "LOCAL CHAT USER PROFILE:\n"
        f"Display name: {profile['display_name']}\n"
        f"Known profile data: {profile['profile']}\n"
        "Use this only to personalize the reply for this user."
    )
    append_message(request.user_id, conversation_id, "user", request.message)
    try:
        result = get_agent_response(
            request.message,
            user_id=f"local:{request.user_id}",
            conversation_id=f"local:{conversation_id}",
            system_prompt=profile_context,
            conversation_history=history,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InferenceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    append_message(request.user_id, conversation_id, "assistant", result.response)
    return {
        "response": result.response,
        "conversation_id": conversation_id,
        "model": result.model,
        "memories_used": result.memories_used,
        "latency_ms": result.latency_ms,
    }


LOCAL_CHAT_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CriderGPT Local Chat</title>
<style>
:root{color-scheme:dark;--bg:#07101d;--panel:#101b2c;--line:#26364f;--text:#eef5ff;--muted:#9eb0c8;--accent:#65a7ff;--user:#17467a;--ai:#17243a}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#07101d,#0d1930);color:var(--text);font-family:Inter,system-ui,Arial}
main{max-width:980px;margin:auto;padding:18px}.top{display:flex;gap:12px;justify-content:space-between;align-items:center;flex-wrap:wrap}.top h1{margin:0}.top a{color:#9bc5ff}
.settings,.composer{display:flex;gap:8px;flex-wrap:wrap;background:rgba(16,27,44,.95);border:1px solid var(--line);border-radius:14px;padding:12px;margin-top:14px}
input,textarea,button{font:inherit;border-radius:10px;border:1px solid var(--line)}input,textarea{background:#071321;color:var(--text);padding:10px}input{flex:1;min-width:180px}textarea{width:100%;min-height:90px;resize:vertical}button{background:var(--accent);color:#06101e;font-weight:700;padding:10px 16px;cursor:pointer}.secondary{background:#182840;color:var(--text)}
#messages{display:flex;flex-direction:column;gap:10px;margin-top:16px;min-height:52vh}.bubble{max-width:84%;padding:12px 14px;border-radius:14px;white-space:pre-wrap;line-height:1.45}.user{align-self:flex-end;background:var(--user)}.assistant{align-self:flex-start;background:var(--ai);border:1px solid var(--line)}.meta{font-size:12px;color:var(--muted);margin-top:4px}.empty{color:var(--muted);text-align:center;margin-top:60px}
</style>
</head><body><main>
<div class="top"><div><h1>CriderGPT Local Chat</h1><div style="color:var(--muted)">Talk directly to the engine at <code>/chat</code>. Messages are stored locally per user.</div></div><div><a href="/dashboard">Dashboard</a> · <a href="/docs">API Docs</a></div></div>
<div class="settings"><input id="userId" placeholder="User ID" value="owner"><input id="displayName" placeholder="Display name" value="Jessie"><input id="conversationId" placeholder="Conversation ID" value="default"><button class="secondary" onclick="loadHistory()">Load</button></div>
<div id="messages"><div class="empty">Start a conversation.</div></div>
<div class="composer"><textarea id="message" placeholder="Ask the local AI anything..."></textarea><button onclick="sendMessage()">Send</button></div>
</main><script>
const esc=s=>String(s).replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[c]));
function ids(){return{user_id:document.getElementById('userId').value.trim()||'owner',display_name:document.getElementById('displayName').value.trim()||null,conversation_id:document.getElementById('conversationId').value.trim()||'default'}}
function render(messages){const box=document.getElementById('messages');box.innerHTML=messages.length?'':'<div class="empty">Start a conversation.</div>';for(const m of messages){const d=document.createElement('div');d.className='bubble '+m.role;d.innerHTML=esc(m.content)+'<div class="meta">'+esc(m.role)+'</div>';box.appendChild(d)}window.scrollTo(0,document.body.scrollHeight)}
async function loadHistory(){const x=ids();const r=await fetch('/local-chat/history?user_id='+encodeURIComponent(x.user_id)+'&conversation_id='+encodeURIComponent(x.conversation_id));const d=await r.json();render(d.messages||[])}
async function sendMessage(){const input=document.getElementById('message');const text=input.value.trim();if(!text)return;const x=ids();input.value='';const old=[...document.querySelectorAll('.bubble')].map(n=>({role:n.classList.contains('user')?'user':'assistant',content:n.childNodes[0].textContent}));render([...old,{role:'user',content:text},{role:'assistant',content:'Thinking...'}]);try{const r=await fetch('/local-chat/send',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...x,message:text})});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Request failed');await loadHistory()}catch(e){render([...old,{role:'user',content:text},{role:'assistant',content:'Error: '+e.message}])}}
document.getElementById('message').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage()}});loadHistory();
</script></body></html>"""
