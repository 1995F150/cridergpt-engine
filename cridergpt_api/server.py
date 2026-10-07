"""Production-oriented HTTP API for a single always-on CriderGPT model process."""
from __future__ import annotations
import asyncio,os,time,uuid
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI,Header,HTTPException,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel,Field
from cridergpt_stage5.inference import TransformersBackend

MODEL_NAME=os.getenv("CRIDERGPT_MODEL_NAME","cridergpt-2.1-nova")
CHECKPOINT=os.getenv("CRIDERGPT_CHECKPOINT","model/cridergpt-2.1-nova/checkpoint")
API_KEY=os.getenv("CRIDERGPT_API_KEY","")
MAX_INPUT_CHARS=int(os.getenv("CRIDERGPT_MAX_INPUT_CHARS","16000"))
MAX_NEW_TOKENS=int(os.getenv("CRIDERGPT_MAX_NEW_TOKENS","192"))
backend=None
generation_lock=asyncio.Lock()

class Message(BaseModel):
    role:Literal["system","user","assistant"]
    content:str=Field(min_length=1,max_length=MAX_INPUT_CHARS)
class ChatRequest(BaseModel):
    model:str|None=None
    messages:list[Message]=Field(min_length=1,max_length=64)
    stream:bool=False
    max_tokens:int|None=Field(default=None,ge=1,le=1024)
class GenerateRequest(BaseModel):
    prompt:str=Field(min_length=1,max_length=MAX_INPUT_CHARS)
    max_tokens:int|None=Field(default=None,ge=1,le=1024)

def require_key(authorization:str|None):
    if not API_KEY:return
    if authorization != f"Bearer {API_KEY}":raise HTTPException(401,"Invalid API key")

def prompt_from_messages(messages:list[Message])->str:
    # Current native runtime is single-turn. Preserve useful prior turns as readable context.
    parts=[]
    for m in messages:
        label={"system":"System","user":"User","assistant":"Assistant"}[m.role]
        parts.append(f"{label}: {m.content.strip()}")
    return "\n".join(parts)

async def infer(prompt:str,max_tokens:int|None=None)->str:
    global backend
    if backend is None:raise HTTPException(503,"Model is not loaded")
    async with generation_lock:
        old=backend.max_new_tokens
        if max_tokens is not None:backend.max_new_tokens=max_tokens
        try:return await asyncio.to_thread(backend.generate,prompt)
        finally:backend.max_new_tokens=old

@asynccontextmanager
async def lifespan(app:FastAPI):
    global backend
    from pathlib import Path
    checkpoint=Path(CHECKPOINT)
    if not checkpoint.exists():raise RuntimeError(f"Checkpoint not found: {checkpoint}")
    backend=TransformersBackend(checkpoint,MAX_NEW_TOKENS)
    yield
    backend=None

app=FastAPI(title="CriderGPT API",version="1.0.0",lifespan=lifespan)
origins=[x.strip() for x in os.getenv("CRIDERGPT_CORS_ORIGINS","").split(",") if x.strip()]
if origins:app.add_middleware(CORSMiddleware,allow_origins=origins,allow_credentials=True,allow_methods=["GET","POST"],allow_headers=["Authorization","Content-Type"])

@app.get("/health")
async def health():return {"status":"ok","model":MODEL_NAME,"loaded":backend is not None}

@app.get("/v1/models")
async def models(authorization:str|None=Header(default=None)):
    require_key(authorization)
    return {"object":"list","data":[{"id":MODEL_NAME,"object":"model","owned_by":"cridergpt"}]}

@app.post("/v1/generate")
async def generate(body:GenerateRequest,authorization:str|None=Header(default=None)):
    require_key(authorization);created=int(time.time())
    text=await infer(body.prompt,body.max_tokens)
    return {"id":"gen-"+uuid.uuid4().hex,"object":"text_completion","created":created,"model":MODEL_NAME,"text":text}

@app.post("/v1/chat/completions")
async def chat(body:ChatRequest,request:Request,authorization:str|None=Header(default=None)):
    require_key(authorization)
    if body.model and body.model!=MODEL_NAME:raise HTTPException(404,f"Model '{body.model}' is not served here")
    prompt=prompt_from_messages(body.messages);created=int(time.time());rid="chatcmpl-"+uuid.uuid4().hex
    text=await infer(prompt,body.max_tokens)
    if not body.stream:
        return {"id":rid,"object":"chat.completion","created":created,"model":MODEL_NAME,
                "choices":[{"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"}]}
    async def events():
        # HTTP streaming contract is ready now. Native token-by-token generation can replace this
        # single response chunk later without changing clients.
        import json
        chunk={"id":rid,"object":"chat.completion.chunk","created":created,"model":MODEL_NAME,
               "choices":[{"index":0,"delta":{"role":"assistant","content":text},"finish_reason":None}]}
        yield "data: "+json.dumps(chunk)+"\n\n"
        done={"id":rid,"object":"chat.completion.chunk","created":created,"model":MODEL_NAME,
              "choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
        yield "data: "+json.dumps(done)+"\n\n";yield "data: [DONE]\n\n"
    return StreamingResponse(events(),media_type="text/event-stream")
