"""CriderGPT Native single-model HTTP inference API."""
from __future__ import annotations
import asyncio,json,os,time,uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from fastapi import FastAPI,Header,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel,Field
from cridergpt_stage5.inference import TransformersBackend,clean_generated_text
ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=Path(os.getenv("CRIDERGPT_CHECKPOINT",str(ROOT/"model"/"cridergpt-2.1-nova"/"checkpoint"))).expanduser()
MODEL_NAME=os.getenv("CRIDERGPT_MODEL_NAME","cridergpt-2.1-nova")
API_KEY=os.getenv("CRIDERGPT_API_KEY","").strip()
MAX_INPUT_CHARS=int(os.getenv("CRIDERGPT_MAX_INPUT_CHARS","24000"))
DEFAULT_MAX_TOKENS=int(os.getenv("CRIDERGPT_MAX_NEW_TOKENS","192"))
lock=asyncio.Lock()
class Message(BaseModel):
    role:Literal["system","user","assistant"];content:str=Field(min_length=1,max_length=12000)
class ChatRequest(BaseModel):
    model:str=MODEL_NAME;messages:list[Message]=Field(min_length=1,max_length=64);stream:bool=False
    max_tokens:int=Field(default=DEFAULT_MAX_TOKENS,ge=1,le=1024)
class State: backend:TransformersBackend|None=None
state=State()
def authorize(value):
    if API_KEY and value!=f"Bearer {API_KEY}":raise HTTPException(401,"Invalid or missing API key")
def render(messages):
    parts=[]
    for m in messages:
        if m.role=="system":parts.append(f"System instruction: {m.content.strip()}")
        elif m.role=="user":parts.append(f"<|user|>\n{m.content.strip()}")
        else:parts.append(f"<|assistant|>\n{m.content.strip()}\n<|eos|>")
    parts.append("<|assistant|>\n");text="\n".join(parts)
    if len(text)>MAX_INPUT_CHARS:raise HTTPException(413,"Conversation is too large")
    return text
def infer(messages,max_tokens):
    b=state.backend
    if b is None:raise RuntimeError("model not loaded")
    prompt=render(messages);inputs=b.tokenizer(prompt,return_tensors="pt")
    inputs={k:v.to(b.device) for k,v in inputs.items()};n=inputs["input_ids"].shape[-1]
    with b.torch.inference_mode():
        out=b.model.generate(**inputs,max_new_tokens=max_tokens,min_new_tokens=1,do_sample=False,repetition_penalty=1.10,pad_token_id=b.tokenizer.pad_token_id,eos_token_id=b.tokenizer.eos_token_id)
    return clean_generated_text(b.tokenizer.decode(out[0][n:],skip_special_tokens=False))
@asynccontextmanager
async def lifespan(app):
    if not CHECKPOINT.exists():raise RuntimeError(f"checkpoint not found: {CHECKPOINT}")
    state.backend=TransformersBackend(CHECKPOINT,DEFAULT_MAX_TOKENS);yield;state.backend=None
app=FastAPI(title="CriderGPT Native API",version="2.1.0",lifespan=lifespan)
origins=[x.strip() for x in os.getenv("CRIDERGPT_CORS_ORIGINS","").split(",") if x.strip()]
if origins:app.add_middleware(CORSMiddleware,allow_origins=origins,allow_credentials=True,allow_methods=["GET","POST"],allow_headers=["Authorization","Content-Type"])
@app.get("/")
async def root():return {"service":"CriderGPT Native API","model":MODEL_NAME,"docs":"/docs"}
@app.get("/health")
async def health():return {"status":"ok" if state.backend else "loading","model":MODEL_NAME,"device":getattr(state.backend,"device",None)}
@app.get("/v1/models")
async def models(authorization:str|None=Header(default=None)):
    authorize(authorization);return {"object":"list","data":[{"id":MODEL_NAME,"object":"model","owned_by":"cridergpt"}]}
@app.post("/v1/chat/completions")
async def chat(req:ChatRequest,authorization:str|None=Header(default=None)):
    authorize(authorization)
    if req.model!=MODEL_NAME:raise HTTPException(404,f"Unknown model: {req.model}")
    rid="chatcmpl-"+uuid.uuid4().hex;created=int(time.time())
    async with lock:text=await asyncio.to_thread(infer,req.messages,req.max_tokens)
    if not req.stream:return {"id":rid,"object":"chat.completion","created":created,"model":MODEL_NAME,"choices":[{"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"}]}
    async def events():
        yield "data: "+json.dumps({"id":rid,"object":"chat.completion.chunk","created":created,"model":MODEL_NAME,"choices":[{"index":0,"delta":{"role":"assistant","content":text},"finish_reason":None}]},ensure_ascii=False)+"\n\n"
        yield "data: "+json.dumps({"id":rid,"object":"chat.completion.chunk","created":created,"model":MODEL_NAME,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]})+"\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(events(),media_type="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
