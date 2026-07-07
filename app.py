import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from engine.agent import get_agent_response
from memory.memory_exporter import export_memory
import logging

# Configure loggingimport uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from engine.agent import get_agent_response
from memory.memory_exporter import export_memory
import logging
from api.image import router as image_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CriderGPT Engine")
app.include_router(image_router)

class ChatRequest(BaseModel):
    text: Optional[str] = None
    message: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None

async def process_chat(request: ChatRequest):
    logger.info(f"Received request: message={request.message}, text={request.text}, user_id={request.user_id}, conversation_id={request.conversation_id}")
    
    actual_message = request.message if request.message else request.text
    if not actual_message:
        raise HTTPException(status_code=400, detail="Either 'text' or 'message' must be provided")
    
    conv_id = request.conversation_id or str(uuid.uuid4())
    
    try:
        response = get_agent_response(
            message=actual_message,
            user_id=request.user_id,
            conversation_id=conv_id
        )
        
        return {
            "response": response,
            "conversation_id": conv_id,
            "user_id": request.user_id,
            "model": "cridergpt-engine",
            "memories_used": []
        }
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/chat-with-ai")
async def chat_with_ai(request: ChatRequest):
    return await process_chat(request)

@app.post("/chat")
async def chat(request: ChatRequest):
    return await process_chat(request)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "capabilities": ["chat", "image_generate", "image_recognize"]
    }

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CriderGPT Engine")

class ChatRequest(BaseModel):
    text: Optional[str] = None
    message: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None

async def process_chat(request: ChatRequest):
    logger.info(f"Received request: message={request.message}, text={request.text}, user_id={request.user_id}, conversation_id={request.conversation_id}")
    
    actual_message = request.message if request.message else request.text
    if not actual_message:
        raise HTTPException(status_code=400, detail="Either 'text' or 'message' must be provided")
        
    conv_id = request.conversation_id or str(uuid.uuid4())
    
    try:
        response = get_agent_response(
            message=actual_message,
            user_id=request.user_id,
            conversation_id=conv_id
        )
        return {
            "response": response,
            "conversation_id": conv_id,
            "user_id": request.user_id,
            "model": "cridergpt-engine",
            "memories_used": []
        }
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/chat-with-ai")
async def chat_with_ai(request: ChatRequest):
    return await process_chat(request)

@app.post("/chat")
async def chat(request: ChatRequest):
    return await process_chat(request)
