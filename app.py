from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from engine.agent import get_agent_response
from memory.memory_exporter import export_memory
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CriderGPT Engine")

class ChatRequest(BaseModel):
    text: str = None
    message: str = None
    user_id: str
    conversation_id: str = None

async def process_chat(request: ChatRequest):
    logger.info(f"Received request: message={request.message}, text={request.text}, user_id={request.user_id}, conversation_id={request.conversation_id}")
    
    if not request.user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    
    actual_message = request.message if request.message else request.text
    if not actual_message:
        raise HTTPException(status_code=400, detail="Either 'text' or 'message' must be provided")
    
    try:
        response = get_agent_response(
            message=actual_message,
            user_id=request.user_id,
            conversation_id=request.conversation_id
        )
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/chat-with-ai")
async def chat_with_ai(request: ChatRequest):
    return await process_chat(request)

@app.post("/chat")
async def chat(request: ChatRequest):
    return await process_chat(request)

@app.post("/export-memory")
async def export_mem():
    return export_memory()
