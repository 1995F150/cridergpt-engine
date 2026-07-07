import uuid
from typing import Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from api.auth import validate_api_key
from engine.agent import get_agent_response

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    memories_used: list = []
    model: str = "cridergpt-engine"
    latency_ms: int = 0

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, api_key: str = Depends(validate_api_key)):
    """
    Secure chat endpoint that validates the API key and generates a response.
    """
    # 1. Handle missing conversation_id
    conversation_id = request.conversation_id
    if not conversation_id:
        conversation_id = str(uuid.uuid4())

    # 2. Handle missing user_id (guest turn)
    user_id = request.user_id or "guest"

    # 3. Get response from fixed agent logic
    # This uses engine/agent.py which has the memory cap and system prompt.
    response_text = get_agent_response(request.message, user_id, conversation_id)
    
    return ChatResponse(
        conversation_id=conversation_id,
        response=response_text
    )
