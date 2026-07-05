# API endpoint for chat functionality
from fastapi import APIRouter

router = APIRouter()

@router.post("/chat")
async def chat_endpoint(message: str):
    """Handles chat messages via the API."""
    return {"message": "Chat endpoint received: " + messfrom fastapi import APIRouter, Depends
from api.auth import validate_api_key
from engine.inference import generate
from pydantic import BaseModel

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
async def chat(request: ChatRequest, api_key: str = Depends(validate_api_key)):
    """
    Secure chat endpoint that validates the API key and generates a response.
    """
    response = generate(request.message)
    return {"response": response}age}
