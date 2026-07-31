"""Chat endpoints matching the Supabase chat-with-ai Edge Function contract."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.auth import validate_api_key
from engine.agent import get_agent_response
from engine.inference import InferenceUnavailable

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    system_prompt: str | None = Field(default=None, max_length=150_000)
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    user_id: str | None = None
    conversation_id: str | None = None
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    image_url: str | None = None


@router.post("/chat")
@router.post("/chat-with-ai", include_in_schema=False)
async def chat(request: ChatRequest, _api_key: str = Depends(validate_api_key)):
    conversation_id = request.conversation_id or str(uuid.uuid4())
    try:
        result = get_agent_response(
            request.message,
            user_id=request.user_id,
            conversation_id=conversation_id,
            system_prompt=request.system_prompt,
            conversation_history=request.conversation_history,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except InferenceUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected chat failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat generation failed",
        ) from exc

    return {
        "response": result.response,
        "conversation_id": conversation_id,
        "model": result.model,
        "memories_used": result.memories_used,
        "latency_ms": result.latency_ms,
    }
