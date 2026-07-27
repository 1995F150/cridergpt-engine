"""Authenticated token-equivalent estimation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.auth import validate_api_key
from usage.tokenizer import Modality, estimate_usage

router = APIRouter(prefix="/usage")


class UsageEstimateRequest(BaseModel):
    modality: Modality
    input_text: str | None = Field(default=None, max_length=100_000)
    output_text: str | None = Field(default=None, max_length=100_000)
    width: int = Field(default=0, ge=0, le=16384)
    height: int = Field(default=0, ge=0, le=16384)
    duration_seconds: float = Field(default=0, ge=0, le=7200)
    fps: int = Field(default=24, ge=1, le=240)
    include_audio: bool = False


@router.post("/estimate")
async def estimate(request: UsageEstimateRequest, _key: str = Depends(validate_api_key)):
    return estimate_usage(**request.model_dump()).as_dict()
