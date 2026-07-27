"""Authenticated asynchronous local video generation API."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, HttpUrl

from api.auth import validate_api_key
from config import settings
from usage.tokenizer import estimate_usage
from video.jobs import apply_webhook, cancel_job, create_job, get_job, list_jobs
from video.providers import VideoProviderError

router = APIRouter(prefix="/video")


class VideoGenerationRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=20_000)
    negative_prompt: str | None = Field(default=None, max_length=8_000)
    model: str | None = Field(default=None, max_length=200)
    duration_seconds: int = Field(default=5, ge=1, le=60)
    aspect_ratio: str = Field(default="16:9", pattern=r"^(16:9|9:16|1:1|4:3|3:4)$")
    width: int = Field(default=1024, ge=256, le=4096)
    height: int = Field(default=576, ge=256, le=4096)
    reference_image_url: HttpUrl | None = None
    seed: int | None = Field(default=None, ge=0)
    fps: int = Field(default=24, ge=8, le=60)
    guidance_scale: float | None = Field(default=None, ge=0, le=30)
    narration: str | None = Field(default=None, max_length=20_000)
    sound_prompt: str | None = Field(default=None, max_length=4_000)
    music_prompt: str | None = Field(default=None, max_length=4_000)
    include_audio: bool = True
    callback_url: HttpUrl | None = None


class VideoJobQuery(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)


def _payload(request: VideoGenerationRequest) -> dict:
    data = request.model_dump(mode="json", exclude_none=True)
    data.pop("user_id", None)
    return data


@router.post("/generate", status_code=status.HTTP_202_ACCEPTED)
async def generate_video(request: VideoGenerationRequest, _key: str = Depends(validate_api_key)):
    try:
        job = create_job(request.user_id, _payload(request))
        estimate = estimate_usage(
            "video",
            input_text=" ".join(filter(None, [request.prompt, request.narration, request.sound_prompt, request.music_prompt])),
            width=request.width,
            height=request.height,
            duration_seconds=request.duration_seconds,
            fps=request.fps,
            include_audio=request.include_audio,
        )
        return {**job, "estimated_usage": estimate.as_dict()}
    except VideoProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/jobs/{job_id}")
async def video_job(job_id: str, user_id: str, refresh: bool = True, _key: str = Depends(validate_api_key)):
    try:
        return get_job(user_id, job_id, refresh=refresh)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Video job not found") from exc
    except VideoProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/jobs")
async def video_jobs(user_id: str, limit: int = 20, _key: str = Depends(validate_api_key)):
    try:
        return {"jobs": list_jobs(user_id, limit)}
    except VideoProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/cancel")
async def cancel_video(job_id: str, query: VideoJobQuery, _key: str = Depends(validate_api_key)):
    try:
        return cancel_job(query.user_id, job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Video job not found") from exc
    except VideoProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/files/{job_id}/{filename}")
async def generated_file(job_id: str, filename: str, _key: str = Depends(validate_api_key)):
    if filename not in {"video.mp4", "preview.jpg", "audio.wav"}:
        raise HTTPException(status_code=404, detail="Generated file not found")
    root = Path(settings.video_output_dir).resolve()
    path = (root / job_id / filename).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Generated file not found")
    return FileResponse(path)


@router.post("/webhook", include_in_schema=False)
async def video_webhook(request: Request, x_video_signature: str | None = Header(default=None)):
    raw = await request.body()
    if settings.video_webhook_secret:
        expected = hmac.new(settings.video_webhook_secret.encode(), raw, hashlib.sha256).hexdigest()
        supplied = (x_video_signature or "").removeprefix("sha256=")
        if not hmac.compare_digest(expected, supplied):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")
    try:
        payload = json.loads(raw)
        provider_job_id = str(payload.get("id") or payload.get("job_id") or payload.get("task_id") or "")
        if not provider_job_id:
            raise ValueError("missing provider job id")
        result = apply_webhook(provider_job_id, payload)
        return {"accepted": result is not None}
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail="Invalid webhook payload") from exc
