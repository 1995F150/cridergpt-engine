"""Native GPU image generation and Ollama vision endpoints."""

from __future__ import annotations

import base64
import ipaddress
import logging
import time
from typing import Any
from urllib.parse import urlparse

import requests
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.auth import validate_api_key
from config import settings
from memory.memory_store import get_supabase

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/image")


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    reference_urls: list[str] = Field(default_factory=list)
    character_name: str | None = None
    size: str = "1024x1024"
    model: str | None = None
    negative_prompt: str = (
        "low quality, blurry, distorted face, extra fingers, duplicate person"
    )
    steps: int = Field(default=28, ge=1, le=100)
    cfg_scale: float = Field(default=7.0, ge=1, le=30)


class ImageAnalysisRequest(BaseModel):
    image_url: str | None = None
    image_base64: str | None = None
    prompt: str = "Describe this image in detail."
    model: str | None = None


def _dimensions(size: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in size.lower().split("x", 1))
    except (TypeError, ValueError):
        return 1024, 1024
    return max(256, min(width, 2048)), max(256, min(height, 2048))


def _allowed_reference_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname:
            return False
        try:
            address = ipaddress.ip_address(parsed.hostname)
            if address.is_private or address.is_loopback or address.is_link_local:
                return False
        except ValueError:
            pass
        allowed = set(settings.allowed_reference_hosts)
        if settings.supabase_url:
            host = urlparse(settings.supabase_url).hostname
            if host:
                allowed.add(host)
        return not allowed or parsed.hostname in allowed
    except ValueError:
        return False


def _download_image(url: str) -> str:
    if not _allowed_reference_url(url):
        raise ValueError("Reference URL host is not allowed")
    response = requests.get(url, timeout=(5, 20), allow_redirects=False)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise ValueError("Reference URL did not return an image")
    if len(response.content) > 12 * 1024 * 1024:
        raise ValueError("Reference image exceeds 12 MB")
    return base64.b64encode(response.content).decode("ascii")


def _database_references(prompt: str, character_name: str | None) -> list[str]:
    client = get_supabase()
    if client is None:
        return []
    try:
        rows = (
            client.table("character_references")
            .select("name,slug,reference_photo_url,is_primary,generation_count")
            .eq("is_system", True)
            .execute()
            .data
            or []
        )
    except Exception as exc:
        logger.warning("Character reference lookup failed: %s", exc)
        return []

    haystack = f"{prompt} {character_name or ''}".lower()
    matched = [
        row
        for row in rows
        if row.get("reference_photo_url")
        and (
            str(row.get("name") or "").lower() in haystack
            or str(row.get("slug") or "").lower().replace("-", " ") in haystack
        )
    ]
    matched.sort(
        key=lambda row: (
            bool(row.get("is_primary")),
            int(row.get("generation_count") or 0),
        ),
        reverse=True,
    )
    return [
        str(row["reference_photo_url"])
        for row in matched[: settings.max_reference_images]
    ]


def _generate_automatic1111(
    request: ImageGenerationRequest, refs: list[str]
) -> tuple[str, str]:
    width, height = _dimensions(request.size)
    payload: dict[str, Any] = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "width": width,
        "height": height,
        "steps": request.steps,
        "cfg_scale": request.cfg_scale,
        "batch_size": 1,
    }
    if request.model or settings.image_model:
        payload["override_settings"] = {
            "sd_model_checkpoint": request.model or settings.image_model
        }

    endpoint = "/sdapi/v1/txt2img"
    if refs:
        payload["init_images"] = [_download_image(url) for url in refs]
        payload["denoising_strength"] = 0.65
        endpoint = "/sdapi/v1/img2img"

    response = requests.post(
        f"{settings.image_api_url}{endpoint}",
        json=payload,
        timeout=(10, settings.image_timeout_seconds),
    )
    response.raise_for_status()
    data = response.json()
    images = data.get("images") or []
    if not images:
        raise RuntimeError("GPU backend returned no image")
    raw = str(images[0]).split(",", 1)[-1]
    return raw, str(request.model or settings.image_model or "automatic1111")


@router.post("/generate")
async def generate_image(
    request: ImageGenerationRequest, _api_key: str = Depends(validate_api_key)
):
    started = time.monotonic()
    refs = request.reference_urls[: settings.max_reference_images]
    if not refs:
        refs = _database_references(request.prompt, request.character_name)

    try:
        if settings.image_backend not in {"automatic1111", "a1111", "forge"}:
            raise RuntimeError(f"Unsupported IMAGE_BACKEND: {settings.image_backend}")
        encoded, model = _generate_automatic1111(request, refs)
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        logger.error("Image generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The local GPU image backend is unavailable or rejected the request",
        ) from exc

    return {
        "image_base64": encoded,
        "image_url": f"data:image/png;base64,{encoded}",
        "model": model,
        "references_used": len(refs),
        "latency_ms": int((time.monotonic() - started) * 1000),
    }


@router.post("/analyze")
async def analyze_image(
    request: ImageAnalysisRequest, _api_key: str = Depends(validate_api_key)
):
    if not request.image_url and not request.image_base64:
        raise HTTPException(
            status_code=422, detail="image_url or image_base64 is required"
        )
    try:
        encoded = request.image_base64 or _download_image(request.image_url or "")
        if encoded.startswith("data:"):
            encoded = encoded.split(",", 1)[-1]
        started = time.monotonic()
        response = requests.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": request.model or settings.ollama_vision_model,
                "stream": False,
                "messages": [
                    {"role": "user", "content": request.prompt, "images": [encoded]}
                ],
            },
            timeout=(10, settings.ollama_timeout_seconds),
        )
        response.raise_for_status()
        body = response.json()
        description = str(
            body.get("message", {}).get("content") or body.get("response") or ""
        ).strip()
        if not description:
            raise RuntimeError("Vision model returned an empty response")
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        logger.error("Image analysis failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The local vision model is unavailable or rejected the request",
        ) from exc

    return {
        "description": description,
        "model": request.model or settings.ollama_vision_model,
        "latency_ms": int((time.monotonic() - started) * 1000),
    }


def route_for_mode(mode: str | None) -> str:
    routes = {
        "chat": "/chat",
        "image_generate": "/image/generate",
        "image_recognize": "/image/analyze",
    }
    return routes.get((mode or "chat").lower(), "/chat")
