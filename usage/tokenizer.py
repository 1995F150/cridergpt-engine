"""Deterministic token-equivalent estimates for every engine modality.

Text uses a conservative UTF-8/word estimate. Images and video use spatial patch
counts so all generation tools can share quotas without pretending media models
consume normal language tokens internally.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Modality = Literal["text", "image", "vision", "video", "audio", "multimodal"]


@dataclass(frozen=True)
class UsageEstimate:
    modality: Modality
    input_tokens: int = 0
    output_tokens: int = 0
    media_tokens: int = 0
    total_tokens: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def count_text_tokens(text: str | None) -> int:
    """Estimate tokenizer output without adding a heavyweight runtime dependency."""
    if not text:
        return 0
    words = len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
    byte_estimate = math.ceil(len(text.encode("utf-8")) / 4)
    return max(1, words, byte_estimate)


def image_tokens(width: int, height: int, patch_size: int = 32) -> int:
    width = max(1, width)
    height = max(1, height)
    patch_size = max(8, patch_size)
    return math.ceil(width / patch_size) * math.ceil(height / patch_size)


def audio_tokens(duration_seconds: float, tokens_per_second: int = 50) -> int:
    return max(0, math.ceil(max(0.0, duration_seconds) * max(1, tokens_per_second)))


def video_tokens(
    duration_seconds: float,
    fps: int,
    width: int,
    height: int,
    temporal_stride: int = 4,
    spatial_patch: int = 64,
) -> int:
    sampled_frames = math.ceil(max(0.0, duration_seconds) * max(1, fps) / max(1, temporal_stride))
    return sampled_frames * image_tokens(width, height, spatial_patch)


def estimate_usage(
    modality: Modality,
    *,
    input_text: str | None = None,
    output_text: str | None = None,
    width: int = 0,
    height: int = 0,
    duration_seconds: float = 0,
    fps: int = 24,
    include_audio: bool = False,
) -> UsageEstimate:
    input_count = count_text_tokens(input_text)
    output_count = count_text_tokens(output_text)
    media = 0
    details: dict[str, Any] = {}

    if modality in {"image", "vision"}:
        media = image_tokens(width or 1024, height or 1024)
        details.update(width=width or 1024, height=height or 1024)
    elif modality == "video":
        media = video_tokens(duration_seconds, fps, width or 1024, height or 576)
        if include_audio:
            media += audio_tokens(duration_seconds)
        details.update(
            width=width or 1024,
            height=height or 576,
            duration_seconds=duration_seconds,
            fps=fps,
            audio=include_audio,
        )
    elif modality == "audio":
        media = audio_tokens(duration_seconds)
        details.update(duration_seconds=duration_seconds)
    elif modality == "multimodal":
        media = image_tokens(width or 1024, height or 1024)
        if duration_seconds:
            media += audio_tokens(duration_seconds)

    total = input_count + output_count + media
    return UsageEstimate(
        modality=modality,
        input_tokens=input_count,
        output_tokens=output_count,
        media_tokens=media,
        total_tokens=total,
        details=details,
    )
