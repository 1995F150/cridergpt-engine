"""Deterministic token-equivalent accounting for every engine modality.

Language is estimated from UTF-8 text without requiring a model-specific tokenizer.
Media is converted to stable token-equivalent units so text, images, vision,
audio, and video can share quotas and usage reports without claiming that all
models internally tokenize media in the same way.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Modality = Literal[
    "text",
    "embedding",
    "image",
    "vision",
    "video",
    "audio",
    "multimodal",
    "tool",
]


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
    """Return a deterministic, conservative language-token estimate."""
    if not text:
        return 0
    lexical = len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
    utf8 = math.ceil(len(text.encode("utf-8")) / 4)
    return max(1, lexical, utf8)


def image_tokens(width: int, height: int, patch_size: int = 32, images: int = 1) -> int:
    width = max(1, int(width))
    height = max(1, int(height))
    patch_size = max(8, int(patch_size))
    images = max(1, int(images))
    return images * math.ceil(width / patch_size) * math.ceil(height / patch_size)


def audio_tokens(duration_seconds: float, tokens_per_second: int = 50) -> int:
    return max(0, math.ceil(max(0.0, float(duration_seconds)) * max(1, tokens_per_second)))


def video_tokens(
    duration_seconds: float,
    fps: int,
    width: int,
    height: int,
    temporal_stride: int = 4,
    spatial_patch: int = 64,
) -> int:
    sampled_frames = math.ceil(
        max(0.0, float(duration_seconds)) * max(1, int(fps)) / max(1, temporal_stride)
    )
    return sampled_frames * image_tokens(width, height, spatial_patch)


def estimate_usage(
    modality: Modality,
    *,
    input_text: str | None = None,
    output_text: str | None = None,
    width: int = 0,
    height: int = 0,
    image_count: int = 1,
    duration_seconds: float = 0,
    fps: int = 24,
    include_audio: bool = False,
) -> UsageEstimate:
    input_count = count_text_tokens(input_text)
    output_count = count_text_tokens(output_text)
    media = 0
    details: dict[str, Any] = {}

    if modality in {"image", "vision"}:
        resolved_width = width or 1024
        resolved_height = height or 1024
        media = image_tokens(resolved_width, resolved_height, images=image_count)
        details.update(width=resolved_width, height=resolved_height, image_count=max(1, image_count))
    elif modality == "video":
        resolved_width = width or 1024
        resolved_height = height or 576
        media = video_tokens(duration_seconds, fps, resolved_width, resolved_height)
        audio_equivalent = audio_tokens(duration_seconds) if include_audio else 0
        media += audio_equivalent
        details.update(
            width=resolved_width,
            height=resolved_height,
            duration_seconds=duration_seconds,
            fps=fps,
            audio=include_audio,
            audio_tokens=audio_equivalent,
        )
    elif modality == "audio":
        media = audio_tokens(duration_seconds)
        details.update(duration_seconds=duration_seconds)
    elif modality == "multimodal":
        if width or height or image_count > 0:
            media += image_tokens(width or 1024, height or 1024, images=image_count)
        if duration_seconds:
            media += audio_tokens(duration_seconds)
        details.update(
            width=width or 1024,
            height=height or 1024,
            image_count=max(1, image_count),
            duration_seconds=duration_seconds,
        )

    total = input_count + output_count + media
    return UsageEstimate(
        modality=modality,
        input_tokens=input_count,
        output_tokens=output_count,
        media_tokens=media,
        total_tokens=total,
        details=details,
    )
