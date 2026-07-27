"""Local video generation request and result models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LocalVideoRequest:
    job_id: str
    prompt: str
    negative_prompt: str | None = None
    model: str | None = None
    duration_seconds: int = 5
    width: int = 1024
    height: int = 576
    fps: int = 24
    seed: int | None = None
    guidance_scale: float | None = None
    reference_image_url: str | None = None
    narration: str | None = None
    sound_prompt: str | None = None
    music_prompt: str | None = None
    include_audio: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LocalVideoResult:
    video_path: Path
    preview_path: Path | None = None
    audio_path: Path | None = None
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
