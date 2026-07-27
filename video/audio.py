"""Local audio generation and mixing for generated videos.

The engine calls local, operator-controlled HTTP services. No third-party API key
is required. Each service may be disabled independently until its local model is
installed.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import requests

from config import settings
from video.ffmpeg import FFmpegError, _run


class AudioGenerationError(RuntimeError):
    pass


def _generate(url: str | None, payload: dict[str, Any], output: Path) -> Path | None:
    if not url:
        return None
    try:
        response = requests.post(url, json=payload, timeout=(10, settings.video_audio_timeout_seconds))
        response.raise_for_status()
    except requests.RequestException as exc:
        raise AudioGenerationError(f"Local audio model failed: {exc}") from exc
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        body = response.json()
        download_url = body.get("audio_url") or body.get("url")
        if not download_url:
            raise AudioGenerationError("Local audio model returned no audio")
        response = requests.get(download_url, timeout=(10, settings.video_audio_timeout_seconds))
        response.raise_for_status()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(response.content)
    return output


def narration(text: str | None, output: Path) -> Path | None:
    return _generate(settings.local_tts_url, {"text": text, "format": "wav"}, output) if text else None


def sound_effects(prompt: str | None, duration: int, output: Path) -> Path | None:
    return _generate(settings.local_sfx_url, {"prompt": prompt, "duration_seconds": duration, "format": "wav"}, output) if prompt else None


def music(prompt: str | None, duration: int, output: Path) -> Path | None:
    return _generate(settings.local_music_url, {"prompt": prompt, "duration_seconds": duration, "format": "wav"}, output) if prompt else None


def silence(duration: int, output: Path, sample_rate: int = 44100) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(b"\x00\x00\x00\x00" * sample_rate * max(1, duration))
    return output


def mix_tracks(tracks: list[Path], output: Path) -> Path:
    existing = [path for path in tracks if path.exists()]
    if not existing:
        raise AudioGenerationError("No audio tracks were generated")
    if len(existing) == 1:
        output.write_bytes(existing[0].read_bytes())
        return output
    args = ["ffmpeg", "-y"]
    for track in existing:
        args += ["-i", str(track)]
    args += ["-filter_complex", f"amix=inputs={len(existing)}:duration=longest:normalize=0", "-c:a", "pcm_s16le", str(output)]
    try:
        _run(args)
    except FFmpegError as exc:
        raise AudioGenerationError(str(exc)) from exc
    return output
