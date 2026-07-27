"""Safe FFmpeg helpers for local video assembly."""

from __future__ import annotations

import subprocess
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def _run(args: list[str], timeout: int = 1800) -> None:
    try:
        subprocess.run(args, check=True, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise FFmpegError("ffmpeg is not installed") from exc
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or "ffmpeg failed")[-2000:]
        raise FFmpegError(message) from exc


def mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path),
        "-c:v", "copy", "-c:a", "aac", "-shortest", "-movflags", "+faststart",
        str(output_path),
    ])
    return output_path


def make_preview(video_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-ss", "00:00:01", "-i", str(video_path), "-frames:v", "1", str(output_path)], timeout=120)
    return output_path
