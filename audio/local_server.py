"""Optional loopback-only narration, sound-effect, and music service.

Narration uses the locally installed Piper executable and voice model. Sound effects
and music use AudioCraft lazily when its optional dependencies and model weights are
installed. The main engine remains usable when this service is disabled.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="CriderGPT Local Audio", version="1.0.0")
OUTPUT_DIR = Path(os.getenv("LOCAL_AUDIO_OUTPUT_DIR", "/opt/cridergpt-engine/data/audio"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class PromptRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4_000)
    duration_seconds: int = Field(default=5, ge=1, le=60)


def _safe_output(prefix: str) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".wav", dir=OUTPUT_DIR, delete=False)
    handle.close()
    return Path(handle.name)


@app.get("/health")
def health():
    return {
        "status": "running",
        "piper_configured": bool(os.getenv("PIPER_MODEL")),
        "audiocraft_enabled": os.getenv("ENABLE_AUDIOCRAFT", "0") == "1",
    }


@app.post("/tts")
def tts(request: SpeechRequest):
    model = os.getenv("PIPER_MODEL")
    executable = os.getenv("PIPER_EXECUTABLE", "/usr/local/bin/piper")
    if not model:
        raise HTTPException(status_code=503, detail="PIPER_MODEL is not configured")
    output = _safe_output("tts-")
    try:
        subprocess.run(
            [executable, "--model", model, "--output_file", str(output)],
            input=request.text.encode("utf-8"),
            check=True,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        output.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=f"Piper failed: {exc}") from exc
    return FileResponse(output, media_type="audio/wav", filename=output.name)


def _audiocraft(prompt: str, duration: int, kind: str) -> Path:
    if os.getenv("ENABLE_AUDIOCRAFT", "0") != "1":
        raise HTTPException(status_code=503, detail="AudioCraft is disabled")
    try:
        import torchaudio
        from audiocraft.models import AudioGen, MusicGen
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="AudioCraft dependencies are not installed") from exc

    if kind == "music":
        model_name = os.getenv("MUSICGEN_MODEL", "facebook/musicgen-small")
        model = MusicGen.get_pretrained(model_name)
    else:
        model_name = os.getenv("AUDIOGEN_MODEL", "facebook/audiogen-medium")
        model = AudioGen.get_pretrained(model_name)
    model.set_generation_params(duration=duration)
    waveform = model.generate([prompt])[0].cpu()
    output = _safe_output(f"{kind}-")
    torchaudio.save(str(output), waveform, model.sample_rate)
    return output


@app.post("/sfx")
def sfx(request: PromptRequest):
    output = _audiocraft(request.prompt, request.duration_seconds, "sfx")
    return FileResponse(output, media_type="audio/wav", filename=output.name)


@app.post("/music")
def music(request: PromptRequest):
    output = _audiocraft(request.prompt, request.duration_seconds, "music")
    return FileResponse(output, media_type="audio/wav", filename=output.name)
