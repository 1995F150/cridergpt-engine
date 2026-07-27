"""Local text/image-to-video backend controlled by CriderGPT Engine.

This adapter submits workflows to a locally hosted ComfyUI-compatible service.
The service URL defaults to loopback, keeping generation independent from an
external video provider. Workflow JSON is supplied by the engine deployment.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from config import settings
from video.models import LocalVideoRequest


class LocalVideoError(RuntimeError):
    pass


class LocalComfyVideoBackend:
    def __init__(self) -> None:
        self.base_url = settings.local_video_url.rstrip("/")
        self.workflow_path = Path(settings.local_video_workflow)

    def available(self) -> bool:
        try:
            return requests.get(f"{self.base_url}/system_stats", timeout=2).ok
        except requests.RequestException:
            return False

    def _workflow(self, request: LocalVideoRequest) -> dict[str, Any]:
        if not self.workflow_path.is_file():
            raise LocalVideoError(f"Local video workflow is missing: {self.workflow_path}")
        workflow = json.loads(self.workflow_path.read_text(encoding="utf-8"))
        replacements = {
            "{{PROMPT}}": request.prompt,
            "{{NEGATIVE_PROMPT}}": request.negative_prompt or "",
            "{{MODEL}}": request.model or settings.local_video_model,
            "{{WIDTH}}": request.width,
            "{{HEIGHT}}": request.height,
            "{{FPS}}": request.fps,
            "{{FRAMES}}": request.duration_seconds * request.fps,
            "{{SEED}}": request.seed if request.seed is not None else 0,
            "{{GUIDANCE}}": request.guidance_scale or 7.0,
            "{{REFERENCE_IMAGE_URL}}": request.reference_image_url or "",
            "{{JOB_ID}}": request.job_id,
        }
        encoded = json.dumps(workflow)
        for key, value in replacements.items():
            encoded = encoded.replace(key, str(value))
        return json.loads(encoded)

    def generate(self, request: LocalVideoRequest, work_dir: Path) -> Path:
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            response = requests.post(
                f"{self.base_url}/prompt",
                json={"prompt": self._workflow(request), "client_id": request.job_id},
                timeout=(10, 60),
            )
            response.raise_for_status()
            prompt_id = str(response.json()["prompt_id"])
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise LocalVideoError(f"Could not start local video generation: {exc}") from exc

        deadline = time.monotonic() + settings.local_video_timeout_seconds
        while time.monotonic() < deadline:
            try:
                history = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=10)
                history.raise_for_status()
                record = history.json().get(prompt_id)
            except requests.RequestException as exc:
                raise LocalVideoError(f"Could not read local video progress: {exc}") from exc
            if record:
                for output in (record.get("outputs") or {}).values():
                    candidates = output.get("videos") or output.get("gifs") or output.get("images") or []
                    for item in candidates:
                        filename = item.get("filename")
                        if filename and str(filename).lower().endswith((".mp4", ".webm", ".gif")):
                            params = {"filename": filename, "subfolder": item.get("subfolder", ""), "type": item.get("type", "output")}
                            media = requests.get(f"{self.base_url}/view", params=params, timeout=(10, 300))
                            media.raise_for_status()
                            output_path = work_dir / "silent-video.mp4"
                            output_path.write_bytes(media.content)
                            return output_path
                status = record.get("status") or {}
                if status.get("status_str") == "error":
                    raise LocalVideoError("Local video workflow failed")
            time.sleep(settings.local_video_poll_seconds)
        raise LocalVideoError("Local video generation timed out")
