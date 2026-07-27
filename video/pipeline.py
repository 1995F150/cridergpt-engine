"""End-to-end local video, sound, and encoding pipeline."""

from __future__ import annotations

from pathlib import Path

from config import settings
from usage.meter import record_usage
from usage.tokenizer import estimate_usage
from video import audio
from video.ffmpeg import make_preview, mux_audio
from video.local_backend import LocalComfyVideoBackend
from video.models import LocalVideoRequest, LocalVideoResult


class LocalVideoPipeline:
    def __init__(self) -> None:
        self.backend = LocalComfyVideoBackend()

    def run(self, request: LocalVideoRequest, user_id: str) -> LocalVideoResult:
        root = Path(settings.video_output_dir).resolve() / request.job_id
        root.mkdir(parents=True, exist_ok=True)
        silent_video = self.backend.generate(request, root)
        final_video = root / "video.mp4"
        audio_path: Path | None = None

        if request.include_audio:
            tracks: list[Path] = []
            voice = audio.narration(request.narration, root / "narration.wav")
            effects = audio.sound_effects(request.sound_prompt, request.duration_seconds, root / "effects.wav")
            soundtrack = audio.music(request.music_prompt, request.duration_seconds, root / "music.wav")
            tracks.extend(path for path in (voice, effects, soundtrack) if path is not None)
            if not tracks:
                tracks.append(audio.silence(request.duration_seconds, root / "silence.wav"))
            audio_path = audio.mix_tracks(tracks, root / "audio.wav")
            mux_audio(silent_video, audio_path, final_video)
        else:
            final_video.write_bytes(silent_video.read_bytes())

        preview = make_preview(final_video, root / "preview.jpg")
        estimate = estimate_usage(
            "video",
            input_text=" ".join(filter(None, [request.prompt, request.narration, request.sound_prompt, request.music_prompt])),
            width=request.width,
            height=request.height,
            duration_seconds=request.duration_seconds,
            fps=request.fps,
            include_audio=request.include_audio,
        )
        record_usage(user_id, "video.generate", estimate, model=request.model, request_id=request.job_id)
        return LocalVideoResult(
            video_path=final_video,
            preview_path=preview,
            audio_path=audio_path,
            model=request.model or settings.local_video_model,
            metadata={"usage": estimate.as_dict()},
        )
