from usage.tokenizer import estimate_usage
from video.models import LocalVideoRequest


def test_local_video_request_contains_audio_controls():
    request = LocalVideoRequest(
        job_id="job-1",
        prompt="A tractor crossing a field",
        narration="The tractor starts moving.",
        sound_prompt="diesel engine and birds",
        music_prompt="quiet country instrumental",
    )
    data = request.as_dict()
    assert data["include_audio"] is True
    assert data["sound_prompt"] == "diesel engine and birds"


def test_video_usage_includes_prompt_and_media_units():
    usage = estimate_usage(
        "video",
        input_text="A short generated scene",
        width=1024,
        height=576,
        duration_seconds=5,
        fps=24,
        include_audio=True,
    )
    assert usage.input_tokens > 0
    assert usage.media_tokens > 0
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens + usage.media_tokens
