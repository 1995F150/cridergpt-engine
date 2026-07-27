from usage.tokenizer import (
    audio_tokens,
    count_text_tokens,
    estimate_usage,
    image_tokens,
    video_tokens,
)


def test_text_tokens_are_deterministic_and_nonzero():
    assert count_text_tokens("Hello, world!") == count_text_tokens("Hello, world!")
    assert count_text_tokens("Hello, world!") > 0
    assert count_text_tokens("") == 0


def test_image_tokens_scale_with_resolution_and_count():
    small = image_tokens(512, 512)
    large = image_tokens(1024, 1024)
    assert large > small
    assert image_tokens(512, 512, images=2) == small * 2


def test_audio_tokens_scale_with_duration():
    assert audio_tokens(10) == audio_tokens(5) * 2


def test_video_usage_includes_text_output_media_and_audio():
    estimate = estimate_usage(
        "video",
        input_text="A truck driving through the rain",
        output_text="queued",
        width=1024,
        height=576,
        duration_seconds=5,
        fps=24,
        include_audio=True,
    )
    assert estimate.input_tokens > 0
    assert estimate.output_tokens > 0
    assert estimate.media_tokens > video_tokens(5, 24, 1024, 576)
    assert estimate.total_tokens == (
        estimate.input_tokens + estimate.output_tokens + estimate.media_tokens
    )


def test_multimodal_usage_combines_language_images_and_audio():
    estimate = estimate_usage(
        "multimodal",
        input_text="Describe this picture and recording",
        width=1024,
        height=1024,
        image_count=2,
        duration_seconds=4,
    )
    assert estimate.input_tokens > 0
    assert estimate.media_tokens > image_tokens(1024, 1024, images=2)
