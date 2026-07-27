from usage.tokenizer import audio_tokens, count_text_tokens, estimate_usage, image_tokens, video_tokens


def test_text_tokens_are_deterministic_and_nonzero():
    assert count_text_tokens("hello world") >= 2
    assert count_text_tokens("") == 0


def test_image_tokens_scale_with_resolution():
    assert image_tokens(2048, 2048) > image_tokens(512, 512)


def test_video_tokens_scale_with_duration_and_audio():
    short = video_tokens(2, 24, 1024, 576)
    long = video_tokens(8, 24, 1024, 576)
    assert long > short
    silent = estimate_usage("video", input_text="scene", duration_seconds=5, fps=24, width=1024, height=576)
    with_audio = estimate_usage("video", input_text="scene", duration_seconds=5, fps=24, width=1024, height=576, include_audio=True)
    assert with_audio.total_tokens > silent.total_tokens


def test_audio_tokens_scale_with_duration():
    assert audio_tokens(10) == 2 * audio_tokens(5)
