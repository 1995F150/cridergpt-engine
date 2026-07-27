from video.providers import HttpVideoProvider


def test_provider_normalizes_common_job_fields():
    job = HttpVideoProvider._normalize(
        {
            "task_id": "abc123",
            "status": "completed",
            "video_url": "https://example.com/video.mp4",
            "thumbnail_url": "https://example.com/preview.jpg",
        }
    )
    assert job.provider_job_id == "abc123"
    assert job.status == "completed"
    assert job.output_url.endswith("video.mp4")
    assert job.preview_url.endswith("preview.jpg")


def test_provider_normalizes_failed_job():
    job = HttpVideoProvider._normalize(
        {"id": "failed-1", "status": "failed", "message": "generation failed"}
    )
    assert job.error == "generation failed"
