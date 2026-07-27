# Production Readiness

## Automated gates

The pull request must not merge unless the `Engine CI` workflow passes on Python 3.11 and 3.12. The workflow verifies:

- dependency installation from `requirements.txt`
- compilation of every tracked Python file
- import of the production FastAPI application
- the complete pytest suite
- shell syntax for the updater
- presence of all required systemd units

## Required deployment configuration

Before enabling local video generation in production:

1. Apply all migrations under `supabase/migrations/` in timestamp order.
2. Confirm `/opt/cridergpt-engine/.env` contains valid engine API keys, Supabase credentials, Ollama configuration, image backend configuration, and local video settings.
3. Install FFmpeg and verify `ffmpeg` and `ffprobe` are available in `PATH`.
4. Install and start a loopback-only ComfyUI-compatible video runtime.
5. Install a licensed local video model and export its workflow in API format.
6. Set `LOCAL_VIDEO_WORKFLOW` or provide `video/workflows/default.json`.
7. Configure local narration, sound-effect, and music endpoints when generated audio is required.
8. Install and enable both `cridergpt-engine.service` and `cridergpt-video-worker.service`.

## Required smoke tests

Run these after deployment and before exposing the release to users:

- `GET /health`
- authenticated text chat request
- authenticated `/usage/estimate` request
- image generation request against Automatic1111 or Forge
- image-analysis request against the configured Ollama vision model
- video job creation, worker pickup, progress update, and cancellation
- one completed local video with a valid preview image
- one completed local video containing a playable audio track
- confirmation that an `ai_usage_events` row is stored
- confirmation that a `video_generation_jobs` row reaches `completed`
- restart both systemd services and repeat `/health`

## Rollback

Keep the pre-merge `main` commit available. If the production health check, image smoke test, database writes, or worker startup fails, revert the merge and restart the existing engine service. Do not modify or replace the production `.env` or virtual environment during an automatic rollback.
