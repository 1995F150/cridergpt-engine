# Local video workflow

`LOCAL_VIDEO_WORKFLOW` points to a ComfyUI API-format workflow JSON file. Export the workflow in API format from the locally hosted ComfyUI instance and keep these placeholders in the appropriate node inputs:

- `{{PROMPT}}`
- `{{NEGATIVE_PROMPT}}`
- `{{MODEL}}`
- `{{WIDTH}}`
- `{{HEIGHT}}`
- `{{FPS}}`
- `{{FRAMES}}`
- `{{SEED}}`
- `{{GUIDANCE}}`
- `{{REFERENCE_IMAGE_URL}}`
- `{{JOB_ID}}`

The exact node graph depends on the local video model installed on the machine. Model weights and checkpoints must stay outside Git because they are large and may have separate licenses. The engine defaults to `/opt/cridergpt-engine/video/workflows/default.json`; deployment should place the selected model's exported API workflow there or set `LOCAL_VIDEO_WORKFLOW` to another path in the existing `.env`.

Audio generation is local and optional. Configure any locally hosted compatible services with `LOCAL_TTS_URL`, `LOCAL_SFX_URL`, and `LOCAL_MUSIC_URL`. If none are configured, the pipeline produces a valid silent audio track and still creates an MP4.
