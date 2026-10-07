# CriderGPT Native API

CriderGPT uses one local model process. It does not route prompts to another LLM.

## What kind of API is this?
It is a REST-style HTTP API using JSON, with OpenAI-style endpoint naming. An **endpoint** is an address inside an API, not a different competing kind of API. The main endpoint is `POST /v1/chat/completions`.

Endpoints:
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `GET /docs` for interactive FastAPI documentation

The process loads CriderGPT 2.1 Nova once at startup and keeps it in RAM/VRAM.

## Windows local test
After Nova training completes:

```cmd
set CRIDERGPT_API_KEY=dev-secret
.venv\Scripts\python.exe -m uvicorn cridergpt_api.app:app --host 127.0.0.1 --port 8000
```

Then browse locally to `http://127.0.0.1:8000/docs`.

PowerShell request example:

```powershell
$headers = @{ Authorization = "Bearer dev-secret" }
$body = @{
  model = "cridergpt-2.1-nova"
  messages = @(@{ role = "user"; content = "How do I check disk space on Linux?" })
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/chat/completions" -Headers $headers -ContentType "application/json" -Body $body
```

## Production layout

```
Web / mobile client
        |
      HTTPS
        |
public CriderGPT backend / reverse proxy
        |
 private localhost HTTP
        |
CriderGPT Native API :8000
        |
CriderGPT 2.1 Nova
```

The included systemd unit assumes `/opt/cridergpt-engine`, a dedicated `cridergpt` Linux account, and one Uvicorn worker. Keep the inference API on `127.0.0.1`; do not expose the raw model port directly to the Internet. Put authentication/public HTTPS at the backend or reverse-proxy boundary as well.

Never put API keys in training data or public frontend code.

`stream: true` already returns an SSE-compatible response envelope. This first implementation emits the completed generation as a stream chunk; true token-by-token generation can be added later without changing the public endpoint.
