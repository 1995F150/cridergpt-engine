# CriderGPT Engine

Production FastAPI bridge between the CriderGPT Supabase Edge Functions and system-level AI services on the GPU server. It does not use Docker.

## What it does

- Receives the exact `/chat`, `/image/generate`, and `/image/analyze` contracts used by the `chat-with-ai` Edge Function.
- Generates chat responses through Ollama.
- Loads bounded context from `writing_samples`, `ai_memory`, `user_preferences`, `profiles`, `chat_messages`, `training_inputs`, and `cridergpt_training_corpus`.
- Keeps user memory scoped to the supplied authenticated Supabase `user_id`.
- Recognizes Jessie Crider as CriderGPT's founder while still keeping authorization separate from identity context.
- Generates images through a local Automatic1111-compatible API (Automatic1111 or Forge) and can attach trusted Supabase character-reference images.
- Analyzes images through an Ollama vision-capable model.

## Required server services

1. Python 3.11 or newer.
2. Ollama listening locally (default `127.0.0.1:11434`) with both the configured chat and vision models installed.
3. Automatic1111 or Forge started with its API enabled (default `127.0.0.1:7860`).
4. Nginx with HTTPS in front of Uvicorn.

## Installation

```bash
sudo bash deployment/install.sh
sudo nano /opt/cridergpt-engine/.env
sudo systemctl restart cridergpt-engine
curl http://127.0.0.1:8000/health
```

The installer deliberately stops the first time it creates `.env`; configure it before starting the service. Never put actual secrets in Git.

The value in `CRIDERGPT_ENGINE_API_KEY` must match the Supabase Edge Function secret with the same name. The plural `CRIDERGPT_ENGINE_API_KEYS` is also accepted for key rotation. The public HTTPS origin must be stored in the Edge Function secret `CRIDERGPT_ENGINE_URL` without an extra path suffix.

CriderGPT's production origin is:

```text
https://engine.cridergpt.com
```

Set these two Edge Function secrets in the CriderGPT Supabase project:

```text
CRIDERGPT_ENGINE_URL=https://engine.cridergpt.com
CRIDERGPT_ENGINE_API_KEY=<same private value configured on the engine server>
```

The hostname must have a public DNS record or Cloudflare Tunnel and must route
through HTTPS to Nginx, which proxies to Uvicorn at `127.0.0.1:8000`. The
repository cannot create DNS records or certificates during `update.sh`.
Install `deployment/nginx.conf.example` only after the certificate for
`engine.cridergpt.com` exists.

The `/health` route is intentionally public and contains no credentials. Chat
and image-generation routes require `X-API-Key`. Do not create an API-key
generator on the public website; generate and rotate the key on the server,
then store the matching value in Supabase Edge Function secrets.

## API

All generation routes require `X-API-Key`. `/health` is intentionally non-secret and reports dependency readiness without revealing credentials.

- `POST /chat`
- `POST /chat-with-ai` (compatibility alias)
- `POST /image/generate`
- `POST /image/analyze`
- `/api/...` compatibility aliases for older clients

Example chat body:

```json
{
  "message": "Hello",
  "system_prompt": "Context assembled by the Supabase Edge Function",
  "conversation_history": [],
  "user_id": null,
  "conversation_id": null,
  "model": null,
  "temperature": 0.7,
  "max_tokens": 2000
}
```

## Validation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/ruff check .
.venv/bin/pytest -q
```

## Updating

```bash
sudo bash /opt/cridergpt-engine/deployment/update.sh
journalctl -u cridergpt-engine -n 100 --no-pager
```

The updater never creates, replaces, edits, changes ownership of, or changes permissions on the existing `.env` file. It verifies the file checksum before restarting the service.
It also performs a required local health check and a non-fatal public HTTPS
check against `https://engine.cridergpt.com/health`. A DNS, certificate, or
proxy problem is reported clearly without taking the healthy local service
back down.
