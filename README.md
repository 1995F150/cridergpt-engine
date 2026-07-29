# CriderGPT Engine

CriderGPT Engine is the system-level AI runtime behind CriderGPT. It runs as a FastAPI service on a Linux server and connects CriderGPT applications, Supabase Edge Functions, local Ollama models, memory, media generation, usage tracking, and administrative controls.

It is designed to run directly under `systemd`, not Docker.

## Why it exists

The web and mobile apps should not talk directly to Ollama, local image models, or files on the server. CriderGPT Engine provides one controlled API between those clients and the local AI infrastructure.

Typical production flow:

```text
CriderGPT web/mobile app
        ↓
Supabase Edge Function: chat-with-ai
        ↓
Public HTTPS reverse proxy / CriderShield
        ↓
CriderGPT Engine
        ↓
Ollama, memory, RAG, image/video services, and tools
```

The engine can also be used directly on the private network through its built-in local chat page.

## Main capabilities

- Local AI chat through Ollama.
- A browser chat interface at `/chat`.
- Persistent local chat history in SQLite, separated by user ID and conversation ID.
- Optional Supabase-backed memory and retrieval context.
- Retrieval from writing samples, AI memory, user preferences, profiles, chat messages, training inputs, and the CriderGPT training corpus.
- Bearer-token and legacy `X-API-Key` authentication.
- API key generation and management through the engine dashboard.
- Image generation and image analysis.
- Video, narration, sound-effect, and music service integration scaffolding.
- Planning, tool orchestration, response evaluation, and usage accounting.
- Health checks, configuration reloads, API documentation, and a local diagnostics dashboard.

## Built-in pages

After installation, replace `SERVER-IP` with the server's LAN or Tailscale address.

```text
http://SERVER-IP:8000/chat
http://SERVER-IP:8000/dashboard
http://SERVER-IP:8000/dashboard/keys
http://SERVER-IP:8000/docs
http://SERVER-IP:8000/api/health
```

### Local chat

The `/chat` page talks directly to the local engine. It asks for:

- User ID
- Display name
- Conversation ID

Messages are stored in:

```text
/opt/cridergpt-engine/data/local_chat.sqlite3
```

The local chat database contains user profiles and message history. Each user and conversation is isolated by its identifiers. The full history survives engine restarts and server reboots.

The local chat endpoints are:

```text
GET  /chat
GET  /local-chat/history
POST /local-chat/send
POST /local-chat/profile
```

The local browser UI is intended for private LAN or Tailscale use. Do not expose it publicly without adding a login layer or reverse-proxy access policy.

## Production API

Generation endpoints accept either:

```http
Authorization: Bearer cgpt_your_key
```

or the legacy form:

```http
X-API-Key: cgpt_your_key
```

Main endpoints:

```text
GET  /health
GET  /api/health
GET  /auth/check
GET  /api/auth/check
POST /chat
POST /api/chat
POST /chat-with-ai
POST /image/generate
POST /image/analyze
POST /video/generate
POST /config/reload
POST /api/config/reload
GET  /docs
```

Example chat request:

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

## Memory systems

CriderGPT Engine can use two memory paths.

### 1. Supabase memory

When Supabase credentials are configured, the engine can retrieve bounded context from CriderGPT tables. This supports production users signed in through the CriderGPT platform.

### 2. Local chat memory

The built-in `/chat` page uses a local SQLite database. It stores:

- User ID
- Display name
- Profile JSON
- Conversation ID
- User messages
- Assistant messages
- Timestamps

The complete API key is never stored in the local chat database.

## Required server services

1. Python 3.11 or newer.
2. Ollama listening locally, normally at `127.0.0.1:11434`.
3. The configured Ollama chat model installed, normally `llama3.1:8b`.
4. Optional vision, image, video, TTS, SFX, and music services.
5. Nginx, Cloudflare Tunnel, or CriderShield for public HTTPS access.

## Installation

```bash
sudo bash deployment/install.sh
sudo nano /opt/cridergpt-engine/.env
sudo systemctl restart cridergpt-engine
curl http://127.0.0.1:8000/health
```

The installer deliberately stops after creating a blank, mode-`0600` `.env` on the first run. Configure the file and run the installer again. Never commit actual secrets.

Common server variables:

```text
CRIDERGPT_ENGINE_API_KEY=<private engine key>
SUPABASE_URL=https://udpldrrpebdyuiqdtqnq.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<server-only value>
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_VISION_MODEL=llava:7b
IMAGE_BACKEND=automatic1111
IMAGE_API_URL=http://127.0.0.1:7860
ALLOWED_REFERENCE_HOSTS=udpldrrpebdyuiqdtqnq.supabase.co
CRIDERGPT_FOUNDER_NAME=Jessie Crider
CRIDERGPT_FOUNDER_EMAIL=jessiecrider3@gmail.com
```

Multiple legacy environment names are accepted for engine keys, but newly generated dashboard keys are stored by the engine key manager and can be used as Bearer tokens or `X-API-Key` values.

## Supabase Edge Function configuration

The `chat-with-ai` function needs a public engine origin and matching engine credential.

```text
CRIDERGPT_ENGINE_URL=https://cridergpt.com/engine/api
CRIDERGPT_ENGINE_API_KEY=<generated engine key>
```

The public reverse proxy must preserve request methods, request bodies, `Authorization`, `X-API-Key`, and `Content-Type` headers.

Example route mapping:

```text
https://cridergpt.com/engine/api/chat
        ↓
http://127.0.0.1:8000/api/chat
```

The local dashboard and local chat can remain private on Tailscale even while the API route is made public through CriderShield.

## Security notes

- Do not commit API keys, Supabase service-role credentials, or `.env` files.
- The API key manager stores hashes of generated keys, not reusable plaintext secrets.
- A generated key is shown in full only when it is created.
- Keep `/chat`, `/dashboard`, and `/dashboard/keys` private unless protected by an authentication layer.
- Use HTTPS for all public traffic.
- Keep Supabase service-role credentials only on the server and inside trusted Edge Functions.

## Validation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/ruff check .
.venv/bin/pytest -q
```

## Updating the production server

```bash
git config --global --add safe.directory /opt/cridergpt-engine
cd /opt/cridergpt-engine
sudo git pull origin main
sudo systemctl restart cridergpt-engine
sudo systemctl status cridergpt-engine --no-pager
```

Verify listening and health:

```bash
sudo ss -ltnp | grep 8000
curl http://127.0.0.1:8000/api/health
```

For private remote access through Tailscale:

```text
http://100.106.17.103:8000/chat
http://100.106.17.103:8000/dashboard
```

## Project layout

```text
api/                 FastAPI routes and authentication
engine/              Ollama inference and cognitive orchestration
memory/              Supabase context and local SQLite chat storage
usage/               Usage metering and limits
dashboard/           API-key manager assets
deployment/          systemd, installer, updater, and proxy examples
data/                 Runtime databases and generated state
app.py                Main FastAPI application
config.py             Environment and runtime configuration
```

## Intended uses

CriderGPT Engine is intended to power:

- CriderGPT web and Android applications
- Supabase `chat-with-ai`
- Private local AI chat
- Personal memory-aware assistants
- RAG over CriderGPT-owned data
- Local image and media tools
- Future agents and server automation
- API access for other CriderGPT products

It is not intended to be an unauthenticated public Ollama proxy.
