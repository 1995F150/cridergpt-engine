#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="/opt/cridergpt-engine"
ENV_FILE="$DEST_DIR/.env"
VENV_DIR="$DEST_DIR/venv"
REQUIREMENTS_FILE="$DEST_DIR/requirements.txt"
REQUIREMENTS_STATE="$DEST_DIR/data/requirements.sha256"
PUBLIC_HEALTH_URL="https://engine.cridergpt.com/health"

if [ ! -f "$ENV_FILE" ]; then
  echo "Refusing to update: $ENV_FILE is missing. The updater will not create or replace it."
  exit 1
fi

if [ ! -x "$VENV_DIR/bin/python" ] || [ ! -x "$VENV_DIR/bin/uvicorn" ]; then
  echo "Refusing to update: the existing virtual environment is missing or incomplete at $VENV_DIR."
  echo "The updater will not recreate or replace it automatically."
  exit 1
fi

ENV_CHECKSUM_BEFORE="$(sha256sum "$ENV_FILE" | awk '{print $1}')"
VENV_PYTHON_BEFORE="$(readlink -f "$VENV_DIR/bin/python")"
REQUIREMENTS_BEFORE=""
if [ -f "$REQUIREMENTS_FILE" ]; then
  REQUIREMENTS_BEFORE="$(sha256sum "$REQUIREMENTS_FILE" | awk '{print $1}')"
fi

sudo -u cridergpt git -C "$DEST_DIR" pull --ff-only

ENV_CHECKSUM_AFTER="$(sha256sum "$ENV_FILE" | awk '{print $1}')"
if [ "$ENV_CHECKSUM_BEFORE" != "$ENV_CHECKSUM_AFTER" ]; then
  echo "Update stopped: .env changed unexpectedly. The service was not restarted."
  exit 1
fi

VENV_PYTHON_AFTER="$(readlink -f "$VENV_DIR/bin/python")"
if [ "$VENV_PYTHON_BEFORE" != "$VENV_PYTHON_AFTER" ]; then
  echo "Update stopped: the virtual environment path changed unexpectedly."
  exit 1
fi

REQUIREMENTS_AFTER=""
if [ -f "$REQUIREMENTS_FILE" ]; then
  REQUIREMENTS_AFTER="$(sha256sum "$REQUIREMENTS_FILE" | awk '{print $1}')"
fi

# Normal engine code updates must not alter the working virtual environment.
# If dependencies change, stop and require an explicit, separately reviewed
# dependency upgrade instead of silently changing the known-good environment.
if [ -n "$REQUIREMENTS_BEFORE" ] && [ "$REQUIREMENTS_BEFORE" != "$REQUIREMENTS_AFTER" ]; then
  echo "Update stopped: requirements.txt changed."
  echo "The existing virtual environment was left untouched."
  echo "Review and install dependency changes separately before restarting the engine."
  exit 1
fi

mkdir -p "$DEST_DIR/data"
printf '%s\n' "$REQUIREMENTS_AFTER" | sudo -u cridergpt tee "$REQUIREMENTS_STATE" >/dev/null

sudo install -o root -g root -m 0644 \
  "$DEST_DIR/deployment/cridergpt-engine.service" \
  "$DEST_DIR/deployment/cridergpt-engine-update.service" \
  "$DEST_DIR/deployment/cridergpt-engine-update.timer" \
  /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cridergpt-engine-update.timer
sudo systemctl restart cridergpt-engine.service

echo "Checking the local engine..."
LOCAL_HEALTHY=false
for _attempt in $(seq 1 15); do
  if curl --fail --silent --show-error --max-time 5 \
    http://127.0.0.1:8000/health; then
    LOCAL_HEALTHY=true
    break
  fi
  sleep 2
done
echo
if [ "$LOCAL_HEALTHY" != "true" ]; then
  echo "Update failed: the engine did not become healthy after restart." >&2
  sudo journalctl -u cridergpt-engine.service -n 50 --no-pager >&2
  exit 1
fi

# The public check is intentionally non-fatal. DNS and TLS are managed outside
# this repository, and a temporary public routing issue must not roll back or
# stop an otherwise healthy local engine after an update.
echo "Checking the public Supabase engine origin: $PUBLIC_HEALTH_URL"
if curl --fail --silent --show-error --connect-timeout 10 --max-time 20 \
  "$PUBLIC_HEALTH_URL" >/dev/null; then
  echo "Public engine endpoint is reachable."
else
  echo "WARNING: Local engine is healthy, but $PUBLIC_HEALTH_URL is not reachable." >&2
  echo "Check the engine.cridergpt.com DNS record, HTTPS certificate, and Nginx/Cloudflare routing." >&2
fi

echo "Update complete. Existing .env and virtual environment were preserved."
