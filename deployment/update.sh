#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="/opt/cridergpt-engine"
ENV_FILE="$DEST_DIR/.env"
PUBLIC_HEALTH_URL="https://engine.cridergpt.com/health"

if [ ! -f "$ENV_FILE" ]; then
  echo "Refusing to update: $ENV_FILE is missing. The updater will not create or replace it."
  exit 1
fi

ENV_CHECKSUM_BEFORE="$(sha256sum "$ENV_FILE" | awk '{print $1}')"

sudo -u cridergpt git -C "$DEST_DIR" pull --ff-only
sudo -u cridergpt "$DEST_DIR/venv/bin/pip" install -r "$DEST_DIR/requirements.txt"

ENV_CHECKSUM_AFTER="$(sha256sum "$ENV_FILE" | awk '{print $1}')"
if [ "$ENV_CHECKSUM_BEFORE" != "$ENV_CHECKSUM_AFTER" ]; then
  echo "Update stopped: .env changed unexpectedly. The service was not restarted."
  exit 1
fi

sudo install -o root -g root -m 0644 \
  "$DEST_DIR/deployment/cridergpt-engine.service" \
  /etc/systemd/system/cridergpt-engine.service
sudo systemctl daemon-reload
sudo systemctl restart cridergpt-engine.service
echo "Checking the local engine..."
curl --fail --silent --show-error http://127.0.0.1:8000/health
echo

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

echo "Update complete. Existing .env checksum: $ENV_CHECKSUM_AFTER"
