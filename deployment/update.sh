#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="/opt/cridergpt-engine"
ENV_FILE="$DEST_DIR/.env"

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

sudo systemctl restart cridergpt-engine.service
curl --fail --silent http://127.0.0.1:8000/health
