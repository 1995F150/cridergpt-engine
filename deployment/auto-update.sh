#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="/opt/cridergpt-engine"
ENV_FILE="$DEST_DIR/.env"
SERVICE_USER="cridergpt"
REMOTE_REF="origin/main"

if [ ! -d "$DEST_DIR/.git" ]; then
  echo "Auto-update skipped: $DEST_DIR is not a Git checkout." >&2
  exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
  echo "Auto-update refused: $ENV_FILE is missing." >&2
  exit 1
fi

sudo -u "$SERVICE_USER" git -C "$DEST_DIR" fetch --quiet --prune origin main

LOCAL_COMMIT="$(sudo -u "$SERVICE_USER" git -C "$DEST_DIR" rev-parse HEAD)"
REMOTE_COMMIT="$(sudo -u "$SERVICE_USER" git -C "$DEST_DIR" rev-parse "$REMOTE_REF")"

if [ "$LOCAL_COMMIT" = "$REMOTE_COMMIT" ]; then
  echo "CriderGPT Engine is already current at ${LOCAL_COMMIT:0:12}."
  exit 0
fi

if ! sudo -u "$SERVICE_USER" git -C "$DEST_DIR" merge-base --is-ancestor \
  "$LOCAL_COMMIT" "$REMOTE_COMMIT"; then
  echo "Auto-update refused: origin/main is not a fast-forward from the deployed commit." >&2
  exit 1
fi

echo "New CriderGPT Engine release detected: ${LOCAL_COMMIT:0:12} -> ${REMOTE_COMMIT:0:12}"
exec /usr/bin/bash "$DEST_DIR/deployment/update.sh"
