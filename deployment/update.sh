#!/bin/bash
set -e

DEST_DIR="/opt/cridergpt-engine"

echo "Pulling latest changes..."
cd $DEST_DIR
sudo git pull

echo "Reinstalling requirements..."
sudo ./venv/bin/pip install -r requirements.txt

echo "Restarting service..."
sudo systemctl restart cridergpt-engine.service

echo "Update complete."
