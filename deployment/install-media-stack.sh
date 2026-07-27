#!/usr/bin/env bash
set -euo pipefail

# Installs host dependencies and local services for CriderGPT media generation.
# Model downloads are opt-in because they are large and may require accepting a license.

ENGINE_DIR="${ENGINE_DIR:-/opt/cridergpt-engine}"
COMFY_DIR="${COMFY_DIR:-/opt/comfyui}"
COMFY_USER="${COMFY_USER:-cridergpt}"
COMFY_HOST="${COMFY_HOST:-127.0.0.1}"
COMFY_PORT="${COMFY_PORT:-8188}"
INSTALL_LTX_NODES="${INSTALL_LTX_NODES:-1}"
ENABLE_SERVICES="${ENABLE_SERVICES:-1}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-0}"
LTX_MODEL_REPO="${LTX_MODEL_REPO:-Lightricks/LTX-Video}"

if [[ $EUID -ne 0 ]]; then
  echo "Run as root: sudo $0" >&2
  exit 1
fi

id "$COMFY_USER" >/dev/null 2>&1 || useradd --system --home /var/lib/cridergpt --shell /usr/sbin/nologin "$COMFY_USER"

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ffmpeg git python3 python3-venv python3-pip curl jq ca-certificates

command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null

if [[ ! -d "$COMFY_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
else
  git -C "$COMFY_DIR" pull --ff-only
fi

python3 -m venv "$COMFY_DIR/venv"
"$COMFY_DIR/venv/bin/python" -m pip install --upgrade pip wheel
"$COMFY_DIR/venv/bin/python" -m pip install -r "$COMFY_DIR/requirements.txt"

if [[ "$INSTALL_LTX_NODES" == "1" ]]; then
  mkdir -p "$COMFY_DIR/custom_nodes"
  if [[ ! -d "$COMFY_DIR/custom_nodes/ComfyUI-LTXVideo/.git" ]]; then
    git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git \
      "$COMFY_DIR/custom_nodes/ComfyUI-LTXVideo"
  else
    git -C "$COMFY_DIR/custom_nodes/ComfyUI-LTXVideo" pull --ff-only
  fi
  if [[ -f "$COMFY_DIR/custom_nodes/ComfyUI-LTXVideo/requirements.txt" ]]; then
    "$COMFY_DIR/venv/bin/python" -m pip install -r \
      "$COMFY_DIR/custom_nodes/ComfyUI-LTXVideo/requirements.txt"
  fi
fi

mkdir -p "$COMFY_DIR/models/checkpoints" "$COMFY_DIR/models/text_encoders" \
  "$COMFY_DIR/models/vae" "$COMFY_DIR/output" "$ENGINE_DIR/data/generated"
chown -R "$COMFY_USER:$COMFY_USER" "$COMFY_DIR" "$ENGINE_DIR/data"

if [[ "$DOWNLOAD_MODELS" == "1" ]]; then
  echo "Model download requested. A Hugging Face token and accepted model license may be required."
  "$COMFY_DIR/venv/bin/python" -m pip install --upgrade huggingface_hub
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required when DOWNLOAD_MODELS=1" >&2
    exit 1
  fi
  # Download the official Lightricks repository into an isolated cache. The operator
  # must choose model files suitable for the installed GPU and link them into ComfyUI.
  sudo -u "$COMFY_USER" env HF_TOKEN="$HF_TOKEN" \
    "$COMFY_DIR/venv/bin/huggingface-cli" download "$LTX_MODEL_REPO" \
    --local-dir "$COMFY_DIR/models/ltx-video-repository"
fi

install -o root -g root -m 0644 "$ENGINE_DIR/deployment/comfyui.service" /etc/systemd/system/comfyui.service
install -o root -g root -m 0644 "$ENGINE_DIR/deployment/cridergpt-video-worker.service" /etc/systemd/system/cridergpt-video-worker.service
systemctl daemon-reload

if [[ "$ENABLE_SERVICES" == "1" ]]; then
  systemctl enable --now comfyui.service
  systemctl enable --now cridergpt-video-worker.service
fi

cat <<EOF
Media stack installation complete.
FFmpeg: $(ffmpeg -version | head -1)
ComfyUI: http://${COMFY_HOST}:${COMFY_PORT}

Next required steps:
1. Select model files that fit the actual GPU and link them into $COMFY_DIR/models.
2. Load an official LTX workflow in ComfyUI and export it in API format.
3. Save the exported workflow as $ENGINE_DIR/video/workflows/default.json.
4. Set LOCAL_VIDEO_URL=http://${COMFY_HOST}:${COMFY_PORT} and VIDEO_BACKEND=local in $ENGINE_DIR/.env.
5. Restart cridergpt-engine and cridergpt-video-worker.
EOF
