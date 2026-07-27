#!/usr/bin/env bash
# Startet den Chatterbox-Container auf dem DGX Spark.
set -euo pipefail

HF_MODELS_DIR="${HF_MODELS_DIR:-$HOME/hf_models}"
CONTAINER_NAME="${CONTAINER_NAME:-chatterbox-tts}"
HOST_PORT="${HOST_PORT:-8003}"
IMAGE="${IMAGE:-spark-chatterbox:v1}"
VOICES_DIR="${VOICES_DIR:-}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' existiert -> wird entfernt."
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

EXTRA=()
[[ -n "${VOICES_DIR}" ]] && EXTRA+=(-v "${VOICES_DIR}:/voices:ro")

docker run -d --name "${CONTAINER_NAME}" \
  --gpus all \
  -p "${HOST_PORT}:8003" \
  -v "${HF_MODELS_DIR}:/hf_models:ro" \
  "${EXTRA[@]}" \
  "${IMAGE}"

echo "Gestartet: http://0.0.0.0:${HOST_PORT}"
