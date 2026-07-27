#!/usr/bin/env bash
# Startet den MagpieTTS-Container auf dem DGX Spark.
set -euo pipefail

HF_MODELS_DIR="${HF_MODELS_DIR:-$HOME/hf_models}"
CONTAINER_NAME="${CONTAINER_NAME:-magpie-tts}"
HOST_PORT="${HOST_PORT:-8001}"
IMAGE="${IMAGE:-spark-magpie-tts:v1}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' existiert -> wird entfernt."
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

docker run -d --name "${CONTAINER_NAME}" \
  --gpus all \
  -p "${HOST_PORT}:8001" \
  -v "${HF_MODELS_DIR}:/hf_models:ro" \
  "${IMAGE}"

echo "Gestartet: http://0.0.0.0:${HOST_PORT}"
echo "  Logs : docker logs -f ${CONTAINER_NAME}"
echo "  Test : curl -s http://127.0.0.1:${HOST_PORT}/health"
