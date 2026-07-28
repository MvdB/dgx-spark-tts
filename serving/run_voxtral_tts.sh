#!/usr/bin/env bash
# Startet den Voxtral-TTS-Container (vLLM-Omni) auf dem DGX Spark.
set -euo pipefail

HF_MODELS_DIR="${HF_MODELS_DIR:-$HOME/hf_models}"
CONTAINER_NAME="${CONTAINER_NAME:-voxtral-tts}"
HOST_PORT="${HOST_PORT:-8005}"
IMAGE="${IMAGE:-spark-voxtral-tts:v1}"
MODEL_DIR="${MODEL_DIR:-mistralai--Voxtral-4B-TTS-2603}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' existiert -> wird entfernt."
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

docker run -d --name "${CONTAINER_NAME}" \
  --gpus all \
  --shm-size 8g \
  -p "${HOST_PORT}:8005" \
  -v "${HF_MODELS_DIR}:/hf_models:ro" \
  -v "${SCRIPT_DIR}/voxtral_tts_stages.yaml:/config/voxtral_tts_stages.yaml:ro" \
  "${IMAGE}" \
  "/hf_models/${MODEL_DIR}" \
  --omni \
  --stage-configs-path /config/voxtral_tts_stages.yaml \
  --served-model-name mistralai/Voxtral-4B-TTS-2603 \
  --port 8005 \
  --tokenizer-mode mistral --config-format mistral --load-format mistral

echo "Gestartet: http://0.0.0.0:${HOST_PORT}"
echo "  Logs : docker logs -f ${CONTAINER_NAME}"
echo "  Test : curl -s http://127.0.0.1:${HOST_PORT}/v1/models"
