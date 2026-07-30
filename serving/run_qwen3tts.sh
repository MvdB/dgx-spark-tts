#!/usr/bin/env bash
# Startet den Qwen3-TTS-Container auf dem DGX Spark.
set -euo pipefail

HF_MODELS_DIR="${HF_MODELS_DIR:-$HOME/hf_models}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen3-tts}"
HOST_PORT="${HOST_PORT:-8002}"
IMAGE="${IMAGE:-spark-qwen3-tts:v1}"
MODEL_DIR="${MODEL_DIR:-Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' existiert -> wird entfernt."
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

# VoiceDesign: Stimme per Preset-Name (QWEN_TTS_VOICE_DESIGN) oder freiem
# instruct-Text (QWEN_TTS_VOICE_INSTRUCT, sticht das Preset). Beide muessen in
# den Container durchgereicht werden — sonst laeuft immer die eingebackene
# Default-Stimme, egal was ausserhalb gesetzt ist.
docker run -d --name "${CONTAINER_NAME}" \
  --gpus all \
  -p "${HOST_PORT}:8002" \
  -e QWEN_TTS_PATH="/hf_models/${MODEL_DIR}" \
  -e QWEN_TTS_VOICE_DESIGN="${QWEN_TTS_VOICE_DESIGN:-de_female_news}" \
  -e QWEN_TTS_VOICE_INSTRUCT="${QWEN_TTS_VOICE_INSTRUCT:-}" \
  -v "${HF_MODELS_DIR}:/hf_models:ro" \
  "${IMAGE}"

echo "Gestartet: http://0.0.0.0:${HOST_PORT}"
echo "  Logs : docker logs -f ${CONTAINER_NAME}"
echo "  Test : curl -s http://127.0.0.1:${HOST_PORT}/health"
