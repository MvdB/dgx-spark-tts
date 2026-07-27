#!/usr/bin/env bash
# Startet den VoxCPM2-Container auf dem DGX Spark.
set -euo pipefail

HF_MODELS_DIR="${HF_MODELS_DIR:-$HOME/hf_models}"
CONTAINER_NAME="${CONTAINER_NAME:-voxcpm2}"
HOST_PORT="${HOST_PORT:-8004}"
IMAGE="${IMAGE:-spark-voxcpm:v1}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' existiert -> wird entfernt."
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

docker run -d --name "${CONTAINER_NAME}" \
  --gpus all \
  -p "${HOST_PORT}:8004" \
  -v "${HF_MODELS_DIR}:/hf_models:ro" \
  "${IMAGE}"

echo "Gestartet: http://0.0.0.0:${HOST_PORT}"
