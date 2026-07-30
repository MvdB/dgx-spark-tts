#!/usr/bin/env bash
# Haupt-Judge der Evaluation: Whisper large-v3 auf vLLM (Port 8007).
#
# Loest granite-speech-4.1-2b ab (Stand 2026-07-30). Begruendung, gemessen auf
# einem Kalibrierset mit bekanntem Audioinhalt (testset/judge_calib_v1.jsonl —
# ein TTS spricht die bereits ausgeschriebenen refs, der Inhalt steht also
# fest):
#
#   Judge                    Ziffernquote   WER    Wortverlust
#   whisper-v3 + Prompt          0.333     0.137      0.126
#   granite-speech-4.1-2b        0.056     0.147      0.143
#   voxtral-mini-3b              0.833     0.245      0.231
#
# granite schreibt zwar seltener Ziffern, verliert dafuer Inhalt: aus
# "siebzehn Uhr fuenfundvierzig" wurde "Der Zug faehrt um uhr", aus
# "eine Million zweihundertfuenfzigtausend Euro" ein abgebrochenes
# "belaeuft sich auf eine". Whisper hoert diese Faelle korrekt und schreibt
# lediglich manchmal Ziffern — eine Formatfrage statt Informationsverlust.
# Der Verbatim-Prompt (ASR_VERBATIM_PROMPT in eval/roundtrip_eval.py) druckt
# die Ziffernquote von 0.833 auf 0.333; seine Beispiele stammen bewusst NICHT
# aus dem Testsatz, sonst souffliert man dem Judge die erwarteten Antworten.
set -euo pipefail

HF_MODELS_DIR="${HF_MODELS_DIR:-$HOME/hf_models}"
CONTAINER_NAME="${CONTAINER_NAME:-whisper-judge}"
HOST_PORT="${HOST_PORT:-8007}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.25.1}"
GPU_UTIL="${GPU_UTIL:-0.20}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' existiert -> wird entfernt."
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

docker run -d --name "${CONTAINER_NAME}" \
  --gpus all --shm-size=4g \
  -p "${HOST_PORT}:${HOST_PORT}" \
  -v "${HF_MODELS_DIR}:/hf_models:ro" \
  --entrypoint bash "${IMAGE}" \
  -c "pip install --quiet 'vllm[audio]' && exec vllm serve \
        /hf_models/openai--whisper-large-v3 \
        --served-model-name whisper-large-v3 --port ${HOST_PORT} \
        --gpu-memory-utilization ${GPU_UTIL} --max-num-seqs 4"

echo "Gestartet: http://0.0.0.0:${HOST_PORT}"
echo "  Logs : docker logs -f ${CONTAINER_NAME}"
echo "  Test : curl -s http://127.0.0.1:${HOST_PORT}/v1/models"
