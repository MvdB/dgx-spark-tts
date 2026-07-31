#!/usr/bin/env bash
# Kandidat fuer einen dritten, unabhaengigen STT-Judge (Port 8008):
# Voxtral-Mini-4B-Realtime-2602. Apache-2.0, laut Modellkarte auf Fleurs
# Deutsch konkurrenzfaehig zu Offline-Modellen.
#
# ERGEBNIS 2026-07-31: ALS JUDGE UNBRAUCHBAR. Auf dem Kalibrier-Audio
# (results/judge_calib_qwen_v2, alle drei Judges auf demselben Audio):
#
#   Judge                  Ziffernquote   WER    Wortanteil
#   whisper-large-v3           0.278     0.154      0.99
#   voxtral-mini-3b            0.667     0.208      0.94
#   voxtral-realtime-4b        0.833     0.231      0.97
#
# Es verliert keinen Inhalt (Wortanteil 0.97, anders als seinerzeit granite),
# schreibt aber gesprochene Zahlwoerter in 83 % der normalization-Faelle in
# Ziffern zurueck — "neun bis zehn Uhr dreissig" wird "9 bis 10.30 Uhr",
# "Paragraf zwoelf" wird "§ 12". Genau diese Faelle sind 18 der 43 Testfaelle,
# ein solcher Judge wertet eine falsche Verbalisierung als Treffer. Der
# Verbatim-Prompt hilft nicht: die Voxtral-Familie ignoriert ihn, waehrend er
# bei Whisper die Ziffernquote von 0.833 auf 0.278 druckt.
#
# Das Skript bleibt fuer die Nachvollziehbarkeit hier (und weil die beiden
# --hf-overrides-Kniffe unten sonst wieder gesucht werden muessten).
#
# Zwei Eigenheiten gegenueber den anderen Judges:
#
# 1. Die config.json nennt die Architektur "VoxtralRealtimeForConditionalGeneration",
#    vLLM 0.25.1 registriert sie unter dem aelteren Namen "VoxtralRealtimeGeneration"
#    (dieselbe Klasse, spaeter umbenannt). Ohne --hf-overrides bricht der Start
#    mit "architecture not supported" ab. Der lokale vllm_profile.conf des
#    Modells behauptet aus demselben Grund, es sei gar nicht servierbar.
# 2. Die Modellkarte bewirbt /v1/realtime (Websocket-Streaming). Fuer uns
#    zaehlt aber, dass die Klasse SupportsTranscription mitbringt — der
#    normale /v1/audio/transcriptions-Endpunkt funktioniert, und nur der
#    darf fuer einen Judge benutzt werden (s. CLAUDE.md).
#
# enforce_eager wie bei Voxtral-TTS: auf GB10 haben CUDA-Graphen bei dieser
# Modellfamilie schon Audio korrumpiert — bei einem Judge waere ein solcher
# Fehler nicht als solcher erkennbar.
set -euo pipefail

HF_MODELS_DIR="${HF_MODELS_DIR:-$HOME/hf_models}"
CONTAINER_NAME="${CONTAINER_NAME:-voxtral-realtime-judge}"
HOST_PORT="${HOST_PORT:-8008}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.25.1}"
GPU_UTIL="${GPU_UTIL:-0.18}"
# Ein Text-Token entspricht 80 ms Audio; die Eval-Clips sind < 60 s.
MAX_LEN="${MAX_LEN:-8192}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' existiert -> wird entfernt."
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

docker run -d --name "${CONTAINER_NAME}" \
  --gpus all --shm-size=4g \
  -p "${HOST_PORT}:${HOST_PORT}" \
  -v "${HF_MODELS_DIR}:/hf_models:ro" \
  -e VLLM_DISABLE_COMPILE_CACHE=1 \
  --entrypoint bash "${IMAGE}" \
  -c "pip install --quiet 'vllm[audio]' && exec vllm serve \
        /hf_models/mistralai--Voxtral-Mini-4B-Realtime-2602 \
        --served-model-name voxtral-realtime-4b --port ${HOST_PORT} \
        --tokenizer-mode mistral \
        --hf-overrides '{\"architectures\": [\"VoxtralRealtimeGeneration\"]}' \
        --gpu-memory-utilization ${GPU_UTIL} --max-model-len ${MAX_LEN} \
        --max-num-seqs 4 --enforce-eager"

echo "Gestartet: http://0.0.0.0:${HOST_PORT}"
echo "  Logs : docker logs -f ${CONTAINER_NAME}"
echo "  Test : curl -s http://127.0.0.1:${HOST_PORT}/v1/models"
