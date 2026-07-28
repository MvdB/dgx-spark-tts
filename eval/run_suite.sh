#!/usr/bin/env bash
# Komplette Eval-Suite: alle Modell/Stimme-Konfigurationen seriell,
# je N=3 gegen den granite-Judge (Port 8000) + Rescoring mit dem
# Zweit-Judge (Port 8006). Voraussetzungen: beide Judges laufen bereits;
# es wird immer nur EIN TTS-Container gleichzeitig gestartet (Unified
# Memory). Einzelne Konfigurationen duerfen fehlschlagen (kein set -e) —
# am Ende steht eine Zusammenfassung in $LOG.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
DATE=$(date +%F)
STT=http://127.0.0.1:8000
STT2=http://127.0.0.1:8006
LOG="results/suite_${DATE}.log"
mkdir -p results

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

wait_ready() { # port
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && return 0
    curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1 && return 0
    sleep 10
  done
  return 1
}

run_config() { # name port voice
  local name=$1 port=$2 voice=$3 out="results/${DATE}_suite_${1}"
  if ! wait_ready "$port"; then log "FEHLER: $name — Server auf :$port nicht bereit, uebersprungen"; return 1; fi
  log "START $name (voice=$voice)"
  if python3 eval/roundtrip_eval.py --testset testset/german_tts_v1.jsonl \
       --tts "http://127.0.0.1:$port" --stt "$STT" --voice "$voice" \
       --repeats 3 --out "$out" >> "$LOG" 2>&1; then
    python3 eval/rescore_with_judge.py --stt2 "$STT2" "$out" >> "$LOG" 2>&1 \
      || log "WARNUNG: $name — Rescoring fehlgeschlagen"
    log "FERTIG $name: $(python3 -c "import json; s=json.load(open('$out/summary.json')); print(f\"WER {s['wer_mean']}\")" 2>/dev/null)"
  else
    log "FEHLER: $name — Eval fehlgeschlagen"
  fi
}

# TN-Image parallel bauen (wird erst fuer die letzte Konfiguration gebraucht)
log "TN-Image-Build startet im Hintergrund"
( cd serving && docker build -t spark-magpie-tts:v1-tn -f Dockerfile.tn . \
    > "$REPO/results/tn_build_${DATE}.log" 2>&1 ) &
TN_BUILD_PID=$!

# ── Qwen3-TTS CustomVoice: 3 Stimmen ueber einen Server ─────────────────────
./serving/run_qwen3tts.sh >> "$LOG" 2>&1
run_config qwen-cv-serena  8002 serena
run_config qwen-cv-aiden   8002 aiden
run_config qwen-cv-unclefu 8002 uncle_fu

# ── Qwen3-TTS VoiceDesign ───────────────────────────────────────────────────
MODEL_DIR=Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign ./serving/run_qwen3tts.sh >> "$LOG" 2>&1
run_config qwen-vd-design 8002 design
docker stop qwen3-tts >> "$LOG" 2>&1

# ── Chatterbox: default + de_f1 ─────────────────────────────────────────────
VOICES_DIR="$REPO/voices" ./serving/run_chatterbox.sh >> "$LOG" 2>&1
run_config chatterbox-de-f1   8003 de_f1
run_config chatterbox-default 8003 default
docker stop chatterbox-tts >> "$LOG" 2>&1

# ── VoxCPM2 ─────────────────────────────────────────────────────────────────
./serving/run_voxcpm.sh >> "$LOG" 2>&1
run_config voxcpm2-design 8004 design
docker stop voxcpm2 >> "$LOG" 2>&1

# ── Voxtral: beide deutschen Preset-Stimmen ─────────────────────────────────
./serving/run_voxtral_tts.sh >> "$LOG" 2>&1
run_config voxtral-de-female 8005 de_female
run_config voxtral-de-male   8005 de_male
docker stop voxtral-tts >> "$LOG" 2>&1

# ── Magpie ohne TN ──────────────────────────────────────────────────────────
./serving/run_server.sh >> "$LOG" 2>&1
run_config magpie-sofia 8001 sofia
docker stop magpie-tts >> "$LOG" 2>&1

# ── Magpie mit TN (wartet ggf. auf den Image-Build) ─────────────────────────
log "warte auf TN-Image-Build (PID $TN_BUILD_PID)"
if wait "$TN_BUILD_PID"; then
  IMAGE=spark-magpie-tts:v1-tn ./serving/run_server.sh >> "$LOG" 2>&1
  run_config magpie-tn-sofia 8001 sofia
  docker stop magpie-tts >> "$LOG" 2>&1
else
  log "FEHLER: TN-Image-Build fehlgeschlagen (s. results/tn_build_${DATE}.log) — magpie-tn uebersprungen"
fi

log "SUITE KOMPLETT"
