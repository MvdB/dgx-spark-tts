#!/usr/bin/env bash
# Nachzuegler-Lauf zum Suite-Nachtlauf 2026-07-29:
#   1. Qwen3-TTS VoiceDesign mit benannten Stimm-Presets (de_male_news neu,
#      de_female_news ersetzt die namenlose Spalte "design")
#   2. Magpie mit TN — im Nachtlauf ausgefallen, weil der OpenFst-Download
#      einen HTTP-Fehler lieferte; das Image wird hier abgewartet.
# Laeuft ohne Eingriff durch und baut am Ende die docs/ neu.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
DATE=${SUITE_DATE:-$(date +%F)}
STT=http://127.0.0.1:8000
STT2=http://127.0.0.1:8006
LOG="results/followup_${DATE}.log"
DOCS_PY=${DOCS_PY:-python3}
mkdir -p results

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

if ! curl -sf --max-time 10 "$STT/v1/models" >/dev/null; then
  log "ABBRUCH: granite-Judge auf $STT nicht erreichbar"; exit 1
fi

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
  if ! curl -sf --max-time 10 "$STT/v1/models" >/dev/null; then
    log "ABBRUCH: granite-Judge weggebrochen (vor $name)"; exit 1
  fi
  if ! wait_ready "$port"; then log "FEHLER: $name — Server auf :$port nicht bereit"; return 1; fi
  log "START $name (voice=$voice)"
  if python3 eval/roundtrip_eval.py --testset testset/german_tts_v1.jsonl \
       --tts "http://127.0.0.1:$port" --stt "$STT" --voice "$voice" \
       --repeats 3 --out "$out" >> "$LOG" 2>&1; then
    python3 eval/rescore_with_judge.py --stt2 "$STT2" "$out" >> "$LOG" 2>&1 \
      || log "WARNUNG: $name — Rescoring fehlgeschlagen"
    log "FERTIG $name: $(python3 -c "import json; s=json.load(open('$out/summary.json')); print(f\"WER {s.get('wer_capped_mean', s['wer_mean'])} (Cap) / {s['wer_mean']} (roh), ASR-Runaways: {s.get('n_asr_runaway', 0)}\")" 2>/dev/null)"
  else
    log "FEHLER: $name — Eval fehlgeschlagen"
  fi
}

# ── Qwen3-TTS VoiceDesign: Presets werden je Request gewaehlt, ein Server
#    genuegt fuer beide Stimmen ────────────────────────────────────────────
if ! wait_ready 8002; then
  log "VoiceDesign-Server nicht da — wird gestartet"
  MODEL_DIR=Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign ./serving/run_qwen3tts.sh >> "$LOG" 2>&1
fi
run_config qwen-vd-de-male-news   8002 de_male_news
run_config qwen-vd-de-female-news 8002 de_female_news
docker stop qwen3-tts >> "$LOG" 2>&1

# ── Magpie mit TN: auf den laufenden Image-Build warten ───────────────────
log "warte auf TN-Image (Build laeuft im Hintergrund, max. 90 min)"
for _ in $(seq 1 180); do
  docker image inspect spark-magpie-tts:v1-tn >/dev/null 2>&1 && break
  sleep 30
done
if docker image inspect spark-magpie-tts:v1-tn >/dev/null 2>&1; then
  IMAGE=spark-magpie-tts:v1-tn ./serving/run_server.sh >> "$LOG" 2>&1
  # Kontrolle: ohne nemo_text_processing waere apply_TN ein stiller No-op und
  # der Lauf waere eine Kopie des Laufs ohne TN.
  if wait_ready 8001 && [ "$(curl -s http://127.0.0.1:8001/health | python3 -c 'import json,sys; print(json.load(sys.stdin).get("tn"))')" = "True" ]; then
    run_config magpie-tn-sofia 8001 sofia
  else
    log "FEHLER: magpie-tn — /health meldet kein TN, Lauf waere wertlos"
  fi
  docker stop magpie-tts >> "$LOG" 2>&1
else
  log "FEHLER: TN-Image nach Wartezeit nicht da — magpie-tn uebersprungen"
fi

log "docs/ neu bauen"
"$DOCS_PY" eval/make_docs.py >> "$LOG" 2>&1 && log "DOCS REGENERIERT" \
  || log "FEHLER: make_docs fehlgeschlagen"
log "FOLLOWUP KOMPLETT"
