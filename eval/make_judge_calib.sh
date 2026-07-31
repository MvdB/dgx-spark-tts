#!/usr/bin/env bash
# Erzeugt das Kalibrier-Audio fuer Judge-Vergleiche neu.
#
# Warum es ein Skript braucht: results/ ist per .gitignore lokal, und beim
# Verwerfen aller Ergebnisse am 2026-07-30 ist das Kalibrier-Audio
# mitgegangen — uebrig blieben nur die Bench-Auswertungen in
# results/judge_bench_calib/, also die Zahlen ohne ihre Messgrundlage. Ein
# neuer Judge liess sich damit nicht mehr einordnen.
#
# WICHTIG: Die absoluten Werte haengen am Audio. Whisper kam auf dem alten
# Kalibrier-Audio auf WER 0.137, auf dem neuen auf 0.154 — dasselbe Modell,
# derselbe Prompt. Judge-Zahlen sind darum NUR innerhalb eines Audiosatzes
# vergleichbar. Wer einen Kandidaten misst, misst die Amtsinhaber auf
# demselben Audio mit.
#
# Der Trick des Kalibriersets: das TTS spricht die bereits ausgeschriebenen
# refs, der Audioinhalt steht also fest. Jeder Fehler ist eindeutig dem Judge
# zuzuordnen (siehe testset/judge_calib_v1.jsonl).
set -euo pipefail
cd "$(dirname "$0")/.."

TTS="${TTS:-http://127.0.0.1:8002}"
STT="${STT:-http://127.0.0.1:8007}"
VOICE="${VOICE:-de_female_news}"
OUT="${OUT:-results/judge_calib_qwen_v2}"

curl -sf --max-time 10 "$TTS/health" >/dev/null \
  || { echo "TTS auf $TTS nicht erreichbar (./serving/run_qwen3tts.sh mit MODEL_DIR=...VoiceDesign)"; exit 1; }

python3 eval/roundtrip_eval.py \
  --testset testset/judge_calib_v1.jsonl \
  --tts "$TTS" --stt "$STT" --voice "$VOICE" --repeats 1 --out "$OUT"

cat <<EOF

Kalibrier-Audio liegt in $OUT/audio.
Kandidaten und Amtsinhaber jetzt auf DIESEM Audio messen, z. B.:

  PROMPT=\$(python3 -c 'import sys;sys.path.insert(0,"eval");from roundtrip_eval import ASR_VERBATIM_PROMPT;print(ASR_VERBATIM_PROMPT)')
  python3 eval/judge_bench.py --judge http://127.0.0.1:8007 --label calib-v2-whisper-v3 \\
      --runs $OUT --category "" --endpoint transcriptions --asr-prompt "\$PROMPT" \\
      --out results/judge_bench_calib
EOF
