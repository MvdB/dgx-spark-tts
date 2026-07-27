# dgx-spark-tts

TTS serving and German-language evaluation on the **NVIDIA DGX Spark**
(GB10, aarch64). Part of the DGX Spark repo family — shared infrastructure
lives in [dgx-spark-core](https://github.com/MvdB/dgx-spark-core), the
LLM stack in [dgx-spark-vllm](https://github.com/MvdB/dgx-spark-vllm).

Current model: [nvidia/magpie_tts_multilingual_357m](https://huggingface.co/nvidia/magpie_tts_multilingual_357m)
(NeMo, 12 languages, 5 voices, built-in text normalization).

## serving/

FastAPI wrapper around `MagpieTTSModel` exposing an OpenAI-compatible API.
The `.nemo` checkpoint is mounted read-only from the local model store —
no downloads at serve time.

```bash
cd serving
docker build -t spark-magpie-tts:v1 .
./run_server.sh                          # port 8001

curl -s http://127.0.0.1:8001/health
curl -s http://127.0.0.1:8001/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "Guten Morgen!", "voice": "sofia", "language": "de"}' \
  -o hallo.wav
```

| Endpoint | Description |
|---|---|
| `POST /v1/audio/speech` | OpenAI Audio API compatible; returns WAV (22.05 kHz mono) |
| `GET /v1/voices` | Available voices (aria, jason, john, leo, sofia) + languages |
| `GET /health` | Liveness + model status |

Environment: `MAGPIE_NEMO_PATH` (checkpoint path inside container),
`HOST_PORT` (default 8001), `HF_MODELS_DIR` (default `~/hf_models`).

## testset/

`german_tts_v1.jsonl` — 43 German test cases targeting failure modes that
public (English-centric) benchmarks do not cover:

| Category | Cases | Probes |
|---|---|---|
| normalization | 18 | currency, dates, times, phone numbers, abbreviations (`z. B.`, `GmbH & Co. KG`), §-references, units, ordinals, Roman numerals |
| compound | 6 | long/novel compounds (Rechtsschutzversicherungsgesellschaften) |
| loanword | 6 | French/Italian loanwords, English tech terms, Denglisch code-switching |
| umlaut | 6 | minimal pairs (schon/schön, drücken/drucken, Höhle/Hölle), ß |
| longform | 4 | nested clauses, enumerations, mixed sentence modes, direct speech |
| names | 3 | German/European place names, non-German surnames |

Each case lists one or more acceptable verbalizations (`refs`); scoring takes
the best match. What this testset deliberately does **not** measure:
prosody/naturalness (use a human MOS spot-check) and homograph stress
(umfahren/umfahren — indistinguishable in a transcript).

## eval/

`roundtrip_eval.py` — automated intelligibility eval, no human in the loop:

```
Magpie TTS (:8001) → WAV → granite-speech STT (vLLM, :8000) → WER/CER vs. refs
```

```bash
python eval/roundtrip_eval.py \
  --testset testset/german_tts_v1.jsonl \
  --tts http://127.0.0.1:8001 \
  --stt http://<spark-a>:8000 \
  --voice sofia \
  --out results/$(date +%Y-%m-%d)_sofia
```

Outputs (raw data first, summary fail-safe afterwards):

- `results_raw.jsonl` — per-case transcript, WER/CER per ref, timing
- `audio/*.wav` — generated audio for manual inspection
- `summary.json` — means, per-category WER, real-time factor, worst cases

Caveat: measured WER includes STT errors (upper bound of the TTS error).
Baseline idea: pipe the same refs through STT as ground-truth audio is not
available; interpret per-category *deltas* rather than absolute values.

## Requirements

- Docker with GPU support; NGC base image `nvcr.io/nvidia/nemo:26.06` (multi-arch)
- A running granite-speech STT endpoint (see dgx-spark-vllm) for evaluation
- Model store at `~/hf_models/` populated by dgx-spark-core's `hf-sync`
