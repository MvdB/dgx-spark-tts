# dgx-spark-tts

TTS serving and German-language evaluation on the **NVIDIA DGX Spark**
(GB10, aarch64). Part of the DGX Spark repo family — shared infrastructure
lives in [dgx-spark-core](https://github.com/MvdB/dgx-spark-core), the
LLM stack in [dgx-spark-vllm](https://github.com/MvdB/dgx-spark-vllm).

Supported models:

| Model | Adapter | Port | License | Notes |
|---|---|---|---|---|
| [nvidia/magpie_tts_multilingual_357m](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) | `server.py` (NeMo) | 8001 | NVIDIA Open | German TN only with the `Dockerfile.tn` layer — the NGC container ships without `nemo_text_processing`, so `apply_TN` is a silent no-op otherwise |
| [Qwen/Qwen3-TTS-12Hz-*](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | `server_qwen3tts.py` | 8002 | Apache-2.0 | CustomVoice (9 preset voices) and VoiceDesign (voice from a German `instruct` description); normalizes German numbers without external TN |

Both adapters expose the same OpenAI-compatible API, so the evaluator only
needs a different `--tts` URL. Models are mounted read-only from the local
store — no downloads at serve time (exception: Magpie fetches its NanoCodec
vocoder from HF on first start).

## serving/

```bash
cd serving

# MagpieTTS (base image, no German TN):
docker build -t spark-magpie-tts:v1 .
# + German text normalization (OpenFst/pynini aarch64 artifacts required,
#   see Dockerfile.tn header):
docker build -t spark-magpie-tts:v1-tn -f Dockerfile.tn .
IMAGE=spark-magpie-tts:v1-tn ./run_server.sh          # port 8001

# Qwen3-TTS (CustomVoice or VoiceDesign, selected via MODEL_DIR):
docker build -t spark-qwen3-tts:v1 -f Dockerfile.qwen3tts .
./run_qwen3tts.sh                                     # port 8002
MODEL_DIR=Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign ./run_qwen3tts.sh

curl -s http://127.0.0.1:8002/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "Guten Morgen!", "voice": "serena", "language": "de"}' \
  -o hallo.wav
```

| Endpoint | Description |
|---|---|
| `POST /v1/audio/speech` | OpenAI Audio API compatible; returns WAV mono 16 bit |
| `GET /v1/voices` | Voices + languages (queried from the loaded model) |
| `GET /health` | Liveness + model status |

Environment: `MAGPIE_NEMO_PATH` / `QWEN_TTS_PATH` (checkpoint path inside the
container), `MODEL_DIR` (Qwen variant), `QWEN_TTS_VOICE_INSTRUCT` (VoiceDesign
voice description, default: German newsreader), `HOST_PORT`, `HF_MODELS_DIR`.

⚠️ Voice choice is not cosmetic: with Qwen3-TTS the voice shifts entire error
classes (a Chinese-native voice derailed German compounds into English, an
English-native voice mangled German digits). Never fix a voice without
running the eval against it.

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
TTS adapter → WAV → granite-speech-4.1-2b (vLLM) → WER/CER vs. refs
```

The judge is the granite-speech **base** model (not `-plus`): truecasing and
punctuation are only produced via chat/completions with the prompt
"transcribe the speech with proper punctuation and capitalization." — the
`/v1/audio/transcriptions` default stays lowercase.

```bash
python eval/roundtrip_eval.py \
  --testset testset/german_tts_v1.jsonl \
  --tts http://127.0.0.1:8002 \
  --stt http://<judge-host>:8000 \
  --voice uncle_fu \
  --repeats 3 \
  --out results/$(date +%Y-%m-%d)_qwen-unclefu_n3
```

`--repeats 3` synthesizes each case three times (Magpie and Qwen sample
stochastically; single runs swing small categories by ±0.1 WER). Per case the
mean plus `wer_min`/`wer_max` are reported, the summary adds `wer_best_mean`.
`--category` restricts to selected categories.

Outputs (raw data first, summary fail-safe afterwards):

- `results_raw.jsonl` — per-case transcripts, WER/CER per ref and repeat
- `audio/*.wav` — generated audio for manual inspection
- `summary.json` — means, per-category WER, real-time factor, worst cases
- `listen.html` (via `eval/make_listen_page.py <results-dir>`) — audio players
  with text/transcript/WER per case, for the human MOS spot-check

Caveat: measured WER includes STT errors (upper bound of the TTS error) —
interpret per-category *deltas* rather than absolute values. Word-level WER
also over-penalizes German compounds when the STT hyphenates them; check CER
for that category.

## Requirements

- Docker with GPU support; NGC base images `nvcr.io/nvidia/nemo:26.06`
  (Magpie) and `nvcr.io/nvidia/pytorch:26.06-py3` (Qwen3-TTS), both multi-arch
- A running granite-speech STT endpoint (see dgx-spark-vllm) for evaluation
- Model store at `~/hf_models/` populated by dgx-spark-core's `hf-sync`
