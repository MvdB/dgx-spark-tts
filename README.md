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
| [ResembleAI Chatterbox Multilingual V3](https://github.com/resemble-ai/chatterbox) | `server_chatterbox.py` | 8003 | MIT | Zero-shot voice cloning from `/voices/<name>.wav`; generated audio carries a Perth watermark |
| [openbmb/VoxCPM2](https://github.com/OpenBMB/VoxCPM) | `server_voxcpm.py` | 8004 | Apache-2.0 | No fixed speakers; voice via description prefix (`voice=design`) |
| [mistralai/Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) | — (vLLM-Omni serves the OpenAI API natively) | 8005 | CC BY-NC 4.0 | **Non-commercial license.** 20 preset voices incl. `de_female`/`de_male`; voice cloning impossible (audio-encoder weights not released). See serving notes below — the stable vllm-omni is broken for this model |

All adapters expose the same OpenAI-compatible API, so the evaluator only
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
# VoiceDesign voices are instruct texts, selected by name via the voice field:
#   curl … -d '{"input":"…","voice":"de_male_news","language":"de"}'
# Presets: de_female_news, de_male_news, de_female_calm, de_male_young.
# QWEN_TTS_VOICE_DESIGN
# sets the default, QWEN_TTS_VOICE_INSTRUCT overrides it with free text.

curl -s http://127.0.0.1:8002/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "Guten Morgen!", "voice": "serena", "language": "de"}' \
  -o hallo.wav

# Voxtral-4B-TTS (vLLM-Omni, kein eigener Adapter):
docker build -t spark-voxtral-tts:v1 -f Dockerfile.voxtral .
./run_voxtral_tts.sh                                  # port 8005
```

Voxtral serving notes (learned the hard way, July 2026):

- vllm-omni **0.24.0 (latest stable) is broken** for Voxtral TTS: text
  conditioning is lost, the model babbles fluently in the voice's language
  while ignoring the input. Works from **0.25.0rc1** on the `v0.25.1` vLLM
  base — both must match in minor version.
- On GB10 (sm_120), CUDA graph capture corrupts the audio; the vendored
  `voxtral_tts_stages.yaml` therefore forces `enforce_eager` (RTF ~4).
- The native endpoint requires `model` in the payload and returns no timing
  headers — `roundtrip_eval.py` detects this via `/v1/models` and falls back
  to WAV-length/wall-time.
- ⚠️ **Judge caveat, learned the painful way**: Voxtral's numbers used to look
  bad because the *judge* was wrong, not the audio. granite-speech-4.1-2b
  drops number words, which cost Voxtral roughly 0.02 WER and triggered a
  since-retracted upstream report
  ([vllm-omni#5510](https://github.com/vllm-project/vllm-omni/issues/5510)).
  The judge has been replaced by whisper-large-v3 (see `eval/` below); under
  it Voxtral `de_female` moved from 0.177 into the leading group at 0.158.
  Never verify a categorical TTS finding with a single STT model.

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
TTS adapter → WAV → whisper-large-v3 (vLLM) → WER/CER vs. refs
```

The judge is **whisper-large-v3**, served by `serving/run_whisper_judge.sh`
and queried through `/v1/audio/transcriptions` with a German initial prompt
that pushes it toward spelled-out number words. It replaced
granite-speech-4.1-2b in July 2026 — see the judge calibration below.

```bash
python eval/roundtrip_eval.py \
  --testset testset/german_tts_v1.jsonl \
  --tts http://127.0.0.1:8002 \
  --stt http://127.0.0.1:8007 \
  --voice uncle_fu \
  --repeats 3 \
  --out results/$(date +%Y-%m-%d)_qwen-unclefu_n3
```

`--repeats 3` synthesizes each case three times (Magpie and Qwen sample
stochastically; single runs swing small categories by ±0.1 WER). Per case the
mean plus `wer_min`/`wer_max` are reported, the summary adds `wer_best_mean`.
`--category` restricts to selected categories.

Even at `--repeats 3` the overall mean still moves by about **0.02 WER between
runs** — measured directly by running the same Qwen VoiceDesign voice twice
with a byte-identical instruct: 0.182 and 0.160. Treat differences below that
as noise rather than a ranking.

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

WER is unbounded above, and a single degenerate transcript can dominate the
mean: the judge occasionally enters a repetition loop (1500+ characters of
"null. null. …" for 3.7 s of audio — physically impossible speech), and
VoxCPM2 occasionally produces off-text loop babble (a real TTS failure).
The evaluator therefore (a) detects implausible transcript-length/audio-
duration ratios and retries the transcription once with light sampling
(`temperature 0.3` breaks the deterministic loop), flagging persistent cases
as `asr_runaway`, and (b) reports `wer_capped_mean` alongside the raw mean,
capping each repeat at 1.0 (total substitution). The capped WER is the
headline metric on the docs pages; the gap between capped and raw shows how
much individual repeats derailed.

## docs/ — published comparison

`docs/` holds the static comparison pages (GitHub-Pages-ready: Settings →
Pages → branch `main`, folder `/docs`): `index.html` with the metric table,
plus **one listening page per model/voice combination** with all 43 clips.
`results/` is scanned automatically; per combination the newest complete run
wins — a new run of the same model/voice **overwrites** its existing page,
pages for vanished combinations are pruned. Partial runs (smoke, `--limit`,
`--category`) are skipped. Per case exactly **one** clip is published
(repeat r0, MP3 ~64 kbps mono) — representative, not cherry-picked; the raw
WAV runs stay local in `results/` (gitignored).

Regenerate after new runs:

```bash
pip install soundfile   # needs libsndfile >= 1.2 for MP3
python eval/make_docs.py
```

## License notes

Model licenses as declared upstream: MagpieTTS — NVIDIA Open Model License;
Qwen3-TTS — Apache-2.0; Chatterbox — MIT (generated audio carries a
Perth watermark); VoxCPM2 — Apache-2.0 (upstream forbids impersonation,
fraud, disinformation and recommends marking AI-generated content — the
docs pages do); Voxtral TTS — **CC BY-NC 4.0, non-commercial**. All audio
in `docs/` is AI-generated. These notes are pointers, not legal advice —
the upstream license texts are authoritative.

## Requirements

- Docker with GPU support; NGC base images `nvcr.io/nvidia/nemo:26.06`
  (Magpie) and `nvcr.io/nvidia/pytorch:26.06-py3` (Qwen3-TTS), both multi-arch
- A whisper-large-v3 STT endpoint for evaluation (`serving/run_whisper_judge.sh`);
  optionally a second judge (Voxtral-Mini-3B) for cross-validation
- Model store at `~/hf_models/` populated by dgx-spark-core's `hf-sync`
