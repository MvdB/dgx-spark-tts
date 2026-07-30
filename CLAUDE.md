# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

TTS serving and German-language evaluation on an NVIDIA DGX Spark (GB10, **aarch64** — many x86 wheels don't exist here; this drives most Dockerfile oddities). Part of a repo family: shared infra in `dgx-spark-core` (populates the model store at `~/hf_models/` via `hf-sync`), the vLLM/STT stack in `dgx-spark-vllm`.

Code comments, docstrings, commit messages, and docs are written in **German** — keep that convention.

## Architecture

Four TTS model adapters, all exposing the **same OpenAI-compatible API** (`POST /v1/audio/speech`, `GET /v1/voices`, `GET /health`, WAV mono 16-bit out), so the evaluator only needs a different `--tts` URL:

| Adapter | Model | Port | Image |
|---|---|---|---|
| `serving/server.py` (NeMo) | nvidia/magpie_tts_multilingual_357m | 8001 | `spark-magpie-tts:v1` / `:v1-tn` |
| `serving/server_qwen3tts.py` | Qwen3-TTS-12Hz (CustomVoice or VoiceDesign via `MODEL_DIR`) | 8002 | `spark-qwen3-tts:v1` |
| `serving/server_chatterbox.py` | ResembleAI Chatterbox Multilingual V3 | 8003 | `spark-chatterbox:v1` |
| `serving/server_voxcpm.py` | openbmb/VoxCPM2 | 8004 | `spark-voxcpm:v1` |
| — (vLLM-Omni native) | mistralai/Voxtral-4B-TTS-2603 | 8005 | `spark-voxtral-tts:v1` |

Image layering matters: `Dockerfile.chatterbox` and `Dockerfile.voxcpm` build **FROM `spark-qwen3-tts:v1`** (it already contains NGC torch + source-built torchaudio); `Dockerfile.tn` builds FROM `spark-magpie-tts:v1` and adds German text normalization (pynini has no aarch64 wheel — prebuilt OpenFst/pynini artifacts are required in the build context, see its header). Without the `.tn` layer, Magpie's `apply_TN` is a **silent no-op**. Each derived Dockerfile ends its pip install with an import/CUDA guard (`torch.version.cuda`, torchaudio CUDA check, model import) — keep that when touching dependencies.

Voxtral is different: no adapter of ours — vLLM-Omni serves `/v1/audio/speech` natively (`Dockerfile.voxtral`, FROM `vllm/vllm-openai:v0.25.1` + `vllm-omni==0.25.0rc1`; the stable 0.24.0 is **broken** for this model — it ignores the input text). Two hard-won platform facts in `serving/voxtral_tts_stages.yaml`: `enforce_eager` is required on GB10 (CUDA graphs corrupt the audio), and vllm/vllm-omni must match in minor version. The evaluator auto-detects native endpoints via `/v1/models` (sends `model` field, falls back to WAV-length timing).

Models are mounted read-only from `~/hf_models` (`HF_MODELS_DIR`); no downloads at serve time (exception: Magpie fetches its NanoCodec vocoder from HF on first start).

Eval pipeline (`eval/roundtrip_eval.py`, no human in the loop):

```
TTS adapter → WAV → granite-speech-4.1-2b on vLLM (judge) → WER/CER vs. refs
```

- The judge is the granite-speech **base** model; truecasing/punctuation come only via `chat/completions` with the casing prompt — the `/v1/audio/transcriptions` default is lowercase.
- Testset `testset/german_tts_v1.jsonl`: 43 cases with categories (normalization, compound, loanword, umlaut, longform, names); each case has one or more acceptable `refs`, scoring takes the best match.
- Output convention: raw data is written **first** (`results_raw.jsonl`, `audio/*.wav`), summary generation is fail-safe afterwards (`summary.json`). Results dirs are named `results/YYYY-MM-DD_<config>_nN`.
- `results/` and `*.wav` are **gitignored by design** (raw runs stay local). The published comparison lives in `docs/` (GitHub-Pages-ready): `eval/make_docs.py` scans `results/`, keeps the newest complete run per (model, voice) — the page name derives from model+voice, so a new run **overwrites** the existing page and orphaned pages are pruned. One MP3 clip per case (repeat r0 — representative, never the best repeat). Regenerate and commit `docs/` after new runs.

## Commands

Build and run servers (from `serving/`):

```bash
# Magpie (+ German TN layer):
docker build -t spark-magpie-tts:v1 .
docker build -t spark-magpie-tts:v1-tn -f Dockerfile.tn .
IMAGE=spark-magpie-tts:v1-tn ./run_server.sh          # port 8001

# Qwen3-TTS (variant via MODEL_DIR):
docker build -t spark-qwen3-tts:v1 -f Dockerfile.qwen3tts .
./run_qwen3tts.sh                                     # port 8002
MODEL_DIR=Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign ./run_qwen3tts.sh
# VoiceDesign hat keine festen Sprecher: die Stimme ist ein instruct-Text.
# Benannte Presets (de_female_news, de_male_news, de_female_calm) werden ueber
# das normale voice-Feld gewaehlt — ein Server bedient alle. QWEN_TTS_VOICE_DESIGN
# setzt den Default, QWEN_TTS_VOICE_INSTRUCT ueberschreibt mit Freitext.
# Ohne Preset-Namen hiessen alle Laeufe "design" und kollidierten in docs/.

# Chatterbox / VoxCPM2 (require spark-qwen3-tts:v1 to exist):
docker build -t spark-chatterbox:v1 -f Dockerfile.chatterbox .
VOICES_DIR=$PWD/../voices ./run_chatterbox.sh         # port 8003; /voices/<name>.wav = zero-shot clone
docker build -t spark-voxcpm:v1 -f Dockerfile.voxcpm .
./run_voxcpm.sh                                       # port 8004

# Smoke test any adapter:
curl -s http://127.0.0.1:8002/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "Guten Morgen!", "voice": "serena", "language": "de"}' -o hallo.wav
```

Run the eval (needs a running granite-speech STT endpoint, see dgx-spark-vllm):

```bash
python eval/roundtrip_eval.py \
  --testset testset/german_tts_v1.jsonl \
  --tts http://127.0.0.1:8002 \
  --stt http://<judge-host>:8000 \
  --voice uncle_fu \
  --repeats 3 \
  --out results/$(date +%Y-%m-%d)_<config>_n3

# Quick smoke: --limit 5; restrict categories: --category normalization,umlaut
# Listening page for human spot-check:
python eval/make_listen_page.py results/<run-dir>    # → listen.html

# Regenerate the published docs/ pages (needs: pip install soundfile):
python eval/make_docs.py
```

There are no unit tests or linters; the eval run *is* the test.

## Evaluation rules (learned the hard way)

- **Never fix a voice without running the eval against it.** With Qwen3-TTS the voice shifts entire error classes (a Chinese-native voice derailed German compounds into English; an English-native voice mangled German digits).
- Use `--repeats 3`: Magpie and Qwen sample stochastically; single runs swing small categories by ±0.1 WER.
- **Run-to-run spread at n=3 is ~0.02 WER on the overall mean** — measured directly: the same Qwen VoiceDesign voice (`de_female_news`, byte-identical instruct) scored 0.182 and 0.160 in two runs. Differences below ~0.02 between configurations are noise; do not rank on them. The current top three (uncle_fu 0.151, de_male_news 0.157, de_female_news 0.160) are one indistinguishable group.
- WER is unbounded above and a single degenerate repeat can dominate the mean. Report `wer_capped_mean` (each repeat capped at 1.0); the gap to the raw mean shows how far individual repeats derailed. The judge itself loops (1500+ chars of "null. null. …" for 3.7 s of audio) — `roundtrip_eval.py` detects the implausible length/duration ratio and retries once with `temperature 0.3`, flagging survivors as `asr_runaway`. This is not cosmetic: it moved VoxCPM2 from 0.534 to 0.185, i.e. its apparent collapse was mostly judge hallucination, with one genuine loop-babble repeat left over.
- Measured WER includes STT errors (it's an upper bound of TTS error) — interpret per-category **deltas**, not absolutes. Word-level WER over-penalizes German compounds when the STT hyphenates them; check CER there.
- The testset deliberately does not measure prosody/naturalness (use `listen.html` for a human spot-check) or homograph stress.
- Cross-model comparisons are consolidated in `results/COMPARISON_*.md`.
