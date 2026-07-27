#!/usr/bin/env python3
"""OpenAI-kompatibler TTS-Server für Qwen3-TTS (CustomVoice) auf DGX Spark.

Gleiche API wie server.py (Magpie), damit eval/roundtrip_eval.py unverändert
läuft — nur --tts-URL wechselt:

  POST /v1/audio/speech   – {input, voice, language, instruct?}
  GET  /v1/voices         – Sprecher + Sprachen aus dem Modell
  GET  /health

Antwort: WAV mono 16 bit, native Samplerate des Modells.
"""

from __future__ import annotations

import io
import logging
import os
import time
import wave

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

log = logging.getLogger("qwen3tts-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_PATH = os.environ.get(
    "QWEN_TTS_PATH", "/hf_models/Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice"
)
ATTN_IMPL = os.environ.get("QWEN_TTS_ATTN", "sdpa")  # flash_attention_2 wenn verfügbar

# VoiceDesign-Modelle haben keine festen Sprecher — die Stimme wird per
# instruct-Beschreibung entworfen. Default: deutsche Muttersprachlerin.
VOICE_DESIGN_MODE = "voicedesign" in MODEL_PATH.lower().replace("-", "")
DEFAULT_VOICE_INSTRUCT = os.environ.get(
    "QWEN_TTS_VOICE_INSTRUCT",
    "Klare, professionelle deutsche Frauenstimme mittleren Alters. "
    "Muttersprachliches Hochdeutsch, neutrale Nachrichtensprecher-Intonation, "
    "praezise Artikulation.",
)

# ISO-Kürzel -> Qwen-Sprachname (Auszug; get_supported_languages liefert Namen)
LANG_MAP = {
    "de": "German", "en": "English", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "fr": "French", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "auto": "Auto",
}

app = FastAPI(title="qwen3-tts", version="0.1.0")
model = None
speakers: list[str] = []
languages: list[str] = []
sample_rate = 24000


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=4096)
    voice: str = "serena"
    language: str = "de"
    instruct: str | None = None
    response_format: str = "wav"
    model: str | None = None    # OpenAI-Kompat., ignoriert
    speed: float | None = None  # nicht unterstützt


@app.on_event("startup")
def load_model() -> None:
    global model, speakers, languages
    from qwen_tts import Qwen3TTSModel

    t0 = time.time()
    log.info("Lade Qwen3-TTS aus %s (attn=%s, voice_design=%s) ...",
             MODEL_PATH, ATTN_IMPL, VOICE_DESIGN_MODE)
    model = Qwen3TTSModel.from_pretrained(
        MODEL_PATH, device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPL,
    )
    if VOICE_DESIGN_MODE:
        speakers = ["design"]  # Pseudo-Stimme; tatsächliche Stimme kommt aus instruct
    else:
        speakers = [s.lower() for s in model.get_supported_speakers()]
    languages = list(model.get_supported_languages())
    log.info("Geladen in %.1fs. Sprecher: %s | Sprachen: %s",
             time.time() - t0, speakers, languages)


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if model is not None else "loading", "model": MODEL_PATH}


@app.get("/v1/voices")
def voices() -> dict:
    return {"voices": speakers, "languages": languages}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest) -> Response:
    if model is None:
        raise HTTPException(503, "Modell lädt noch")
    if req.response_format != "wav":
        raise HTTPException(400, f"Nur 'wav' unterstützt, nicht '{req.response_format}'")
    voice = req.voice.lower()
    if not VOICE_DESIGN_MODE and voice not in speakers:
        raise HTTPException(400, f"Unbekannte Stimme '{req.voice}'. Verfügbar: {speakers}")
    lang = LANG_MAP.get(req.language.lower(), req.language)

    t0 = time.time()
    with torch.inference_mode():
        if VOICE_DESIGN_MODE:
            wavs, sr = model.generate_voice_design(
                text=req.input, language=lang,
                instruct=req.instruct or DEFAULT_VOICE_INSTRUCT,
            )
        else:
            kwargs = {"instruct": req.instruct} if req.instruct else {}
            wavs, sr = model.generate_custom_voice(
                text=req.input, language=lang, speaker=voice.capitalize(), **kwargs
            )
    wall = time.time() - t0

    pcm = np.asarray(wavs[0], dtype=np.float32)
    pcm = np.clip(pcm, -1.0, 1.0)
    duration = len(pcm) / sr
    log.info("synthesize: %d Zeichen -> %.2fs Audio in %.2fs (RTF %.2f, voice=%s, lang=%s)",
             len(req.input), duration, wall, wall / max(duration, 1e-6), voice, lang)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((pcm * 32767).astype(np.int16).tobytes())
    return Response(
        content=buf.getvalue(), media_type="audio/wav",
        headers={"X-Audio-Duration": f"{duration:.3f}",
                 "X-Synthesis-Time": f"{wall:.3f}"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8002")))
