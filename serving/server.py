#!/usr/bin/env python3
"""OpenAI-kompatibler TTS-Server für NVIDIA MagpieTTS auf DGX Spark.

Lädt das lokale .nemo-Checkpoint (kein HF-Download zur Laufzeit) und bietet:

  POST /v1/audio/speech   – OpenAI Audio-API-kompatibel (input, voice, language)
  GET  /v1/voices         – verfügbare Stimmen + Sprachen
  GET  /health            – Liveness inkl. Modellstatus

Antwortformat ist immer WAV (22.05 kHz, mono, 16 bit) – `response_format`
wird akzeptiert, aber nur 'wav' unterstützt (400 sonst).
"""

from __future__ import annotations

import importlib.util
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

log = logging.getLogger("magpie-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_PATH = os.environ.get(
    "MAGPIE_NEMO_PATH",
    "/hf_models/nvidia--magpie_tts_multilingual_357m/magpie_tts_multilingual_357m.nemo",
)
SAMPLE_RATE = 22050

SPEAKERS = {"aria": 0, "jason": 1, "john": 2, "leo": 3, "sofia": 4}
LANGUAGES = {"ar", "de", "en", "es", "fr", "hi", "it", "ja", "ko", "pt", "vi", "zh"}

# Ohne nemo_text_processing (nur im :v1-tn-Image enthalten) ist apply_TN ein
# stiller No-op — der Health-Endpoint macht den Unterschied sichtbar, damit
# der Evaluator Laeufe mit/ohne TN auseinanderhalten kann.
HAS_TN = importlib.util.find_spec("nemo_text_processing") is not None

app = FastAPI(title="magpie-tts", version="0.1.0")
model = None


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=4096)
    voice: str = "sofia"
    language: str = "de"
    response_format: str = "wav"
    apply_tn: bool = True          # Magpie-interne Textnormalisierung
    use_cfg: bool = True           # classifier-free guidance
    model: str | None = None       # akzeptiert, ignoriert (OpenAI-Kompat.)
    speed: float | None = None     # nicht unterstützt, akzeptiert


@app.on_event("startup")
def load_model() -> None:
    global model
    from nemo.collections.tts.models import MagpieTTSModel

    t0 = time.time()
    log.info("Lade MagpieTTS aus %s ...", MODEL_PATH)
    model = MagpieTTSModel.restore_from(MODEL_PATH)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    log.info("Modell geladen in %.1fs (device=%s)", time.time() - t0, model.device)


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if model is not None else "loading", "model": MODEL_PATH, "tn": HAS_TN}


@app.get("/v1/voices")
def voices() -> dict:
    return {"voices": sorted(SPEAKERS), "languages": sorted(LANGUAGES)}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest) -> Response:
    if model is None:
        raise HTTPException(503, "Modell lädt noch")
    if req.response_format != "wav":
        raise HTTPException(400, f"Nur 'wav' unterstützt, nicht '{req.response_format}'")
    voice = req.voice.lower()
    if voice not in SPEAKERS:
        raise HTTPException(400, f"Unbekannte Stimme '{req.voice}'. Verfügbar: {sorted(SPEAKERS)}")
    if req.language not in LANGUAGES:
        raise HTTPException(400, f"Sprache '{req.language}' nicht unterstützt: {sorted(LANGUAGES)}")

    t0 = time.time()
    with torch.inference_mode():
        audio, audio_len = model.do_tts(
            req.input,
            language=req.language,
            apply_TN=req.apply_tn,
            use_cfg=req.use_cfg,
            speaker_index=SPEAKERS[voice],
        )
    wall = time.time() - t0

    pcm = audio.squeeze().float().cpu().numpy()
    pcm = np.clip(pcm, -1.0, 1.0)
    duration = len(pcm) / SAMPLE_RATE
    log.info(
        "synthesize: %d Zeichen -> %.2fs Audio in %.2fs (RTF %.2f, voice=%s, lang=%s)",
        len(req.input), duration, wall, wall / max(duration, 1e-6), voice, req.language,
    )

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((pcm * 32767).astype(np.int16).tobytes())
    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": f"{duration:.3f}",
            "X-Synthesis-Time": f"{wall:.3f}",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8001")))
