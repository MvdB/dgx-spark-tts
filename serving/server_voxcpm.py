#!/usr/bin/env python3
"""OpenAI-kompatibler TTS-Server für openbmb/VoxCPM2.

Gleiche API wie die übrigen Adapter:
  POST /v1/audio/speech   – {input, voice, language}
  GET  /v1/voices
  GET  /health

VoxCPM2 hat keine festen Sprecher; Voice-Design geschieht über einen
Beschreibungs-Präfix in Klammern. voice='design' setzt die deutsche
Default-Beschreibung (VOXCPM_VOICE_INSTRUCT) vor den Text, voice='default'
synthetisiert ohne Präfix. Sprache wird nicht parametrisiert (autodetekt).
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

log = logging.getLogger("voxcpm-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_PATH = os.environ.get("VOXCPM_PATH", "/hf_models/openbmb--VoxCPM2")
VOICE_INSTRUCT = os.environ.get(
    "VOXCPM_VOICE_INSTRUCT",
    "Eine klare, professionelle deutsche Frauenstimme mittleren Alters, "
    "muttersprachliches Hochdeutsch, neutrale Nachrichtensprecher-Intonation",
)
CFG_VALUE = float(os.environ.get("VOXCPM_CFG", "2.0"))
TIMESTEPS = int(os.environ.get("VOXCPM_TIMESTEPS", "10"))

app = FastAPI(title="voxcpm2", version="0.1.0")
model = None
sample_rate = 16000


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=4096)
    voice: str = "design"
    language: str = "de"   # akzeptiert, VoxCPM2 erkennt die Sprache selbst
    response_format: str = "wav"
    model: str | None = None
    speed: float | None = None


@app.on_event("startup")
def load_model() -> None:
    global model, sample_rate
    from voxcpm import VoxCPM

    t0 = time.time()
    log.info("Lade VoxCPM2 aus %s ...", MODEL_PATH)
    model = VoxCPM.from_pretrained(MODEL_PATH, load_denoiser=False)
    sample_rate = model.tts_model.sample_rate
    log.info("Geladen in %.1fs (sr=%s)", time.time() - t0, sample_rate)


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if model is not None else "loading", "model": MODEL_PATH}


@app.get("/v1/voices")
def voices() -> dict:
    # 'design' setzt VOICE_INSTRUCT als Klammer-Praefix vor den Text — der
    # Prompt gehoert zur Konfiguration und wird darum mitgeliefert.
    return {"voices": ["design", "default"], "languages": ["auto"],
            "instructs": {"design": VOICE_INSTRUCT}}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest) -> Response:
    if model is None:
        raise HTTPException(503, "Modell lädt noch")
    if req.response_format != "wav":
        raise HTTPException(400, f"Nur 'wav' unterstützt, nicht '{req.response_format}'")

    text = req.input
    if req.voice == "design":
        text = f"({VOICE_INSTRUCT}){text}"
    elif req.voice != "default":
        raise HTTPException(400, "Stimmen: 'design' (deutsche Design-Stimme) oder 'default'")

    t0 = time.time()
    with torch.inference_mode():
        wav = model.generate(text=text, cfg_value=CFG_VALUE, inference_timesteps=TIMESTEPS)
    wall = time.time() - t0

    pcm = np.asarray(wav, dtype=np.float32).squeeze()
    pcm = np.clip(pcm, -1.0, 1.0)
    duration = len(pcm) / sample_rate
    log.info("synthesize: %d Zeichen -> %.2fs Audio in %.2fs (RTF %.2f, voice=%s)",
             len(req.input), duration, wall, wall / max(duration, 1e-6), req.voice)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((pcm * 32767).astype(np.int16).tobytes())
    return Response(content=buf.getvalue(), media_type="audio/wav",
                    headers={"X-Audio-Duration": f"{duration:.3f}",
                             "X-Synthesis-Time": f"{wall:.3f}"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8004")))
