#!/usr/bin/env python3
"""OpenAI-kompatibler TTS-Server für ResembleAI Chatterbox Multilingual V3.

Gleiche API wie die übrigen Adapter (server.py, server_qwen3tts.py):
  POST /v1/audio/speech   – {input, voice, language}
  GET  /v1/voices
  GET  /health

Stimmen: 'default' nutzt die eingebaute Stimme; jeder andere Name lädt
/voices/<name>.wav als Referenz-Audio (Zero-Shot-Cloning). Deutsche
Referenzstimmen können so per Volume eingebunden werden.
"""

from __future__ import annotations

import io
import logging
import os
import time
import wave
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

log = logging.getLogger("chatterbox-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_PATH = os.environ.get("CHATTERBOX_PATH", "/hf_models/ResembleAI--chatterbox")
VOICES_DIR = Path(os.environ.get("CHATTERBOX_VOICES_DIR", "/voices"))

app = FastAPI(title="chatterbox-tts", version="0.1.0")
model = None


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=4096)
    voice: str = "default"
    language: str = "de"
    response_format: str = "wav"
    model: str | None = None
    speed: float | None = None


@app.on_event("startup")
def load_model() -> None:
    global model
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    t0 = time.time()
    log.info("Lade Chatterbox Multilingual V3 aus %s ...", MODEL_PATH)
    try:
        model = ChatterboxMultilingualTTS.from_local(MODEL_PATH, device="cuda", t3_model="v3")
    except (AttributeError, TypeError):
        # Ältere Paketversionen: from_local ohne t3_model-Parameter
        model = ChatterboxMultilingualTTS.from_local(MODEL_PATH, device="cuda")
    log.info("Geladen in %.1fs (sr=%s)", time.time() - t0, model.sr)


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if model is not None else "loading", "model": MODEL_PATH}


@app.get("/v1/voices")
def voices() -> dict:
    refs = sorted(p.stem for p in VOICES_DIR.glob("*.wav")) if VOICES_DIR.is_dir() else []
    return {"voices": ["default", *refs],
            "languages": sorted(getattr(model, "supported_languages", ["de", "en"]))
            if model else []}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest) -> Response:
    if model is None:
        raise HTTPException(503, "Modell lädt noch")
    if req.response_format != "wav":
        raise HTTPException(400, f"Nur 'wav' unterstützt, nicht '{req.response_format}'")

    kwargs = {"language_id": req.language}
    if req.voice != "default":
        ref = VOICES_DIR / f"{req.voice}.wav"
        if not ref.exists():
            raise HTTPException(400, f"Referenz-Audio fehlt: {ref}")
        kwargs["audio_prompt_path"] = str(ref)

    t0 = time.time()
    with torch.inference_mode():
        wav = model.generate(req.input, **kwargs)
    wall = time.time() - t0

    pcm = wav.squeeze().float().cpu().numpy()
    pcm = np.clip(pcm, -1.0, 1.0)
    sr = model.sr
    duration = len(pcm) / sr
    log.info("synthesize: %d Zeichen -> %.2fs Audio in %.2fs (RTF %.2f, voice=%s, lang=%s)",
             len(req.input), duration, wall, wall / max(duration, 1e-6), req.voice, req.language)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((pcm * 32767).astype(np.int16).tobytes())
    return Response(content=buf.getvalue(), media_type="audio/wav",
                    headers={"X-Audio-Duration": f"{duration:.3f}",
                             "X-Synthesis-Time": f"{wall:.3f}"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8003")))
