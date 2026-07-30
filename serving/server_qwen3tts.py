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
# instruct-Beschreibung entworfen. Damit ein Lauf reproduzierbar und in der
# Auswertung unterscheidbar bleibt, gibt es benannte Presets: das voice-Feld
# waehlt das Preset, ein freies instruct im Request sticht es weiterhin.
# (Die Beschreibungen stehen bewusst auf Deutsch — die Modellkarte gibt die
# instruct-Beispiele in der Zielsprache an — und mit echten Umlauten: ASCII-
# Umschreibungen wie "Maennerstimme" sind fuer das Modell Fliesstext und
# schlechter interpretierbar.)
VOICE_DESIGN_MODE = "voicedesign" in MODEL_PATH.lower().replace("-", "")

VOICE_DESIGN_PRESETS = {
    "de_female_news":
        "Klare, professionelle deutsche Frauenstimme mittleren Alters. "
        "Muttersprachliches Hochdeutsch, neutrale Nachrichtensprecher-Intonation, "
        "präzise Artikulation.",
    # Nach Hörvergleich gewählt (2026-07-30): die Vorgängerfassung ("ruhig,
    # sonor, mäßiges Sprechtempo") klang gedehnt und klagend und brauchte für
    # denselben Satz 13.4 s statt 8.6 s. Die expliziten Negativ-Vorgaben am
    # Ende wirken — ohne sie kippt die Satzmelodie ins Fragende.
    "de_male_news":
        "Kräftige deutsche Männerstimme mit tiefer, ruhiger Bruststimme. "
        "Muttersprachliches Hochdeutsch, selbstbewusster und freundlicher Ton, "
        "mittleres Tempo, Satzenden fallen ab, kein Klagen und kein Fragen "
        "in der Stimme.",
    "de_female_calm":
        "Freundliche deutsche Frauenstimme, warm und ruhig. Muttersprachliches "
        "Hochdeutsch, langsames bis mittleres Sprechtempo, sehr deutliche "
        "Aussprache jeder Silbe, keine Dialektfärbung.",
    "de_male_young":
        "Jüngere deutsche Männerstimme, lebendig und zugewandt. "
        "Muttersprachliches Hochdeutsch, natürliche Sprachmelodie mit "
        "leichter Betonung, zügiges Sprechtempo, klare Endsilben.",
}
DEFAULT_DESIGN_VOICE = os.environ.get("QWEN_TTS_VOICE_DESIGN", "de_female_news")

# Freitext-Override: gewinnt gegen die Presets (Ad-hoc-Experimente).
VOICE_INSTRUCT_OVERRIDE = os.environ.get("QWEN_TTS_VOICE_INSTRUCT", "")
DEFAULT_VOICE_INSTRUCT = (
    VOICE_INSTRUCT_OVERRIDE
    or VOICE_DESIGN_PRESETS.get(DEFAULT_DESIGN_VOICE)
    or VOICE_DESIGN_PRESETS["de_female_news"]
)


def design_instruct(voice: str) -> str:
    """instruct fuer eine VoiceDesign-Stimme. 'design' bleibt als historischer
    Alias fuer die Default-Stimme erhalten (aeltere Laeufe/Skripte)."""
    if VOICE_INSTRUCT_OVERRIDE:
        return VOICE_INSTRUCT_OVERRIDE
    if voice in ("design", ""):
        return DEFAULT_VOICE_INSTRUCT
    return VOICE_DESIGN_PRESETS[voice]

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
        # Presets sind die waehlbaren "Stimmen"; 'design' bleibt als Alias.
        speakers = sorted(VOICE_DESIGN_PRESETS) + ["design"]
    else:
        speakers = [s.lower() for s in model.get_supported_speakers()]
    languages = list(model.get_supported_languages())
    log.info("Geladen in %.1fs. Sprecher: %s | Sprachen: %s",
             time.time() - t0, speakers, languages)


@app.get("/health")
def health() -> dict:
    h = {"status": "ok" if model is not None else "loading", "model": MODEL_PATH}
    if VOICE_DESIGN_MODE:
        # Die entworfene Stimme gehoert zur Konfiguration — ohne sie ist ein
        # VoiceDesign-Ergebnis nicht nachvollziehbar.
        h["voice_design_default"] = DEFAULT_DESIGN_VOICE
        h["voice_instruct_override"] = bool(VOICE_INSTRUCT_OVERRIDE)
    return h


@app.get("/v1/voices")
def voices() -> dict:
    v = {"voices": speakers, "languages": languages}
    if VOICE_DESIGN_MODE:
        # Bei VoiceDesign IST der instruct-Text die Stimme — ohne ihn ist ein
        # Ergebnis nicht reproduzierbar, also wird er mitgeliefert und vom
        # Evaluator in die summary.json geschrieben.
        v["instructs"] = {name: design_instruct(name) for name in speakers}
    return v


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest) -> Response:
    if model is None:
        raise HTTPException(503, "Modell lädt noch")
    if req.response_format != "wav":
        raise HTTPException(400, f"Nur 'wav' unterstützt, nicht '{req.response_format}'")
    voice = req.voice.lower()
    if voice not in speakers:
        raise HTTPException(400, f"Unbekannte Stimme '{req.voice}'. Verfügbar: {speakers}")
    lang = LANG_MAP.get(req.language.lower(), req.language)

    t0 = time.time()
    with torch.inference_mode():
        if VOICE_DESIGN_MODE:
            wavs, sr = model.generate_voice_design(
                text=req.input, language=lang,
                instruct=req.instruct or design_instruct(voice),
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
