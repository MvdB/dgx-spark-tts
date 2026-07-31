#!/usr/bin/env python3
"""TTS-Roundtrip-Evaluation: Magpie-TTS → granite-speech-STT → WER/CER.

Für jeden Testfall aus dem JSONL-Testset:
  1. Text via TTS-Server synthetisieren (WAV)
  2. WAV via STT (granite-speech auf vLLM, /v1/audio/transcriptions)
     transkribieren
  3. Transkript gegen die erwarteten Verbalisierungen (refs) vergleichen –
     gewertet wird die beste (niedrigste) WER über alle refs

Rohdaten (Audio, Transkripte, Einzelscores) werden IMMER zuerst geschrieben;
die Aufbereitung (Summary/Markdown) ist fail-safe nachgelagert.

Aufruf:
  python roundtrip_eval.py --testset ../testset/german_tts_v1.jsonl \
      --tts http://127.0.0.1:8001 --stt http://127.0.0.1:8000 \
      --voice sofia --out ../results/<run-name>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import requests

# Judge-Vorauswahl, falls ein Endpunkt mehrere Modelle anbietet.
STT_MODEL_HINT = "whisper"


# ── Textnormalisierung für den Vergleich ─────────────────────────────────────

def normalize(text: str) -> str:
    """Vergleichsnormalisierung: NFC, Kleinschreibung, nur Wortzeichen."""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^a-zäöüß0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(hyp: str, ref: str) -> float:
    r = ref.split()
    return edit_distance(hyp.split(), r) / max(len(r), 1)


def cer(hyp: str, ref: str) -> float:
    return edit_distance(list(hyp.replace(" ", "")), list(ref.replace(" ", ""))) / max(
        len(ref.replace(" ", "")), 1
    )


# ── TTS / STT Clients ────────────────────────────────────────────────────────

def wav_duration(data: bytes) -> float:
    import io
    import wave

    with wave.open(io.BytesIO(data)) as w:
        return w.getnframes() / w.getframerate()


def synthesize(tts_url: str, text: str, voice: str, timeout: int = 300,
               model: str | None = None) -> tuple[bytes, dict]:
    """model wird nur bei nativen OpenAI-Endpoints gesetzt (vLLM-Omni verlangt
    das Feld; die eigenen Adapter brauchen es nicht). Timing-Header liefern
    nur die eigenen Adapter — sonst Fallback auf WAV-Länge und Wall-Zeit."""
    payload = {"input": text, "voice": voice, "language": "de"}
    if model:
        payload["model"] = model
        payload["response_format"] = "wav"
    t0 = time.time()
    r = requests.post(f"{tts_url}/v1/audio/speech", json=payload, timeout=timeout)
    wall = time.time() - t0
    r.raise_for_status()
    duration = float(r.headers.get("X-Audio-Duration", 0))
    if not duration:
        try:
            duration = round(wav_duration(r.content), 3)
        except Exception:
            duration = 0.0
    meta = {
        "audio_duration": duration,
        "synthesis_time": float(r.headers.get("X-Synthesis-Time", 0)) or round(wall, 3),
    }
    return r.content, meta


def stt_model_id(stt_url: str) -> str:
    models = requests.get(f"{stt_url}/v1/models", timeout=30).json()["data"]
    for m in models:
        if STT_MODEL_HINT in m["id"].lower():
            return m["id"]
    return models[0]["id"]


STT_PROMPT = "transcribe the speech with proper punctuation and capitalization."

# Whisper & Co. schreiben Zahlen von Haus aus als Ziffern ("17.45 Uhr") und
# machen damit unmessbar, was der Testsatz prueft: die Verbalisierung. Der
# Initial-Prompt draengt zu ausgeschriebenen Zahlwoertern. Die Beispiele sind
# bewusst NICHT aus dem Testsatz genommen — sonst souffliert man dem Judge die
# erwarteten Antworten und verdeckt echte TTS-Fehler.
ASR_VERBATIM_PROMPT = (
    "Alle Zahlen, Daten, Uhrzeiten und Abkürzungen werden als Wörter "
    "ausgeschrieben, niemals als Ziffern. Beispiele: acht neun sieben sechs, "
    "dreiundzwanzigster März neunzehnhundertachtzig, sechs Uhr zwanzig, "
    "zweiundvierzig Komma sieben, achtundneunzig Prozent, Absatz sieben."
)

# Welcher Endpunkt fuer welchen Judge funktioniert, wird einmal ermittelt und
# gemerkt — Whisper kann kein chat/completions, und ein Fehlversuch je Clip
# waere teuer.
_JUDGE_MODE: dict[str, str] = {}

# Obergrenze fuer die Judge-Generierung (s. transcribe_asr).
MAX_ASR_TOKENS = 512


def transcribe(stt_url: str, model_id: str, wav: bytes, timeout: int = 300,
               temperature: float = 0.0, prompt: str | None = None) -> str:
    """ASR via chat/completions mit Casing-Prompt (granite-speech-4.1-2b:
    Interpunktion + Truecasing gibt es nur über diesen Prompt, der
    /v1/audio/transcriptions-Default liefert lowercase)."""
    import base64

    b64 = base64.b64encode(wav).decode()
    r = requests.post(
        f"{stt_url}/v1/chat/completions",
        json={
            "model": model_id, "temperature": temperature, "max_tokens": 512,
            "messages": [{"role": "user", "content": [
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64}"}},
                {"type": "text", "text": prompt or STT_PROMPT},
            ]}],
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def transcribe_asr(stt_url: str, model_id: str, wav: bytes, timeout: int = 300,
                   temperature: float = 0.0,
                   asr_prompt: str | None = ASR_VERBATIM_PROMPT) -> str:
    """Klassischer ASR-Endpunkt mit Initial-Prompt (Whisper-Judge).

    MAX_ASR_TOKENS ist eine Notbremse, keine Messgroesse: der laengste Text im
    Testsatz hat 188 Zeichen, das Limit erlaubt gut das Zehnfache. Ohne die
    Grenze generiert ein LLM-basierter Judge in der Decoder-Schleife bis
    max_model_len weiter — Voxtral-Mini hat am 2026-07-31 fuenf Minuten lang
    mit 23 tok/s gebabbelt, bis der Client-Timeout zuschlug und das ganze
    Rescoring mitriss. Whisper begrenzt sich mit 448 Tokens je Segment selbst,
    fuer den bleibt das Limit wirkungslos."""
    r = requests.post(
        f"{stt_url}/v1/audio/transcriptions",
        files={"file": ("clip.wav", wav, "audio/wav")},
        data={"model": model_id, "language": "de",
              "temperature": temperature,
              "max_completion_tokens": MAX_ASR_TOKENS,
              **({"prompt": asr_prompt} if asr_prompt else {})},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["text"].strip()


def judge_transcribe(stt_url: str, model_id: str, wav: bytes,
                     temperature: float = 0.0) -> str:
    """Transkription ueber den ASR-Endpunkt mit Verbatim-Prompt; nur wenn ein
    Modell den nicht bedient, wird auf chat/completions ausgewichen.

    Die Reihenfolge ist bewusst so herum: Voxtral-Mini *beantwortet*
    chat/completions bereitwillig — aber als Uebersetzung ins Englische
    ("Das Geraet kostet 3,50 Euro" -> "The device cost 3,500."). Eine
    erfolgreiche Antwort ist eben kein Beweis fuer die richtige Antwort, und
    der Fehler faellt erst auf, wenn die WER unerklaerlich bei 0.85 landet.
    Der ASR-Endpunkt liefert bei allen getesteten Judges Deutsch."""
    mode = _JUDGE_MODE.get(stt_url)
    if mode is None:
        try:
            t = transcribe_asr(stt_url, model_id, wav)
            if t.strip():
                _JUDGE_MODE[stt_url] = "asr"
                return t
        except Exception:
            pass
        _JUDGE_MODE[stt_url] = mode = "chat"
    if mode == "asr":
        # temperature muss durchgereicht werden: sonst wiederholt der Retry in
        # transcribe_guarded exakt dieselbe deterministische Anfrage und bekommt
        # dieselbe Schleife zurueck. Bis 2026-07-31 war der Runaway-Retry fuer
        # den Whisper-Judge damit ein stiller No-op.
        return transcribe_asr(stt_url, model_id, wav, temperature=temperature)
    return transcribe(stt_url, model_id, wav, temperature=temperature)


def looks_runaway(transcript: str, audio_seconds: float) -> bool:
    """ASR-Decoder-Schleife: mehr Transkript, als in die Audiodauer an Sprache
    passt (Deutsch ~15 Zeichen/s inkl. Leerzeichen; Faktor 2 Toleranz).
    Beispiel aus der Praxis: granite schrieb 1538 Zeichen "null. null. …"
    zu 3.7 s Audio — physikalisch unmöglich, reine Judge-Halluzination."""
    return audio_seconds > 0 and len(transcript) > max(80.0, 30.0 * audio_seconds)


def transcribe_guarded(stt_url: str, model_id: str, wav: bytes,
                       audio_seconds: float) -> tuple[str, bool]:
    """Transkription mit Runaway-Schutz. Bei temperature 0.0 ist die Schleife
    deterministisch — ein Retry mit leichtem Sampling bricht sie meist.
    Liefert (transcript, runaway): runaway=True nur, wenn auch der Retry
    davonläuft; dann wird das kürzere Transkript behalten."""
    transcript = judge_transcribe(stt_url, model_id, wav)
    if not looks_runaway(transcript, audio_seconds):
        return transcript, False
    retry = judge_transcribe(stt_url, model_id, wav, temperature=0.3)
    if not looks_runaway(retry, audio_seconds):
        return retry, False
    return min(transcript, retry, key=len), True


# ── Hauptlauf ────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", required=True)
    ap.add_argument("--tts", default="http://127.0.0.1:8001")
    ap.add_argument("--stt", default="http://127.0.0.1:8000")
    ap.add_argument("--voice", default="sofia")
    ap.add_argument("--out", required=True, help="Ergebnisverzeichnis")
    ap.add_argument("--limit", type=int, default=0, help="Nur erste N Fälle (Smoke)")
    ap.add_argument("--category", default="", help="Nur diese Kategorie(n), kommagetrennt")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Synthesen pro Fall (Magpie sampelt stochastisch; N>=3 für stabile Zahlen)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    cases = [json.loads(l) for l in Path(args.testset).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.category:
        wanted = {c.strip() for c in args.category.split(",")}
        cases = [c for c in cases if c["category"] in wanted]
    if args.limit:
        cases = cases[: args.limit]

    model_id = stt_model_id(args.stt)
    tts_payload_model = None  # nur native OpenAI-Endpoints (vLLM-Omni) brauchen 'model'
    try:
        h = requests.get(f"{args.tts}/health", timeout=10).json()
        tts_model = h.get("model", "?")
        # Magpie mit/ohne TN-Layer ist derselbe Checkpoint — der Suffix haelt
        # die beiden Konfigurationen in (tts_model, voice)-Vergleichen getrennt.
        if h.get("tn"):
            tts_model += " +TN"
    except Exception:
        try:
            tts_payload_model = requests.get(
                f"{args.tts}/v1/models", timeout=10).json()["data"][0]["id"]
            tts_model = tts_payload_model
        except Exception:
            tts_model = "?"
    # Prompt-gesteuerte Stimmen (Qwen VoiceDesign, VoxCPM2) liefern ihren
    # instruct-Text ueber /v1/voices — ohne ihn ist der Lauf nicht
    # reproduzierbar, also wandert er in die summary.json.
    voice_instruct = None
    try:
        vv = requests.get(f"{args.tts}/v1/voices", timeout=10).json()
        voice_instruct = (vv.get("instructs") or {}).get(args.voice)
    except Exception:
        pass
    print(f"TTS: {tts_model} | STT: {model_id} | {len(cases)} Fälle, Stimme: {args.voice}")
    if voice_instruct:
        print(f"  instruct: {voice_instruct}")

    results = []
    raw_path = out / "results_raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw:
        for i, case in enumerate(cases, 1):
            row = {"id": case["id"], "category": case["category"],
                   "subcategory": case.get("subcategory"), "text": case["text"],
                   "voice": args.voice, "repeats": [], "error": None}
            try:
                for rep in range(args.repeats):
                    t0 = time.time()
                    wav, meta = synthesize(args.tts, case["text"], args.voice,
                                           model=tts_payload_model)
                    suffix = f"_r{rep}" if args.repeats > 1 else ""
                    (out / "audio" / f"{case['id']}{suffix}.wav").write_bytes(wav)
                    transcript, runaway = transcribe_guarded(
                        args.stt, model_id, wav, meta["audio_duration"])
                    hyp = normalize(transcript)
                    scored = [
                        {"ref": ref, "wer": round(wer(hyp, normalize(ref)), 4),
                         "cer": round(cer(hyp, normalize(ref)), 4)}
                        for ref in case["refs"]
                    ]
                    best = min(scored, key=lambda s: s["wer"])
                    # wer/cer bleiben ungekappt (Diagnose); *_capped begrenzt
                    # einen Ausreisser auf Totalersetzungsniveau, damit er den
                    # Mittelwert nicht dominieren kann (WER ist nach oben offen).
                    row["repeats"].append({
                        "transcript": transcript, "hyp_normalized": hyp,
                        "best_ref": best["ref"], "wer": best["wer"], "cer": best["cer"],
                        "wer_capped": min(best["wer"], 1.0),
                        "cer_capped": min(best["cer"], 1.0),
                        "asr_runaway": runaway,
                        "wall_time": round(time.time() - t0, 2), **meta})
                    if runaway:
                        print(f"    {case['id']}_r{rep}: ASR-Runaway auch nach Retry "
                              f"({len(transcript)} Zeichen / {meta['audio_duration']}s)",
                              file=sys.stderr)
                reps = row["repeats"]
                # Fall-Score: Mittel über Wiederholungen; Min separat, um
                # Modellfähigkeit von Sampling-Glück zu trennen.
                row.update(
                    wer=round(sum(r["wer"] for r in reps) / len(reps), 4),
                    cer=round(sum(r["cer"] for r in reps) / len(reps), 4),
                    wer_capped=round(sum(r["wer_capped"] for r in reps) / len(reps), 4),
                    cer_capped=round(sum(r["cer_capped"] for r in reps) / len(reps), 4),
                    wer_min=min(r["wer"] for r in reps),
                    wer_max=max(r["wer"] for r in reps),
                    transcript=reps[0]["transcript"],
                    audio_duration=reps[0]["audio_duration"],
                    synthesis_time=reps[0]["synthesis_time"])
                spread = f" (min {row['wer_min']:.2f} / max {row['wer_max']:.2f})" if args.repeats > 1 else ""
                print(f"[{i}/{len(cases)}] {case['id']}: WER {row['wer']:.2f}{spread}")
            except Exception as e:  # Rohdaten trotz Einzelfehler weiterschreiben
                row["error"] = f"{type(e).__name__}: {e}"
                print(f"[{i}/{len(cases)}] {case['id']}: FEHLER {row['error']}", file=sys.stderr)
            results.append(row)
            raw.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw.flush()

    # ── Aufbereitung: fail-safe, darf Rohdaten nie gefährden ────────────────
    try:
        ok = [r for r in results if r["error"] is None]
        by_cat: dict[str, list] = {}
        for r in ok:
            by_cat.setdefault(r["category"], []).append(r)
        summary = {
            "testset": args.testset, "voice": args.voice,
            "tts_model": tts_model, "tts_url": args.tts, "stt_model": model_id,
            # Prompts gehoeren zum Ergebnis: bei VoiceDesign/VoxCPM2 ist der
            # instruct-Text die Stimme, und der Judge-Prompt bestimmt, ob
            # ueberhaupt Interpunktion und Grossschreibung entstehen.
            "voice_instruct": voice_instruct,
            "stt_prompt": (STT_PROMPT if _JUDGE_MODE.get(args.stt) == "chat"
                           else ASR_VERBATIM_PROMPT),
            "stt_mode": _JUDGE_MODE.get(args.stt),
            "n_repeats": args.repeats,
            "n_total": len(results), "n_ok": len(ok),
            "n_error": len(results) - len(ok),
            "wer_mean": round(sum(r["wer"] for r in ok) / max(len(ok), 1), 4),
            "wer_capped_mean": round(
                sum(r["wer_capped"] for r in ok) / max(len(ok), 1), 4),
            "wer_best_mean": round(
                sum(r.get("wer_min", r["wer"]) for r in ok) / max(len(ok), 1), 4),
            "cer_mean": round(sum(r["cer"] for r in ok) / max(len(ok), 1), 4),
            "cer_capped_mean": round(
                sum(r["cer_capped"] for r in ok) / max(len(ok), 1), 4),
            "wer_by_category": {
                c: round(sum(r["wer"] for r in rs) / len(rs), 4)
                for c, rs in sorted(by_cat.items())
            },
            "wer_capped_by_category": {
                c: round(sum(r["wer_capped"] for r in rs) / len(rs), 4)
                for c, rs in sorted(by_cat.items())
            },
            "n_asr_runaway": sum(
                1 for r in ok for rep in r["repeats"] if rep.get("asr_runaway")),
            "rtf_mean": round(
                sum(r["synthesis_time"] / max(r["audio_duration"], 1e-6) for r in ok)
                / max(len(ok), 1), 3),
            "worst_cases": [
                {"id": r["id"], "wer": r["wer"], "transcript": r["transcript"]}
                for r in sorted(ok, key=lambda r: -r["wer"])[:5]
            ],
        }
        (out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Summary fehlgeschlagen (Rohdaten OK unter {raw_path}): {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
