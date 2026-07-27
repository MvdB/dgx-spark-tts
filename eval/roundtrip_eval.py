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

STT_MODEL_HINT = "granite-speech"


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

def synthesize(tts_url: str, text: str, voice: str, timeout: int = 300) -> tuple[bytes, dict]:
    r = requests.post(
        f"{tts_url}/v1/audio/speech",
        json={"input": text, "voice": voice, "language": "de"},
        timeout=timeout,
    )
    r.raise_for_status()
    meta = {
        "audio_duration": float(r.headers.get("X-Audio-Duration", 0)),
        "synthesis_time": float(r.headers.get("X-Synthesis-Time", 0)),
    }
    return r.content, meta


def stt_model_id(stt_url: str) -> str:
    models = requests.get(f"{stt_url}/v1/models", timeout=30).json()["data"]
    for m in models:
        if STT_MODEL_HINT in m["id"].lower():
            return m["id"]
    return models[0]["id"]


def transcribe(stt_url: str, model_id: str, wav: bytes, timeout: int = 300) -> str:
    """ASR über den validierten /v1/audio/transcriptions-Endpunkt."""
    r = requests.post(
        f"{stt_url}/v1/audio/transcriptions",
        files={"file": ("audio.wav", wav, "audio/wav")},
        data={"model": model_id, "language": "de", "temperature": "0.0"},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["text"].strip()


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
    print(f"STT-Modell: {model_id}, {len(cases)} Testfälle, Stimme: {args.voice}")

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
                    wav, meta = synthesize(args.tts, case["text"], args.voice)
                    suffix = f"_r{rep}" if args.repeats > 1 else ""
                    (out / "audio" / f"{case['id']}{suffix}.wav").write_bytes(wav)
                    transcript = transcribe(args.stt, model_id, wav)
                    hyp = normalize(transcript)
                    scored = [
                        {"ref": ref, "wer": round(wer(hyp, normalize(ref)), 4),
                         "cer": round(cer(hyp, normalize(ref)), 4)}
                        for ref in case["refs"]
                    ]
                    best = min(scored, key=lambda s: s["wer"])
                    row["repeats"].append({
                        "transcript": transcript, "hyp_normalized": hyp,
                        "best_ref": best["ref"], "wer": best["wer"], "cer": best["cer"],
                        "wall_time": round(time.time() - t0, 2), **meta})
                reps = row["repeats"]
                # Fall-Score: Mittel über Wiederholungen; Min separat, um
                # Modellfähigkeit von Sampling-Glück zu trennen.
                row.update(
                    wer=round(sum(r["wer"] for r in reps) / len(reps), 4),
                    cer=round(sum(r["cer"] for r in reps) / len(reps), 4),
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
            "testset": args.testset, "voice": args.voice, "stt_model": model_id,
            "n_repeats": args.repeats,
            "n_total": len(results), "n_ok": len(ok),
            "n_error": len(results) - len(ok),
            "wer_mean": round(sum(r["wer"] for r in ok) / max(len(ok), 1), 4),
            "wer_best_mean": round(
                sum(r.get("wer_min", r["wer"]) for r in ok) / max(len(ok), 1), 4),
            "cer_mean": round(sum(r["cer"] for r in ok) / max(len(ok), 1), 4),
            "wer_by_category": {
                c: round(sum(r["wer"] for r in rs) / len(rs), 4)
                for c, rs in sorted(by_cat.items())
            },
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
