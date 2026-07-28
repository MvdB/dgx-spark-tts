#!/usr/bin/env python3
"""Rescoring vorhandener Eval-Läufe mit einem zweiten STT-Judge.

Hintergrund (2026-07-28): granite-speech-4.1-2b verschluckt bei Voxtral-TTS-
Audio systematisch Zahlwörter — die publizierten WER-Werte enthalten also
judge-spezifische Fehler, die je TTS-Modell unterschiedlich groß sein können.
Dieses Skript transkribiert die vorhandenen r0-WAVs mit einem weiteren Judge
und rechnet beide Judges mit identischem Protokoll: beste WER über
refs ∪ {normalisierter Originaltext}. Der Text-Zusatz ist nötig, weil ein
Judge, der Ziffern schreibt ("1. Juli 2026"), sonst gegen die verbalisierten
refs verliert, obwohl die TTS-Ausgabe korrekt war.

Aufruf:
  python eval/rescore_with_judge.py --stt2 http://127.0.0.1:8006 \
      results/2026-07-28_voxtral-4b-de_female_n3 [weitere ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from roundtrip_eval import normalize, stt_model_id, wer  # noqa: E402


def transcribe_file(stt_url: str, model_id: str, wav_path: Path) -> str:
    with wav_path.open("rb") as f:
        r = requests.post(
            f"{stt_url}/v1/audio/transcriptions",
            files={"file": (wav_path.name, f, "audio/wav")},
            data={"model": model_id, "language": "de"},
            timeout=300,
        )
    r.raise_for_status()
    return r.json()["text"].strip()


def best_wer(transcript: str, refs: list[str], text: str) -> float:
    hyp = normalize(transcript)
    return min(wer(hyp, normalize(r)) for r in refs + [text])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--stt2", required=True, help="URL des Zweit-Judges")
    args = ap.parse_args()

    model2 = stt_model_id(args.stt2)
    print(f"Zweit-Judge: {model2}")

    for d in args.run_dirs:
        res_dir = Path(d)
        rows = [json.loads(l) for l in
                (res_dir / "results_raw.jsonl").read_text(encoding="utf-8").splitlines()]
        out_rows, cat = [], {}
        for r in rows:
            if r.get("error"):
                continue
            wav = res_dir / "audio" / f"{r['id']}_r0.wav"
            if not wav.exists():
                wav = res_dir / "audio" / f"{r['id']}.wav"
            # refs stehen nicht in results_raw; die kommen unten aus dem Testset
            out_rows.append((r, wav))
        testset = {}
        summary = json.loads((res_dir / "summary.json").read_text(encoding="utf-8"))
        for l in Path(summary["testset"]).read_text(encoding="utf-8").splitlines():
            c = json.loads(l)
            testset[c["id"]] = c

        recs, w1s, w2s = [], [], []
        for r, wav in out_rows:
            c = testset[r["id"]]
            # granite-r0-Transkript liegt schon vor (altes Schema: top-level)
            t1 = (r.get("repeats") or [r])[0].get("transcript", "")
            t2 = transcribe_file(args.stt2, model2, wav)
            w1 = best_wer(t1, c["refs"], c["text"])
            w2 = best_wer(t2, c["refs"], c["text"])
            w1s.append(w1)
            w2s.append(w2)
            cat.setdefault(c["category"], []).append((w1, w2))
            recs.append({"id": r["id"], "category": c["category"], "text": c["text"],
                         "judge1_transcript": t1, "judge2_transcript": t2,
                         "wer_judge1": round(w1, 4), "wer_judge2": round(w2, 4)})
        out = {
            "run": res_dir.name,
            "judge1": summary.get("stt_model"),
            "judge2": model2,
            "protocol": "best WER over refs + normalized original text, repeat r0 only",
            "wer_judge1_mean": round(sum(w1s) / len(w1s), 4),
            "wer_judge2_mean": round(sum(w2s) / len(w2s), 4),
            "wer_by_category": {
                k: {"judge1": round(sum(a for a, _ in v) / len(v), 4),
                    "judge2": round(sum(b for _, b in v) / len(v), 4)}
                for k, v in sorted(cat.items())},
            "cases": recs,
        }
        (res_dir / "rescore_judge2.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{res_dir.name}: granite {out['wer_judge1_mean']} | "
              f"{model2} {out['wer_judge2_mean']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
