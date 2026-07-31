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
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from roundtrip_eval import (  # noqa: E402
    normalize, stt_model_id, transcribe_guarded, wer,
)


def wav_seconds(wav_path: Path) -> float:
    try:
        with wave.open(str(wav_path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0


def transcribe_file(stt_url: str, model_id: str, wav_path: Path) -> tuple[str, bool]:
    """Zweit-Judge ueber denselben Weg wie der Haupt-Judge — inklusive
    Runaway-Schutz. Der fehlte hier bis 2026-07-31: eine Decoder-Schleife im
    Zweit-Judge lief ungebremst in den Client-Timeout und riss das komplette
    Rescoring mit, statt nur den einen Clip zu verlieren."""
    return transcribe_guarded(stt_url, model_id, wav_path.read_bytes(),
                              wav_seconds(wav_path))


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
        n_runaway, n_failed = 0, 0
        for r, wav in out_rows:
            c = testset[r["id"]]
            # r0-Transkript des Haupt-Judges liegt schon vor (altes Schema:
            # top-level statt in repeats)
            t1 = (r.get("repeats") or [r])[0].get("transcript", "")
            try:
                t2, runaway = transcribe_file(args.stt2, model2, wav)
            except Exception as e:
                # Ein kaputter Clip darf die Gegenprobe nicht komplett kosten.
                print(f"  WARNUNG: {r['id']} — Zweit-Judge fehlgeschlagen ({e})")
                n_failed += 1
                continue
            n_runaway += bool(runaway)
            # gekappt bei 1.0 wie im Haupt-Eval: ein ASR-Runaway (WER >> 1)
            # darf den Mittelwert nicht dominieren
            w1 = min(best_wer(t1, c["refs"], c["text"]), 1.0)
            w2 = min(best_wer(t2, c["refs"], c["text"]), 1.0)
            w1s.append(w1)
            w2s.append(w2)
            cat.setdefault(c["category"], []).append((w1, w2))
            recs.append({"id": r["id"], "category": c["category"], "text": c["text"],
                         "judge1_transcript": t1, "judge2_transcript": t2,
                         "wer_judge1": round(w1, 4), "wer_judge2": round(w2, 4),
                         **({"judge2_runaway": True} if runaway else {})})
        if not w1s:
            print(f"{res_dir.name}: kein einziger Clip transkribiert — nichts geschrieben")
            continue
        out = {
            "run": res_dir.name,
            "judge1": summary.get("stt_model"),
            "judge2": model2,
            "protocol": "best WER over refs + normalized original text, repeat r0 only, capped at 1.0",
            "n_cases": len(w1s),
            "n_judge2_runaway": n_runaway,
            "n_judge2_failed": n_failed,
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
        print(f"{res_dir.name}: {out['judge1']} {out['wer_judge1_mean']} | "
              f"{model2} {out['wer_judge2_mean']} "
              f"({len(w1s)} Faelle, {n_runaway} Runaways, {n_failed} Ausfaelle)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
