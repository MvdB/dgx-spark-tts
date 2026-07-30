#!/usr/bin/env python3
"""Judge-Kandidaten auf vorhandenem Audio vergleichen — ohne neue Synthese.

Hintergrund (2026-07-30): granite-speech soll als Haupt-Judge abgeloest
werden. Ein Ersatz muss die Kerneigenschaft mitbringen, die der Testsatz
braucht: **woertliche** Transkription. 18 der 43 Faelle pruefen, ob das
TTS-Modell "01.07.2026" als "erster Juli zweitausendsechsundzwanzig"
ausspricht. Ein Judge, der Zahlwoerter in Ziffern zuruecknormalisiert
("null eins null sieben" -> "01.07."), wertet eine falsche Verbalisierung
als Treffer und macht die Messung wertlos.

Gemessen wird darum je Kandidat auf denselben WAVs:

  ziffernquote  Anteil der Transkripte mit Ziffern in normalization-Faellen.
                Das Audio dieser Faelle enthaelt gesprochene Zahlwoerter —
                wer hier Ziffern schreibt, normalisiert zurueck. NIEDRIG = gut.
  runaway       Transkripte, die laenger sind, als in die Audiodauer an
                Sprache passt (Decoder-Schleife). NIEDRIG = gut.
  wer_refs      WER gegen die verbalisierten refs. Nur aussagekraeftig
                zusammen mit der Ziffernquote: ein rueckschreibender Judge
                bekommt hier kuenstlich gute Werte gegen den Originaltext.
  leer          Anteil leerer Transkripte (Modell verweigert/haengt).

Aufruf:
  python eval/judge_bench.py --judge http://127.0.0.1:8000 \
      --label granite-base --runs results/2026-07-30_suite_magpie-tn-sofia
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import wave
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from roundtrip_eval import (  # noqa: E402
    STT_PROMPT, looks_runaway, normalize, stt_model_id, transcribe, wer,
)


def wav_seconds(p: Path) -> float:
    with wave.open(str(p)) as w:
        return w.getnframes() / w.getframerate()


def transcribe_any(judge_url: str, model_id: str, wav: bytes,
                   name: str, force: str = "auto",
                   prompt: str | None = None,
                   asr_prompt: str | None = None) -> tuple[str, str]:
    """chat/completions mit Casing-Prompt bevorzugt (nur so liefert granite
    Interpunktion); Modelle ohne Audio-Chat fallen auf /v1/audio/transcriptions
    zurueck. Der benutzte Weg wird mitprotokolliert, weil er das Ausgabeformat
    beeinflusst. force='transcriptions' erzwingt den ASR-Endpunkt — noetig fuer
    Voxtral-Mini, das im Chat-Modus die Aufgabe als Uebersetzung auffasst und
    englischen Text liefert."""
    if force != "transcriptions":
        try:
            t = transcribe(judge_url, model_id, wav, prompt=prompt)
            if t.strip():
                return t, "chat"
        except Exception:
            pass
    if True:
        r = requests.post(
            f"{judge_url}/v1/audio/transcriptions",
            files={"file": (name, wav, "audio/wav")},
            data={"model": model_id, "language": "de",
                  **({"prompt": asr_prompt} if asr_prompt else {})}, timeout=300)
        r.raise_for_status()
        return r.json()["text"].strip(), "transcriptions"


def bench(judge_url: str, label: str, run_dirs: list[str],
          categories: set[str] | None, force: str = "auto",
          prompt: str | None = None, asr_prompt: str | None = None) -> dict:
    model_id = stt_model_id(judge_url)
    testset = {}
    cases = []
    for d in run_dirs:
        res = Path(d)
        summary = json.loads((res / "summary.json").read_text(encoding="utf-8"))
        if not testset:
            for line in Path(summary["testset"]).read_text(encoding="utf-8").splitlines():
                c = json.loads(line)
                testset[c["id"]] = c
        for row in (json.loads(l) for l in
                    (res / "results_raw.jsonl").read_text(encoding="utf-8").splitlines()):
            if row.get("error"):
                continue
            if categories and row["category"] not in categories:
                continue
            wav = res / "audio" / f"{row['id']}_r0.wav"
            if not wav.exists():   # --repeats 1 schreibt ohne _rN-Suffix
                wav = res / "audio" / f"{row['id']}.wav"
            if wav.exists():
                cases.append((res.name, row["id"], row["category"], wav))

    recs, t0 = [], time.time()
    for i, (run, cid, cat, wav) in enumerate(cases, 1):
        c = testset[cid]
        try:
            text, weg = transcribe_any(judge_url, model_id, wav.read_bytes(), wav.name, force, prompt, asr_prompt)
        except Exception as e:
            recs.append({"run": run, "id": cid, "category": cat,
                         "transcript": "", "error": f"{type(e).__name__}: {e}"})
            continue
        secs = wav_seconds(wav)
        hyp = normalize(text)
        recs.append({
            "run": run, "id": cid, "category": cat, "transcript": text,
            "endpoint": weg, "audio_seconds": round(secs, 2),
            "hat_ziffern": bool(re.search(r"\d", text)),
            "runaway": looks_runaway(text, secs),
            "wer_refs": round(min(wer(hyp, normalize(r)) for r in c["refs"]), 4),
            "wer_mit_text": round(min(wer(hyp, normalize(r))
                                      for r in c["refs"] + [c["text"]]), 4),
        })
        if i % 10 == 0:
            print(f"  [{label}] {i}/{len(cases)} ...", flush=True)

    ok = [r for r in recs if not r.get("error")]
    norm = [r for r in ok if r["category"] == "normalization"]
    n = max(len(ok), 1)
    return {
        "label": label, "judge_model": model_id, "judge_url": judge_url,
        "prompt": prompt or STT_PROMPT, "asr_prompt": asr_prompt, "n_clips": len(recs), "n_ok": len(ok),
        "dauer_s": round(time.time() - t0, 1),
        "ziffernquote_normalization": round(
            sum(r["hat_ziffern"] for r in norm) / max(len(norm), 1), 4),
        "ziffernquote_gesamt": round(sum(r["hat_ziffern"] for r in ok) / n, 4),
        "runaway_quote": round(sum(r["runaway"] for r in ok) / n, 4),
        "leer_quote": round(sum(1 for r in ok if not r["transcript"].strip()) / n, 4),
        "fehler": len(recs) - len(ok),
        "wer_refs_mean": round(sum(r["wer_refs"] for r in ok) / n, 4),
        "wer_mit_text_mean": round(sum(r["wer_mit_text"] for r in ok) / n, 4),
        "cases": recs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--category", default="normalization,umlaut,loanword",
                    help="leer = alle Kategorien")
    ap.add_argument("--endpoint", default="auto",
                    choices=["auto", "transcriptions"],
                    help="transcriptions erzwingt den ASR-Endpunkt (Voxtral uebersetzt im Chat)")
    ap.add_argument("--prompt", default=None,
                    help="Judge-eigener ASR-Prompt (granite-plus braucht seinen eigenen)")
    ap.add_argument("--asr-prompt", dest="asr_prompt", default=None,
                    help="Whisper-Initial-Prompt: draengt zu ausgeschriebenen Zahlwoertern")
    ap.add_argument("--out", default="results/judge_bench")
    args = ap.parse_args()

    cats = {c.strip() for c in args.category.split(",") if c.strip()} or None
    r = bench(args.judge, args.label, args.runs, cats, args.endpoint, args.prompt, args.asr_prompt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}.json").write_text(
        json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{r['label']} ({r['judge_model']}): {r['n_ok']}/{r['n_clips']} Clips in {r['dauer_s']}s")
    print(f"  Ziffernquote normalization: {r['ziffernquote_normalization']}  (niedrig = woertlich)")
    print(f"  Runaways: {r['runaway_quote']} | leer: {r['leer_quote']} | Fehler: {r['fehler']}")
    print(f"  WER gegen refs: {r['wer_refs_mean']} | refs+Text: {r['wer_mit_text_mean']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
