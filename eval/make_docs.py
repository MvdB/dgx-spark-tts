#!/usr/bin/env python3
"""Erzeugt die statischen Vergleichsseiten unter docs/ (GitHub-Pages-tauglich).

Aus den kuratierten Eval-Läufen (RUNS unten) entsteht:
  docs/index.html          – Modellvergleich (Metriken, Kategorien, Links)
  docs/<slug>.html         – Abhörseite pro Konfiguration (43 Clips)
  docs/audio/<slug>/*.mp3  – je Fall EIN Clip (Wiederholung r0), MP3 statt WAV,
                             damit das Repo klein bleibt (~3 KB/s statt ~44 KB/s)

Es wird bewusst nur r0 veröffentlicht (nicht der beste Repeat) — die Clips
sollen repräsentativ klingen, nicht geschönt. Die WER-Angabe pro Fall ist
der Mittelwert über alle Repeats aus results_raw.jsonl.

Abhängigkeit: soundfile (pip install soundfile; braucht libsndfile >= 1.2 für MP3).

Aufruf:  python eval/make_docs.py            # aus dem Repo-Root
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"

# Kuratierte Läufe (Spaltenreihenfolge = Reihenfolge hier).
# Bei neuen Bestläufen: run-Verzeichnis austauschen, Skript neu laufen lassen.
RUNS = [
    {
        "slug": "qwen-unclefu",
        "title": "Qwen3-TTS 1.7B CustomVoice · uncle_fu",
        "dir": "2026-07-27_qwen3tts-1.7B-unclefu_n3",
        "license": "Apache-2.0",
    },
    {
        "slug": "qwen-design-de",
        "title": "Qwen3-TTS 1.7B VoiceDesign · deutsche Beschreibung",
        "dir": "2026-07-27_qwen3tts-1.7B-voicedesign-de_n3",
        "license": "Apache-2.0",
    },
    {
        "slug": "voxcpm2-design-de",
        "title": "VoxCPM2 · Voice-Design deutsch",
        "dir": "2026-07-28_voxcpm2-design-de_n3",
        "license": "Apache-2.0",
    },
    {
        "slug": "chatterbox-de-f1",
        "title": "Chatterbox Multilingual V3 · Referenzstimme de_f1",
        "dir": "2026-07-28_chatterbox-v3-de_f1_n3",
        "license": "MIT (Audio enthält Perth-Wasserzeichen)",
    },
    {
        "slug": "magpie-tn",
        "title": "MagpieTTS 357M + deutsche TN · sofia",
        "dir": "2026-07-27_sofia_v4-tn-n3",
        "license": "NVIDIA Open Model License",
    },
]

DISCLAIMER = (
    "Alle Clips sind KI-generiert (synthetische Sprache). Die Lizenzangaben "
    "sind Hinweise auf die jeweiligen Modell-Lizenzen, keine Rechtsberatung — "
    "verbindlich sind allein die Lizenztexte der Modellanbieter."
)

CSS = """
 body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 70rem;
        padding: 0 1rem; color: #1a1a1a; background: #fff; }
 a { color: #1a6fb5; }
 h2 { border-bottom: 2px solid #ccc; padding-bottom: .3rem; margin-top: 2.5rem; }
 table { border-collapse: collapse; margin: 1rem 0; width: 100%; }
 th, td { border: 1px solid #ccc; padding: .35rem .6rem; text-align: right; }
 th:first-child, td:first-child { text-align: left; }
 td.best { font-weight: 700; background: #eaf6ea; }
 .case { border: 1px solid #ddd; border-radius: 8px; padding: .8rem 1rem; margin: .8rem 0; }
 .case.bad { border-left: 6px solid #c0392b; }
 .case.mid { border-left: 6px solid #e67e22; }
 .case.good { border-left: 6px solid #27ae60; }
 .text { font-weight: 600; }
 .transcript { color: #555; font-style: italic; }
 .meta { color: #777; font-size: .85rem; }
 footer { margin-top: 3rem; color: #777; font-size: .85rem;
          border-top: 1px solid #ccc; padding-top: 1rem; }
 audio { height: 2rem; vertical-align: middle; margin: .2rem .4rem .2rem 0; }
 @media (prefers-color-scheme: dark) {
   body { color: #ddd; background: #16181c; }
   a { color: #6fb3e8; }
   th, td, .case, h2, footer { border-color: #3a3f47; }
   td.best { background: #1f3a24; }
   .transcript { color: #aaa; }
 }
"""


def page(title: str, body: str) -> str:
    return (f'<!doctype html><html lang="de"><head><meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>{html.escape(title)}</title>\n<style>{CSS}</style></head><body>\n"
            f"{body}\n<footer>{html.escape(DISCLAIMER)}</footer></body></html>")


def load_run(run: dict) -> dict:
    res_dir = REPO / "results" / run["dir"]
    summary = json.loads((res_dir / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(l)
            for l in (res_dir / "results_raw.jsonl").read_text(encoding="utf-8").splitlines()]
    return {**run, "res_dir": res_dir, "summary": summary, "rows": rows}


def encode_clips(run: dict) -> int:
    """<id>_r0.wav → docs/audio/<slug>/<id>.mp3 (mono). Liefert Byte-Summe."""
    out_dir = DOCS / "audio" / run["slug"]
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for r in run["rows"]:
        src = run["res_dir"] / "audio" / f"{r['id']}_r0.wav"
        if not src.exists():
            src = run["res_dir"] / "audio" / f"{r['id']}.wav"
        if not src.exists():
            print(f"  WARNUNG: kein Audio für {r['id']}", file=sys.stderr)
            continue
        data, rate = sf.read(src)
        if data.ndim > 1:
            data = data.mean(axis=1)
        dst = out_dir / f"{r['id']}.mp3"
        sf.write(dst, data, rate)
        total += dst.stat().st_size
    return total


def model_page(run: dict) -> None:
    s = run["summary"]
    parts = [f'<p><a href="index.html">← Übersicht</a></p>',
             f"<h1>{html.escape(run['title'])}</h1>",
             f'<p class="meta">Modell: {html.escape(str(s["tts_model"]))} · '
             f'Stimme: {html.escape(str(s["voice"]))} · '
             f'STT-Judge: {html.escape(str(s["stt_model"]))} · '
             f'WER {s["wer_mean"]:.3f} · CER {s["cer_mean"]:.3f} · RTF {s["rtf_mean"]:.2f} · '
             f'Lizenz: {html.escape(run["license"])}</p>',
             '<p class="meta">Je Fall ein Clip (erste von '
             f'{s["n_repeats"]} Wiederholungen); WER ist der Mittelwert über alle '
             'Wiederholungen. Rot ≥ 0.3, Orange ≥ 0.1, Grün &lt; 0.1.</p>']

    by_cat: dict[str, list] = {}
    for r in run["rows"]:
        by_cat.setdefault(r["category"], []).append(r)

    for cat, cases in sorted(by_cat.items()):
        parts.append(f"<h2>{html.escape(cat)}</h2>")
        for r in sorted(cases, key=lambda x: -(x.get("wer") or 0)):
            w = r.get("wer")
            cls = "bad" if (w or 0) >= 0.3 else ("mid" if (w or 0) >= 0.1 else "good")
            tx = (r.get("repeats") or [r])[0].get("transcript", "")
            wer_str = f"{w:.2f}" if w is not None else "FEHLER"
            parts.append(f"""<div class="case {cls}">
 <div class="text">{html.escape(r["text"])}</div>
 <div><audio controls preload="none" src="audio/{run["slug"]}/{r["id"]}.mp3"></audio></div>
 <div class="transcript">→ {html.escape(tx)}</div>
 <div class="meta">{r["id"]} · WER {wer_str}</div>
</div>""")

    (DOCS / f"{run['slug']}.html").write_text(page(run["title"], "\n".join(parts)),
                                              encoding="utf-8")


def index_page(runs: list[dict]) -> None:
    cats = sorted({c for run in runs for c in run["summary"]["wer_by_category"]})
    metrics = ([("WER (Mittel)", lambda s: s["wer_mean"]),
                ("WER (best-of-N)", lambda s: s["wer_best_mean"]),
                ("CER", lambda s: s["cer_mean"]),
                ("Realtime-Faktor", lambda s: s["rtf_mean"])] +
               [(f"WER {c}", lambda s, c=c: s["wer_by_category"].get(c)) for c in cats])

    head = "".join(f'<th><a href="{r["slug"]}.html">{html.escape(r["title"])}</a></th>'
                   for r in runs)
    body_rows = []
    for label, get in metrics:
        vals = [get(r["summary"]) for r in runs]
        best = min(v for v in vals if v is not None)
        tds = "".join(f'<td class="{"best" if v == best else ""}">'
                      f'{f"{v:.3f}" if v is not None else "–"}</td>' for v in vals)
        body_rows.append(f"<tr><td>{html.escape(label)}</td>{tds}</tr>")

    n = runs[0]["summary"]
    lic_items = "".join(f"<li><b>{html.escape(r['title'])}</b>: {html.escape(r['license'])}</li>"
                        for r in runs)
    body = f"""<h1>Deutscher TTS-Vergleich auf dem DGX Spark</h1>
<p>{n["n_total"]} Testfälle (<a href="https://github.com/MvdB/dgx-spark-tts">Testset &amp; Eval-Code</a>),
N={n["n_repeats"]} Wiederholungen, Judge: {html.escape(str(n["stt_model"]))} mit Casing-Prompt.
Niedriger = besser; bester Wert je Zeile hervorgehoben. Die WER enthält auch STT-Fehler
(obere Schranke des TTS-Fehlers) — Kategorien-<i>Deltas</i> sind aussagekräftiger als Absolutwerte.
Spaltentitel führen zur Abhörseite mit allen Clips.</p>
<table><tr><th>Metrik</th>{head}</tr>{"".join(body_rows)}</table>
<h2>Lizenzhinweise</h2>
<ul>{lic_items}</ul>
<p class="meta">Geplant: mistralai/Voxtral-4B-TTS (CC BY-NC 4.0, nicht-kommerziell) via vLLM-Omni.</p>"""
    (DOCS / "index.html").write_text(page("Deutscher TTS-Vergleich (DGX Spark)", body),
                                     encoding="utf-8")


def main() -> int:
    DOCS.mkdir(exist_ok=True)
    runs = [load_run(r) for r in RUNS]
    total = 0
    for run in runs:
        size = encode_clips(run)
        total += size
        model_page(run)
        print(f"{run['slug']}: {len(run['rows'])} Clips, {size / 1e6:.1f} MB")
    index_page(runs)
    print(f"gesamt: {total / 1e6:.1f} MB Audio → {DOCS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
