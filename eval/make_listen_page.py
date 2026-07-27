#!/usr/bin/env python3
"""Erzeugt eine Abhör-Seite (listen.html) für ein Ergebnisverzeichnis.

Zeigt pro Testfall: Eingabetext, Audio-Player (alle Wiederholungen),
Transkript(e) und WER — gruppiert nach Kategorie, Schlechteste zuerst.

Aufruf:  python make_listen_page.py <results-dir> [weitere ...]
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path


def build(res_dir: Path) -> Path:
    rows = [json.loads(l) for l in (res_dir / "results_raw.jsonl").read_text(encoding="utf-8").splitlines()]
    try:
        summary = json.loads((res_dir / "summary.json").read_text(encoding="utf-8"))
    except Exception:
        summary = {}

    def audio_tag(fname: str) -> str:
        return (f'<audio controls preload="none" src="audio/{fname}"></audio>'
                if (res_dir / "audio" / fname).exists() else "")

    parts = [f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<title>TTS-Abhörseite – {html.escape(res_dir.name)}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 70rem; padding: 0 1rem; }}
 h2 {{ border-bottom: 2px solid #ccc; padding-bottom: .3rem; margin-top: 2.5rem; }}
 .case {{ border: 1px solid #ddd; border-radius: 8px; padding: .8rem 1rem; margin: .8rem 0; }}
 .case.bad {{ border-left: 6px solid #c0392b; }}
 .case.mid {{ border-left: 6px solid #e67e22; }}
 .case.good {{ border-left: 6px solid #27ae60; }}
 .text {{ font-weight: 600; }}
 .transcript {{ color: #555; font-style: italic; }}
 .meta {{ color: #888; font-size: .85rem; }}
 audio {{ height: 2rem; vertical-align: middle; margin: .2rem .4rem .2rem 0; }}
</style></head><body>
<h1>TTS-Abhörseite: {html.escape(res_dir.name)}</h1>
<p class="meta">Stimme: {html.escape(str(summary.get("voice", "?")))} ·
 WER gesamt: {summary.get("wer_mean", "?")} ·
 {summary.get("n_repeats", 1)} Wiederholung(en)/Fall ·
 Rot ≥ 0.3, Orange ≥ 0.1, Grün &lt; 0.1</p>"""]

    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    for cat, cases in sorted(by_cat.items()):
        parts.append(f"<h2>{html.escape(cat)}</h2>")
        for r in sorted(cases, key=lambda x: -(x.get("wer") or 0)):
            w = r.get("wer")
            cls = "bad" if (w or 0) >= 0.3 else ("mid" if (w or 0) >= 0.1 else "good")
            reps = r.get("repeats") or [r]
            audios, seen_tx = [], []
            for idx, rep in enumerate(reps):
                suffix = f"_r{idx}" if len(reps) > 1 else ""
                audios.append(audio_tag(f"{r['id']}{suffix}.wav"))
                tx = rep.get("transcript", "")
                if tx and tx not in seen_tx:
                    seen_tx.append(tx)
            tx_html = "<br>".join(f"→ {html.escape(t)}" for t in seen_tx)
            wer_str = f"{w:.2f}" if w is not None else "FEHLER"
            spread = (f" (min {r['wer_min']:.2f} / max {r['wer_max']:.2f})"
                      if r.get("wer_min") is not None and len(reps) > 1 else "")
            parts.append(f"""<div class="case {cls}">
 <div class="text">{html.escape(r["text"])}</div>
 <div>{"".join(audios)}</div>
 <div class="transcript">{tx_html}</div>
 <div class="meta">{r["id"]} · WER {wer_str}{spread}</div>
</div>""")

    parts.append("</body></html>")
    out = res_dir / "listen.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Aufruf: make_listen_page.py <results-dir> [...]")
    for d in sys.argv[1:]:
        print("geschrieben:", build(Path(d)))
