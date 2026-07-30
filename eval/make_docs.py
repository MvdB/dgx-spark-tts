#!/usr/bin/env python3
"""Erzeugt die statischen Vergleichsseiten unter docs/ (GitHub-Pages-tauglich).

results/ wird automatisch gescannt und nach (TTS-Modell, Stimme) gruppiert;
je Kombination wird der NEUESTE vollständige Lauf publiziert. Der Seitenname
ist stabil aus Modell+Stimme abgeleitet — ein neuer Lauf derselben
Kombination überschreibt also die bestehende Seite. Seiten/Audio zu
Kombinationen, die es nicht mehr gibt, werden entfernt.

Es entsteht:
  docs/index.html          – Modellvergleich (Metriken, Kategorien, Links)
  docs/<modell>-<stimme>.html
  docs/audio/<modell>-<stimme>/*.mp3

Je Fall EIN Clip (Wiederholung r0 — repräsentativ, nicht der beste Repeat),
MP3 statt WAV, damit das Repo klein bleibt (~3 KB/s statt ~44 KB/s). Die
WER-Angabe pro Fall ist der Mittelwert über alle Repeats.

Unvollständige Läufe (Smoke, --limit, --category) werden übersprungen
(weniger als MIN_CASES Fälle).

Abhängigkeit: soundfile (pip install soundfile; braucht libsndfile >= 1.2 für MP3).

Aufruf:  python eval/make_docs.py            # aus dem Repo-Root
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
DOCS = REPO / "docs"
MIN_CASES = 40  # Testset hat 43 Fälle; alles darunter ist ein Teil-Lauf

# Lizenz-Kurzhinweis je Modell (Substring-Match auf tts_model) — Hinweise,
# keine Rechtsberatung; verbindlich sind die Lizenztexte der Anbieter.
LICENSES = [
    ("chatterbox", "MIT (Audio enthält Perth-Wasserzeichen)"),
    ("Qwen", "Apache-2.0"),
    ("VoxCPM", "Apache-2.0"),
    ("magpie", "NVIDIA Open Model License"),
    ("Voxtral", "CC BY-NC 4.0 (nicht-kommerziell)"),
]

DISCLAIMER = (
    "Alle Clips sind KI-generiert (synthetische Sprache). Die Lizenzangaben "
    "sind Hinweise auf die jeweiligen Modell-Lizenzen, keine Rechtsberatung — "
    "verbindlich sind allein die Lizenztexte der Modellanbieter."
)

CSS = """
 body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 80rem;
        padding: 0 1rem; color: #1a1a1a; background: #fff; }
 a { color: #1a6fb5; }
 h2 { border-bottom: 2px solid #ccc; padding-bottom: .3rem; margin-top: 2.5rem; }
 .tablewrap { overflow-x: auto; }
 table { border-collapse: collapse; margin: 1rem 0; }
 th, td { border: 1px solid #ccc; padding: .35rem .6rem; text-align: right; }
 th:first-child, td:first-child { text-align: left; }
 td.best { font-weight: 700; background: #eaf6ea; }
 .case { border: 1px solid #ddd; border-radius: 8px; padding: .8rem 1rem; margin: .8rem 0; }
 .case.bad { border-left: 6px solid #c0392b; }
 .case.mid { border-left: 6px solid #e67e22; }
 .case.good { border-left: 6px solid #27ae60; }
 .text { font-weight: 600; }
 .transcript { color: #555; font-style: italic; }
 .prompt { margin: .6rem 0; padding: .6rem .8rem; background: #f4f6f8;
           border-left: 3px solid #1a6fb5; border-radius: 3px; }
 .prompt b { display: block; font-size: .85rem; text-transform: uppercase;
             letter-spacing: .03em; color: #45596b; margin-bottom: .3rem; }
 .prompt code { font-size: .9rem; white-space: pre-wrap; word-break: break-word; }
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
   .prompt { background: #1c2126; border-left-color: #4a9eda; }
   .prompt b { color: #8fa6b8; }
 }
"""


def page(title: str, body: str) -> str:
    return (f'<!doctype html><html lang="de"><head><meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>{html.escape(title)}</title>\n<style>{CSS}</style></head><body>\n"
            f"{body}\n<footer>{html.escape(DISCLAIMER)}</footer></body></html>")


def model_display(tts_model: str) -> str:
    """'/hf_models/Qwen--Qwen3-TTS-…' → 'Qwen3-TTS-…'; Pfad/Vendor-Präfix weg."""
    base = tts_model.rstrip("/").rsplit("/", 1)[-1]
    if "--" in base:
        base = base.split("--", 1)[1]
    return base


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def license_note(tts_model: str) -> str:
    for needle, note in LICENSES:
        if needle.lower() in tts_model.lower():
            return note
    return "unbekannt — Modellkarte prüfen"


def run_date(res_dir: Path) -> tuple[str, float]:
    """Sortierschlüssel 'neuester Lauf': Datums-Präfix, Gleichstand per mtime."""
    m = re.match(r"\d{4}-\d{2}-\d{2}", res_dir.name)
    return (m.group(0) if m else "", res_dir.stat().st_mtime)


def discover_runs() -> list[dict]:
    """Neuester vollständiger Lauf je (tts_model, voice), sortiert nach WER."""
    by_combo: dict[tuple, dict] = {}
    for d in sorted(RESULTS.iterdir()):
        sfile = d / "summary.json"
        if not sfile.exists():
            continue
        s = json.loads(sfile.read_text(encoding="utf-8"))
        if s.get("n_ok", 0) < MIN_CASES:
            print(f"übersprungen (Teil-Lauf, n_ok={s.get('n_ok')}): {d.name}")
            continue
        combo = (s["tts_model"], s["voice"])
        prev = by_combo.get(combo)
        if prev and run_date(prev["res_dir"]) >= run_date(d):
            continue
        disp = model_display(s["tts_model"])
        rescore = None
        if (d / "rescore_judge2.json").exists():
            rescore = json.loads((d / "rescore_judge2.json").read_text(encoding="utf-8"))
        by_combo[combo] = {
            "res_dir": d,
            "summary": s,
            "slug": f"{slugify(disp)}-{slugify(str(s['voice']))}",
            "title": f"{disp} · {s['voice']}",
            "license": license_note(s["tts_model"]),
            "rescore": rescore,
            "rows": [json.loads(l) for l in
                     (d / "results_raw.jsonl").read_text(encoding="utf-8").splitlines()],
        }
    return sorted(by_combo.values(),
                  key=lambda r: r["summary"].get("wer_capped_mean")
                  or r["summary"].get("wer_mean") or 9)


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


def fmt(v, digits: int = 3) -> str:
    return f"{v:.{digits}f}" if isinstance(v, (int, float)) else "–"


def model_page(run: dict) -> None:
    s = run["summary"]
    parts = ['<p><a href="index.html">← Übersicht</a></p>',
             f"<h1>{html.escape(run['title'])}</h1>",
             f'<p class="meta">Modell: {html.escape(str(s["tts_model"]))} · '
             f'Stimme: {html.escape(str(s["voice"]))} · '
             f'STT-Judge: {html.escape(str(s.get("stt_model", "?")))} · '
             f'WER {fmt(s.get("wer_capped_mean", s.get("wer_mean")))} (Cap 1.0) · '
             f'CER {fmt(s.get("cer_capped_mean", s.get("cer_mean")))} · '
             f'RTF {fmt(s.get("rtf_mean"), 2)} · '
             f'Lizenz: {html.escape(run["license"])} · '
             f'Lauf: {html.escape(run["res_dir"].name)}</p>',
             '<p class="meta">Je Fall ein Clip (erste von '
             f'{s.get("n_repeats", 1)} Wiederholung(en)); WER ist der Mittelwert über '
             'alle Wiederholungen. Rot ≥ 0.3, Orange ≥ 0.1, Grün &lt; 0.1.</p>']

    # Bei prompt-gesteuerten Stimmen (Qwen VoiceDesign, VoxCPM2) IST der
    # instruct-Text die Stimme — ohne ihn ist die Seite nicht nachvollziehbar.
    if s.get("voice_instruct"):
        parts.append(
            f'<div class="prompt"><b>Stimm-Prompt (instruct)</b>'
            f'<code>{html.escape(s["voice_instruct"])}</code></div>')
    if s.get("stt_prompt"):
        parts.append(
            f'<div class="prompt"><b>Judge-Prompt</b>'
            f'<code>{html.escape(s["stt_prompt"])}</code></div>')

    rescore_by_id = {}
    if run["rescore"]:
        rescore_by_id = {c["id"]: c for c in run["rescore"]["cases"]}
        parts.append(
            f'<p class="meta">Kreuzvalidiert mit Zweit-Judge '
            f'{html.escape(run["rescore"]["judge2"])} (r0, beste WER über refs+Text): '
            f'WER {run["rescore"]["wer_judge2_mean"]:.3f} '
            f'(Judge 1 im selben Protokoll: {run["rescore"]["wer_judge1_mean"]:.3f}).</p>')

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
            j2 = rescore_by_id.get(r["id"])
            j2_html = (f'<div class="transcript">→ Judge 2: {html.escape(j2["judge2_transcript"])}'
                       f' <span class="meta">(WER {j2["wer_judge2"]:.2f})</span></div>' if j2 else "")
            parts.append(f"""<div class="case {cls}">
 <div class="text">{html.escape(r["text"])}</div>
 <div><audio controls preload="none" src="audio/{run["slug"]}/{r["id"]}.mp3"></audio></div>
 <div class="transcript">→ {html.escape(tx)}</div>
 {j2_html}
 <div class="meta">{r["id"]} · WER {wer_str}</div>
</div>""")

    (DOCS / f"{run['slug']}.html").write_text(page(run["title"], "\n".join(parts)),
                                              encoding="utf-8")


def index_page(runs: list[dict]) -> None:
    cats = sorted({c for run in runs
                   for c in (run["summary"].get("wer_by_category") or {})})
    # Leitmetrik ist die je Wiederholung bei 1.0 gekappte WER — die rohe WER
    # ist nach oben offen, ein einziger ASR-Runaway (WER 36 bei 3.7 s Audio)
    # wuerde sonst den Mittelwert dominieren. Fallback auf wer_mean fuer
    # Laeufe vor Einfuehrung des Caps.
    metrics = ([("WER (Mittel, Cap 1.0)",
                 lambda s: s.get("wer_capped_mean", s.get("wer_mean"))),
                ("WER (Mittel, ungekappt)", lambda s: s.get("wer_mean")),
                ("WER (best-of-N)", lambda s: s.get("wer_best_mean")),
                ("CER (Cap 1.0)",
                 lambda s: s.get("cer_capped_mean", s.get("cer_mean"))),
                ("Realtime-Faktor", lambda s: s.get("rtf_mean"))] +
               [(f"WER {c}",
                 lambda s, c=c: (s.get("wer_capped_by_category")
                                 or s.get("wer_by_category") or {}).get(c))
                for c in cats])

    head = "".join(f'<th><a href="{r["slug"]}.html">{html.escape(r["title"])}</a></th>'
                   for r in runs)
    body_rows = [("<tr><td>N (Wiederholungen)</td>" +
                  "".join(f'<td>{r["summary"].get("n_repeats", 1)}</td>' for r in runs) +
                  "</tr>")]
    for label, get in metrics:
        vals = [get(r["summary"]) for r in runs]
        present = [v for v in vals if v is not None]
        best = min(present) if present else None
        tds = "".join(f'<td class="{"best" if v is not None and v == best else ""}">'
                      f"{fmt(v)}</td>" for v in vals)
        body_rows.append(f"<tr><td>{html.escape(label)}</td>{tds}</tr>")

    # Zweit-Judge-Zeilen (Kreuzvalidierung; Protokoll: r0, beste WER über
    # refs + Originaltext — siehe Methodik-Absatz)
    if any(r["rescore"] for r in runs):
        judge2 = next(r["rescore"]["judge2"] for r in runs if r["rescore"])
        for label, key in [(f"WER {runs[0]['summary'].get('stt_model', 'Judge 1')} (r0, refs+Text)", "wer_judge1_mean"),
                           (f"WER {judge2} (r0, refs+Text)", "wer_judge2_mean")]:
            vals = [r["rescore"].get(key) if r["rescore"] else None for r in runs]
            present = [v for v in vals if v is not None]
            best = min(present) if present else None
            tds = "".join(f'<td class="{"best" if v is not None and v == best else ""}">'
                          f"{fmt(v)}</td>" for v in vals)
            body_rows.append(f"<tr><td>{html.escape(label)}</td>{tds}</tr>")

    n = runs[0]["summary"]
    lic_items = "".join(f"<li><b>{html.escape(r['title'])}</b>: {html.escape(r['license'])}</li>"
                        for r in runs)
    judge_note = ""
    if any(r["rescore"] for r in runs):
        judge2 = next(r["rescore"]["judge2"] for r in runs if r["rescore"])
        judge_note = (
            f'<p><b>Judge-Wahl und Kreuzvalidierung:</b> Haupt-Judge ist '
            f'<b>Whisper large-v3</b>. Er hat granite-speech-4.1-2b abgelöst, nachdem '
            f'eine Kalibrierung mit <i>bekanntem</i> Audioinhalt granite überführt hat: '
            f'Ein TTS sprach die bereits ausgeschriebenen Referenztexte, sodass feststand, '
            f'was im Audio zu hören ist. granite verlor dort systematisch Zahlen — aus '
            f'„siebzehn Uhr fünfundvierzig" wurde „Der Zug fährt um uhr", aus „eine Million '
            f'zweihundertfünfzigtausend Euro" ein abgebrochenes „beläuft sich auf eine". '
            f'Whisper transkribiert diese Fälle korrekt (WER 0.137 gegen 0.147, Wortverlust '
            f'0.126 gegen 0.143). Sein Schwachpunkt ist umgekehrt die Schreibweise: Ohne '
            f'Gegenmaßnahme notiert er Zahlen als Ziffern („17.45 Uhr") und macht damit '
            f'unmessbar, was der Testsatz prüft. Ein Initial-Prompt drängt ihn zu '
            f'ausgeschriebenen Zahlwörtern und senkt die Ziffernquote von 83&nbsp;% auf '
            f'33&nbsp;%; seine Beispiele stammen bewusst nicht aus dem Testsatz, sonst '
            f'souffliert man dem Judge die erwarteten Antworten.</p>'
            f'<p>Die unteren beiden Zeilen zeigen beide Judges im identischen Protokoll '
            f'(nur Wiederholung r0, beste WER über refs plus normalisierten Originaltext). '
            f'{html.escape(judge2)} dient als Gegenprobe, ist aber kein neutraler Maßstab: '
            f'Er rück-normalisiert aggressiv zu Ziffern und wertet damit eine falsche '
            f'Verbalisierung als Treffer. Die Spanne zwischen beiden Zeilen ist als '
            f'Unsicherheitsband zu lesen, nicht als zwei gleichwertige Messungen. '
            f'Beide Modelle können in Decoder-Endlosschleifen laufen — das ist kein '
            f'Einzelfehler von granite, sondern generelles Verhalten autoregressiver '
            f'ASR-Modelle. Solche Fälle werden erkannt, einmal mit leichtem Sampling '
            f'wiederholt und ansonsten bei WER&nbsp;1.0 gekappt.</p>')
    body = f"""<h1>Deutscher TTS-Vergleich auf dem DGX Spark</h1>
<p>{n["n_total"]} Testfälle (<a href="https://github.com/MvdB/dgx-spark-tts">Testset &amp; Eval-Code</a>),
Judge: {html.escape(str(n.get("stt_model", "?")))} mit Casing-Prompt. Je Modell/Stimme wird der
jeweils neueste vollständige Lauf gezeigt (Spalten nach WER sortiert, bester Wert je Zeile
hervorgehoben; niedriger = besser). Die WER enthält auch STT-Fehler (obere Schranke des
TTS-Fehlers) — Kategorien-<i>Deltas</i> sind aussagekräftiger als Absolutwerte.
Leitmetrik ist die je Wiederholung bei 1.0 gekappte WER (Totalersetzung); die Differenz
zur ungekappten Zeile zeigt, wo einzelne Wiederholungen entgleist sind (ASR-Decoder-Schleifen
oder TTS-Loop-Babble erzeugen sonst WER&nbsp;≫&nbsp;1 und dominieren den Mittelwert).
Spaltentitel führen zur Abhörseite mit allen Clips.</p>
{judge_note}
<div class="tablewrap"><table><tr><th>Metrik</th>{head}</tr>{"".join(body_rows)}</table></div>
<h2>Lizenzhinweise</h2>
<ul>{lic_items}</ul>"""
    (DOCS / "index.html").write_text(page("Deutscher TTS-Vergleich (DGX Spark)", body),
                                     encoding="utf-8")


def prune(runs: list[dict]) -> None:
    """Seiten/Audio zu nicht mehr vorhandenen Kombinationen entfernen."""
    keep = {r["slug"] for r in runs}
    for p in DOCS.glob("*.html"):
        if p.stem != "index" and p.stem not in keep:
            print(f"entfernt (veraltet): {p.relative_to(REPO)}")
            p.unlink()
    for d in (DOCS / "audio").glob("*/"):
        if d.name not in keep:
            print(f"entfernt (veraltet): {d.relative_to(REPO)}/")
            for f in d.iterdir():
                f.unlink()
            d.rmdir()


def main() -> int:
    DOCS.mkdir(exist_ok=True)
    runs = discover_runs()
    if not runs:
        print("keine vollständigen Läufe in results/ gefunden", file=sys.stderr)
        return 1
    total = 0
    for run in runs:
        size = encode_clips(run)
        total += size
        model_page(run)
        print(f"{run['slug']}: {len(run['rows'])} Clips, {size / 1e6:.1f} MB "
              f"(aus {run['res_dir'].name})")
    index_page(runs)
    prune(runs)
    print(f"gesamt: {total / 1e6:.1f} MB Audio → {DOCS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
