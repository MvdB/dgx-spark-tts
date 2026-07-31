# Offene Punkte für die nächste Session

## ZUERST: fehlende Gegenprobe für de_female_coach

Die beiden Schulungsstimmen sind **fertig evaluiert und veröffentlicht**
(Commits 9b0b06b und e0c0a34):

| Stimme | WER | CER |
|---|---|---|
| de_female_coach | 0.164 | 0.124 |
| de_male_coach | 0.171 | 0.145 |

Beide liegen im vorderen Feld — die angenehmere, zügigere Sprechweise kostet
keine Verständlichkeit gegenüber den Nachrichtenstimmen.

**Einziger offener Punkt:** Für `de_female_coach` fehlt die Kreuzvalidierung
mit dem Zweit-Judge (der Witness-Container war nicht rechtzeitig oben, dann lief
die Zeit ab). `de_male_coach` hat seine `rescore_judge2.json`. Nachzuziehen mit:

```bash
docker start stt-witness        # warten bis /v1/models antwortet
python eval/rescore_with_judge.py --stt2 http://127.0.0.1:8006 \
  results/2026-07-30_suite_qwen-vd-de-female-coach
python eval/make_docs.py        # danach docs/ committen
```

Das ist eine Gegenprobe, kein Messwert — die publizierten Zahlen stehen.

**Michael hat die Hörproben nicht mehr bewerten können.** Je drei männliche und
weibliche Varianten wurden erzeugt; gewählt habe ich unter Zeitdruck B (männlich)
und F (weiblich) — die zügigsten mit passendem Charakter. Falls ihm eine andere
besser gefällt: Beschreibung in `VOICE_DESIGN_PRESETS`
(`serving/server_qwen3tts.py`) tauschen, Image neu bauen, Eval nachziehen. Die
Hörproben lagen im Scratchpad und sind weg — mit den Beschreibungen unten neu
erzeugbar.

Sprechtempo der Kandidaten (s/Zeichen; die bisherigen Presets liegen bei 0.099):

| Variante | Charakter | Tempo |
|---|---|---|
| A | männlich, wacher Erklärton | 0.0596 |
| **B** | **männlich, warm-motivierend (gewählt)** | **0.0581** |
| C | männlich, souverän-zugewandt | 0.0667 |
| D | weiblich, wacher Erklärton | 0.0649 |
| E | weiblich, warm-motivierend | 0.0735 |
| **F** | **weiblich, souverän-zugewandt (gewählt)** | **0.0615** |

Beschreibungstexte aller sechs Varianten:

```json
{
  "A_trainer_wach": "Sympathische deutsche Männerstimme, Anfang dreißig, freundlich und wach. Muttersprachliches Hochdeutsch, zugewandter Erklärton wie in einer guten Online-Schulung, zügiges Sprechtempo, lebendige aber ruhige Betonung, Satzenden fallen ab.",
  "B_warm_motivierend": "Angenehme deutsche Männerstimme, mittlere Tonlage, warm und motivierend. Muttersprachliches Hochdeutsch, wacher Trainerton, zügiges Sprechtempo, deutliche Artikulation, freundliche Energie ohne Werbestimme, kein Pathos.",
  "C_souveraen_zugewandt": "Freundliche deutsche Männerstimme eines erfahrenen Trainers, Mitte dreißig. Muttersprachliches Hochdeutsch, ruhig-souverän und zugewandt, zügig ohne Hast, natürliche Betonungswechsel, kein Vorlese-Singsang, keine gedehnten Vokale.",
  "D_trainerin_wach": "Sympathische deutsche Frauenstimme, Anfang dreißig, freundlich und wach. Muttersprachliches Hochdeutsch, zugewandter Erklärton wie in einer guten Online-Schulung, zügiges Sprechtempo, lebendige aber ruhige Betonung, Satzenden fallen ab.",
  "E_warm_motivierend": "Angenehme deutsche Frauenstimme, mittlere Tonlage, warm und motivierend. Muttersprachliches Hochdeutsch, wacher Trainerton, zügiges Sprechtempo, deutliche Artikulation, freundliche Energie ohne Werbestimme, kein Pathos.",
  "F_souveraen_zugewandt": "Freundliche deutsche Frauenstimme einer erfahrenen Trainerin, Mitte dreißig. Muttersprachliches Hochdeutsch, ruhig-souverän und zugewandt, zügig ohne Hast, natürliche Betonungswechsel, kein Vorlese-Singsang, keine gedehnten Vokale."
}
```

Vorlesetext der Hörproben: „Bevor wir zum nächsten Modul kommen, halten wir kurz
fest, was Sie bereits geschafft haben: Sie kennen jetzt die drei Grundprinzipien
und können sie an einem eigenen Beispiel anwenden. Das ist genau der Punkt, an
dem es erfahrungsgemäß Spaß zu machen beginnt."

Stand: 2026-07-30, Ende der Session. Der komplette Suite-Neulauf mit
whisper-large-v3 als Haupt-Judge ist durch und veröffentlicht (14
Konfigurationen, `docs/` auf GitHub Pages).

## Direkt anschlussfähig

1. **Stimmbeschreibungen der übrigen drei VoiceDesign-Presets überarbeiten.**
   `de_male_news` wurde nach Hörvergleich auf die Bruststimmen-Variante
   umgestellt, weil die Vorgängerfassung „jammerig" klang (gedehnt: 13.4 s
   statt 8.6 s für denselben Satz). Ursache war doppelt: ASCII-Umschreibungen
   im Prompt („Maennerstimme") plus die Kombination „ruhig + sonor + mäßiges
   Sprechtempo". Die anderen drei Presets sind auf echte Umlaute korrigiert,
   aber inhaltlich noch nie abgehört worden — insbesondere `de_female_calm`
   enthält mit „langsames bis mittleres Sprechtempo" dasselbe Risiko.
   Vorgehen wie gehabt: Hörproben erzeugen, auswählen, dann Eval nachziehen.

2. **Prosodie ist blind gemessen.** Der Testsatz misst Verständlichkeit, nicht
   Natürlichkeit — genau deshalb fiel das jammerige Timbre erst beim Anhören
   auf, obwohl die Konfiguration mit WER 0.156 in der Spitzengruppe lag. Eine
   billige Ergänzung wäre eine Kennzahl „Sekunden Audio pro Zeichen Text" je
   Konfiguration: Sie hätte den Ausreißer sofort sichtbar gemacht und kostet
   nichts, weil `audio_duration` bereits in den Rohdaten steht.

3. **Dritter Judge fehlt weiterhin.** Mit zwei Judges lässt sich bei
   Uneinigkeit nur eine Spanne angeben, nicht entscheiden, wer recht hat.
   whisper-large-v3 (Haupt) und Voxtral-Mini-3B (Zweit) sind gesetzt;
   granite-speech ist wegen Zahlwort-Verlusten ausgeschieden,
   granite-speech-plus ist in vLLM 0.25.1 defekt
   (`Failed to apply prompt replacement for mm_items['audio'][0]`).
   Kandidaten: `Voxtral-Mini-4B-Realtime` (lokal vorhanden, ungetestet) oder
   ein Whisper-Derivat anderer Herkunft. Prüfen mit `eval/judge_bench.py`
   gegen `testset/judge_calib_v1.jsonl` — das Verfahren steht.

## Beobachtete Instabilitäten (nicht behoben, nur eingefangen)

4. **Qwen VoiceDesign entgleist bei „4.500 U/min"** (`norm-012`): erzeugte
   einmal 56 s Audio statt 8 s und driftete in Kauderwelsch ab. Reproduzierbar?
   Bisher einmalig gesehen. Die Kappung bei WER 1.0 fängt den Effekt auf, die
   Lücke zwischen gekappter und roher WER macht ihn sichtbar.

5. **VoxCPM2 hat seltenen Loop-Babble** (~1 Fall je Lauf). Nach Einführung des
   Runaway-Schutzes fiel es von WER 0.534 auf 0.185–0.193 — der Zusammenbruch
   war weit überwiegend Judge-Halluzination, ein echter Rest bleibt.

6. **Auch Whisper läuft in Decoder-Schleifen** (1187 Zeichen „Eins, zwei, drei,
   zwei, zwei…" für 2.9 s Audio). Das ist generelles Verhalten autoregressiver
   ASR-Modelle, kein granite-Spezifikum. Der Retry mit `temperature 0.3` bricht
   die meisten, nicht alle.

## Betriebszustand beim Sessionende

- Container gestoppt lassen oder starten: `./serving/run_whisper_judge.sh`
  (Haupt-Judge, 8007), `docker start stt-witness` (Zweit-Judge, 8006),
  `docker start vllm-server` (granite — nur falls ein anderes Projekt es
  braucht, für die Eval nicht mehr in Gebrauch).
- `results/judge_bench_calib/` ist bewusst aufgehoben: die Messgrundlage für
  den Judge-Wechsel. Nicht mit den Lauf-Ergebnissen zusammen wegwerfen.
- Der Zweit-Judge war beim letzten Lauf zeitweise nicht erreichbar; falls
  `rescore_judge2.json` für `qwen-vd-de-male-news` fehlt, mit
  `python eval/rescore_with_judge.py --stt2 http://127.0.0.1:8006 <run-dir>`
  nachziehen und `eval/make_docs.py` erneut laufen lassen.

## Methodische Regeln, die diese Session gekostet hat

- Judges nie gegeneinander bewerten, sondern gegen Audio mit **bekanntem**
  Inhalt (`testset/judge_calib_v1.jsonl`).
- Beispiele in Judge-Prompts nie aus dem Testsatz nehmen.
- Judges immer über `/v1/audio/transcriptions` ansprechen: Voxtral-Mini
  beantwortet `chat/completions` bereitwillig — als englische Übersetzung.
- Unterschiede unter ~0.02 WER sind bei n=3 Rauschen und keine Rangfolge.
