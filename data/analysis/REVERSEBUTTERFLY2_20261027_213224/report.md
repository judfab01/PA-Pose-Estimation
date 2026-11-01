# Vergleich MediaPipe und InstantHMR: Reverse Butterfly mit Resistenzband

Video: REVERSEBUTTERFLY2 | Übung: reverse_butterfly | Winkelquelle: tiefenkorrigierte Winkel | gemeinsame Frames: 900 | erstellt: 2026-10-27T21:32:31

## Überblick
|  | MediaPipe | InstantHMR |
| --- | --- | --- |
| Frames (gesamt) | 900 | 900 |
| Dauer [s] | 29.9 | 29.9 |
| Pose erkannt [%] | 100.0 | 100.0 |
| Detektor-Stride (Meta) | – | 1 |
| Frisch ausgewertete Frames `fresh` [%] | 100.0 | 100.0 |
| Latenz Mittel / 95. Perz. [ms] | 53 / 53 | 108 / 123 |
| Gelenke unter Sichtbarkeit 0.5 [%] | 0.9 | – |
| Jitter Ø (RMS 2. Diff., relevante Winkel, gewählte Quelle) [°/Frame²] | 1.0 | 6.7 |
| Jitter Ø roh / korrigiert [°/Frame²] | 2.1 / 1.0 | 2.7 / 6.7 |
| Winkelsprünge > 20°/Frame (relevante Winkel) | 0 | 45 |
| Ø Betrag Segmentabweichung roh zu gemessen [%] | 23.7 | 13.4 |
| Wiederholungen erkannt | 10 | 10 |

## Übereinstimmung (InstantHMR minus MediaPipe, tiefenkorrigierte Winkel)
| Winkel | n | Bias [°] | MAE [°] | RMSE [°] | Korrelation |
| --- | --- | --- | --- | --- | --- |
| elbow_l | 900 | -18.7 | 18.7 | 19.6 | 0.90 |
| elbow_r | 900 | -19.2 | 26.7 | 32.7 | 0.51 |
| shoulder_l | 900 | -8.8 | 9.1 | 10.4 | 0.60 |
| shoulder_r | 900 | +21.3 | 21.3 | 22.4 | 0.69 |
| hip_l | 843 | +6.4 | 6.4 | 7.1 | -0.05 |
| hip_r | 838 | +14.5 | 14.5 | 14.5 | 0.04 |
| knee_l | 843 | +29.4 | 29.4 | 32.9 | 0.06 |
| knee_r | 838 | +15.2 | 21.1 | 21.9 | -0.26 |
| ankle_l | 843 | +19.7 | 19.7 | 20.8 | -0.79 |
| ankle_r | 838 | +16.1 | 16.1 | 17.8 | -0.51 |

Modell-zu-Modell-MAE über die relevanten Winkel: roh 19.65°, nach Tiefenkorrektur 18.96° (näher beieinander). Das ist ein Mass der Übereinstimmung, nicht der Genauigkeit.

## Wiederholungen
| Modell | Nr. | Frame | Zeit [s] | Spitzenwert (Handabstand / Schulterbreite) | bis nächste [s] | Dehnung an Spitze | Ellbogen Mittel an Spitze |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MediaPipe | 1 | 85 | 2.83 | 3.84 | 3.56 | 94 % | 157° |
| MediaPipe | 2 | 192 | 6.39 | 3.87 | 3.13 | 95 % | 156° |
| MediaPipe | 3 | 286 | 9.52 | 3.82 | 3.30 | 95 % | 158° |
| MediaPipe | 4 | 385 | 12.82 | 3.87 | 2.93 | 98 % | 160° |
| MediaPipe | 5 | 473 | 15.75 | 3.71 | 2.46 | 90 % | 158° |
| MediaPipe | 6 | 547 | 18.21 | 3.80 | 2.93 | 93 % | 159° |
| MediaPipe | 7 | 635 | 21.14 | 3.80 | 2.73 | 95 % | 160° |
| MediaPipe | 8 | 717 | 23.88 | 3.83 | 2.46 | 95 % | 159° |
| MediaPipe | 9 | 791 | 26.34 | 3.86 | 2.33 | 97 % | 160° |
| MediaPipe | 10 | 861 | 28.67 | 3.79 | – | 92 % | 157° |
| InstantHMR | 1 | 104 | 3.46 | 4.38 | 2.96 | 115 % | 137° |
| InstantHMR | 2 | 193 | 6.43 | 4.40 | 3.06 | 114 % | 136° |
| InstantHMR | 3 | 285 | 9.49 | 4.36 | 3.30 | 112 % | 161° |
| InstantHMR | 4 | 384 | 12.79 | 4.43 | 2.93 | 118 % | 140° |
| InstantHMR | 5 | 472 | 15.72 | 4.21 | 2.60 | 106 % | 156° |
| InstantHMR | 6 | 550 | 18.31 | 4.32 | 2.76 | 113 % | 159° |
| InstantHMR | 7 | 633 | 21.08 | 4.35 | 2.80 | 112 % | 136° |
| InstantHMR | 8 | 717 | 23.88 | 4.35 | 2.43 | 112 % | 136° |
| InstantHMR | 9 | 790 | 26.31 | 4.42 | 2.36 | 116 % | 158° |
| InstantHMR | 10 | 861 | 28.67 | 4.35 | – | 111 % | 157° |

- Wiederholungen erkannt: MediaPipe 10, InstantHMR 10; davon zeitlich zuordenbar (±20 Frames): 10.
- Mittlerer Zeitversatz der Wiederholungsspitzen zwischen den Modellen: 97 ms.

## Band (Länge = Griffpunktabstand, ab v6 modellabhängig)
| Modell | present [%] | Ø band_red_frac | r Handgelenkabstand | r PCA-Referenz | Versatz Griffpunkt minus PCA Ø ± SD [px] | Versatz Ø [% der Länge] | Versatz SD [% der Länge] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MediaPipe | 100.0 | 0.98 | 1.00 | 0.97 | 1.4 ± 38.4 | 0.2 | 6.3 |
| InstantHMR | 100.0 | 0.97 | 1.00 | 0.97 | 23.7 ± 52.7 | 3.7 | 8.3 |
Der Versatz Griffpunkt minus PCA-Referenz zeigt, ob die Fingerspitzen-Griffpunkte konstant gegenüber dem Band verschoben sind (kleine SD: Kalibrierung des Versatzes möglich) oder haltungsabhängig (grosse SD: Bandenden aus den Rotpixeln bestimmen). Die PCA-Referenz nimmt alle roten Pixel im Bild und ist bei rotem Hintergrund oder roter Kleidung verfälscht.

## Segmentkorrektur und Wirkung auf die Winkel

Δ Winkel = `angle_*_cor_raw` minus `angle_*_raw` (beide ungeglättet). Ein Segment gilt als korrigiert, wenn die Tiefenkorrektur in diesem Frame aktiv war (Abweichung zur Solllänge ab 5 %, Rückfall unter 3 %). Sind beide an einem Winkel beteiligten Segmente nicht korrigiert, bleibt die Richtung erhalten und Δ ist 0.

### MediaPipe
**Wann welche Segmente korrigiert wurden**
| Segment | korrigiert [% Frames] | Episoden | Ø Dauer [s] | Ø Betrag Abw. bei Korrektur [%] | Ø Betrag Abw. ohne Korrektur [%] | Fallback [% Frames] | gesperrt vis [% Frames] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Rumpf l | 6.1 | 1 | 1.82 | 3.9 | 2.4 | 6.1 | 0.0 |
| Rumpf r | 100.0 | 1 | 29.70 | 15.3 | – | 100.0 | 0.0 |
| Oberarm l | 100.0 | 1 | 29.70 | 37.7 | – | 0.0 | 0.0 |
| Oberarm r | 100.0 | 1 | 29.70 | 19.8 | – | 0.0 | 0.0 |
| Unterarm l | 100.0 | 1 | 29.70 | 30.0 | – | 0.0 | 0.0 |
| Unterarm r | 94.3 | 14 | 2.00 | 13.9 | 2.6 | 0.0 | 0.0 |
| Oberschenkel l | 93.7 | 3 | 9.27 | 12.2 | – | 0.0 | 6.3 |
| Oberschenkel r | 93.1 | 3 | 9.22 | 28.1 | – | 0.0 | 6.9 |
| Unterschenkel l | 43.3 | 13 | 0.99 | 6.2 | 2.8 | 0.0 | 6.3 |
| Unterschenkel r | 50.6 | 12 | 1.25 | 4.7 | 3.0 | 0.0 | 6.9 |
| Fuss l | 100.0 | 1 | 29.70 | 69.3 | – | 0.0 | 0.0 |
| Fuss r | 100.0 | 1 | 29.70 | 39.9 | – | 0.0 | 0.0 |

**Wirkung auf die Winkel (Δ = korrigiert minus roh)**
| Winkel | angrenzende Segmente | Ø Betrag Δ [°] | Median Betrag Δ [°] | P95 Betrag Δ [°] | Max Betrag Δ [°] | Ø Δ [°] | Betrag Δ > 1° [% Frames] | Betrag Δ > 5° [% Frames] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| elbow_l | Oberarm l + Unterarm l | 9.77 | 9.29 | 19.1 | 30.1 | 9.77 | 99 | 71 |
| elbow_r | Oberarm r + Unterarm r | 11.10 | 10.76 | 17.6 | 26.3 | 11.10 | 100 | 93 |
| shoulder_l | Rumpf l + Oberarm l | 2.61 | 2.06 | 5.8 | 24.4 | 2.06 | 83 | 11 |
| shoulder_r | Rumpf r + Oberarm r | 9.16 | 9.60 | 13.3 | 15.1 | 9.03 | 98 | 88 |
| hip_l | Rumpf l + Oberschenkel l | 18.41 | 18.29 | 22.4 | 27.5 | -18.41 | 100 | 100 |
| hip_r | Rumpf r + Oberschenkel r | 37.23 | 37.20 | 42.1 | 44.5 | -37.23 | 100 | 100 |
| knee_l | Oberschenkel l + Unterschenkel l | 25.29 | 23.26 | 32.8 | 35.0 | -25.29 | 100 | 100 |
| knee_r | Oberschenkel r + Unterschenkel r | 29.49 | 32.71 | 44.2 | 46.7 | -29.24 | 93 | 82 |
| ankle_l | Unterschenkel l + Fuss l | 35.89 | 34.27 | 49.8 | 54.3 | -35.89 | 100 | 100 |
| ankle_r | Unterschenkel r + Fuss r | 4.70 | 4.14 | 9.2 | 10.7 | -3.15 | 85 | 46 |

**Zuordnung: welche Segmentkorrektur erzeugt welche Winkeländerung**
| Winkel | kein Segment | nur Segment 1 | nur Segment 2 | beide |
| --- | --- | --- | --- | --- |
| elbow_l | keins: n=0, Ø Betrag Δ –° | nur Oberarm l: n=0, Ø Betrag Δ –° | nur Unterarm l: n=0, Ø Betrag Δ –° | beide: n=900, Ø Betrag Δ 9.8° |
| elbow_r | keins: n=0, Ø Betrag Δ –° | nur Oberarm r: n=51, Ø Betrag Δ 3.8° | nur Unterarm r: n=0, Ø Betrag Δ –° | beide: n=849, Ø Betrag Δ 11.5° |
| shoulder_l | keins: n=0, Ø Betrag Δ –° | nur Rumpf l: n=0, Ø Betrag Δ –° | nur Oberarm l: n=845, Ø Betrag Δ 2.7° | beide: n=55, Ø Betrag Δ 1.4° |
| shoulder_r | keins: n=0, Ø Betrag Δ –° | nur Rumpf r: n=0, Ø Betrag Δ –° | nur Oberarm r: n=0, Ø Betrag Δ –° | beide: n=900, Ø Betrag Δ 9.2° |
| hip_l | keins: n=0, Ø Betrag Δ –° | nur Rumpf l: n=0, Ø Betrag Δ –° | nur Oberschenkel l: n=788, Ø Betrag Δ 18.6° | beide: n=55, Ø Betrag Δ 15.6° |
| hip_r | keins: n=0, Ø Betrag Δ –° | nur Rumpf r: n=0, Ø Betrag Δ –° | nur Oberschenkel r: n=0, Ø Betrag Δ –° | beide: n=838, Ø Betrag Δ 37.2° |
| knee_l | keins: n=0, Ø Betrag Δ –° | nur Oberschenkel l: n=453, Ø Betrag Δ 22.2° | nur Unterschenkel l: n=0, Ø Betrag Δ –° | beide: n=390, Ø Betrag Δ 28.9° |
| knee_r | keins: n=0, Ø Betrag Δ –° | nur Oberschenkel r: n=383, Ø Betrag Δ 27.6° | nur Unterschenkel r: n=0, Ø Betrag Δ –° | beide: n=455, Ø Betrag Δ 31.0° |
| ankle_l | keins: n=0, Ø Betrag Δ –° | nur Unterschenkel l: n=0, Ø Betrag Δ –° | nur Fuss l: n=453, Ø Betrag Δ 30.7° | beide: n=390, Ø Betrag Δ 41.9° |
| ankle_r | keins: n=0, Ø Betrag Δ –° | nur Unterschenkel r: n=0, Ø Betrag Δ –° | nur Fuss r: n=383, Ø Betrag Δ 7.9° | beide: n=455, Ø Betrag Δ 2.0° |

Konsistenzprüfung: Frames mit beiden Segmenten unkorrigiert und Betrag Δ > 0.5°: 0 (Soll 0).

**Die 8 Korrektur-Episoden mit der grössten Winkelwirkung** (alle Episoden: `korrekturen.csv`)
| Segment | Frames | Start [s] | Dauer [s] | Ø Abw. roh [%] | max Betrag Δ [°] | Winkel |
| --- | --- | --- | --- | --- | --- | --- |
| Unterschenkel l | 5–11 | 0.17 | 0.23 | -5.6 | 54.3 | ankle_l |
| Fuss l | 0–899 | 0.00 | 29.97 | -69.3 | 54.3 | ankle_l |
| Unterschenkel l | 582–624 | 19.38 | 1.43 | -6.1 | 53.9 | ankle_l |
| Unterschenkel l | 336–373 | 11.19 | 1.27 | -7.0 | 53.4 | ankle_l |
| Unterschenkel l | 883–899 | 29.40 | 0.57 | -7.2 | 53.3 | ankle_l |
| Unterschenkel l | 421–458 | 14.02 | 1.27 | -6.7 | 52.7 | ankle_l |
| Unterschenkel l | 241–278 | 8.03 | 1.27 | -6.7 | 52.6 | ankle_l |
| Unterschenkel l | 819–853 | 27.27 | 1.16 | -6.1 | 50.7 | ankle_l |

### InstantHMR
**Wann welche Segmente korrigiert wurden**
| Segment | korrigiert [% Frames] | Episoden | Ø Dauer [s] | Ø Betrag Abw. bei Korrektur [%] | Ø Betrag Abw. ohne Korrektur [%] | Fallback [% Frames] | gesperrt vis [% Frames] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Rumpf l | 100.0 | 1 | 29.70 | 16.6 | – | 100.0 | 0.0 |
| Rumpf r | 100.0 | 1 | 29.70 | 16.8 | – | 100.0 | 0.0 |
| Oberarm l | 100.0 | 1 | 29.70 | 14.6 | – | 0.0 | 0.0 |
| Oberarm r | 100.0 | 1 | 29.70 | 19.4 | – | 0.0 | 0.0 |
| Unterarm l | 85.3 | 17 | 1.49 | 7.6 | 3.5 | 0.0 | 0.0 |
| Unterarm r | 93.2 | 12 | 2.31 | 8.8 | 1.7 | 0.0 | 0.0 |
| Oberschenkel l | 100.0 | 1 | 29.70 | 9.3 | – | 0.0 | 0.0 |
| Oberschenkel r | 100.0 | 1 | 29.70 | 9.7 | – | 0.0 | 0.0 |
| Unterschenkel l | 0.0 | 0 | – | – | 1.1 | 0.0 | 0.0 |
| Unterschenkel r | 4.4 | 1 | 1.32 | 4.4 | 3.1 | 0.0 | 0.0 |
| Fuss l | 100.0 | 1 | 29.70 | 6.1 | – | 0.0 | 0.0 |
| Fuss r | 100.0 | 1 | 29.70 | 7.0 | – | 0.0 | 0.0 |

**Wirkung auf die Winkel (Δ = korrigiert minus roh)**
| Winkel | angrenzende Segmente | Ø Betrag Δ [°] | Median Betrag Δ [°] | P95 Betrag Δ [°] | Max Betrag Δ [°] | Ø Δ [°] | Betrag Δ > 1° [% Frames] | Betrag Δ > 5° [% Frames] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| elbow_l | Oberarm l + Unterarm l | 11.29 | 11.61 | 17.1 | 33.2 | 11.25 | 100 | 96 |
| elbow_r | Oberarm r + Unterarm r | 26.31 | 27.74 | 39.9 | 47.2 | -0.81 | 100 | 98 |
| shoulder_l | Rumpf l + Oberarm l | 1.61 | 1.65 | 2.7 | 8.4 | 0.52 | 76 | 1 |
| shoulder_r | Rumpf r + Oberarm r | 1.73 | 1.87 | 3.0 | 9.2 | 0.60 | 79 | 0 |
| hip_l | Rumpf l + Oberschenkel l | 14.32 | 14.26 | 15.2 | 16.0 | -14.32 | 100 | 100 |
| hip_r | Rumpf r + Oberschenkel r | 14.44 | 14.06 | 16.5 | 17.7 | -14.44 | 100 | 100 |
| knee_l | Oberschenkel l + Unterschenkel l | 17.37 | 23.46 | 26.2 | 28.2 | -13.52 | 99 | 85 |
| knee_r | Oberschenkel r + Unterschenkel r | 20.27 | 20.05 | 24.3 | 33.4 | -20.27 | 100 | 100 |
| ankle_l | Unterschenkel l + Fuss l | 1.29 | 1.26 | 1.8 | 2.5 | -1.29 | 89 | 0 |
| ankle_r | Unterschenkel r + Fuss r | 2.26 | 1.84 | 3.1 | 13.4 | -2.26 | 99 | 4 |

**Zuordnung: welche Segmentkorrektur erzeugt welche Winkeländerung**
| Winkel | kein Segment | nur Segment 1 | nur Segment 2 | beide |
| --- | --- | --- | --- | --- |
| elbow_l | keins: n=0, Ø Betrag Δ –° | nur Oberarm l: n=132, Ø Betrag Δ 13.3° | nur Unterarm l: n=0, Ø Betrag Δ –° | beide: n=768, Ø Betrag Δ 10.9° |
| elbow_r | keins: n=0, Ø Betrag Δ –° | nur Oberarm r: n=61, Ø Betrag Δ 29.3° | nur Unterarm r: n=0, Ø Betrag Δ –° | beide: n=839, Ø Betrag Δ 26.1° |
| shoulder_l | keins: n=0, Ø Betrag Δ –° | nur Rumpf l: n=0, Ø Betrag Δ –° | nur Oberarm l: n=0, Ø Betrag Δ –° | beide: n=900, Ø Betrag Δ 1.6° |
| shoulder_r | keins: n=0, Ø Betrag Δ –° | nur Rumpf r: n=0, Ø Betrag Δ –° | nur Oberarm r: n=0, Ø Betrag Δ –° | beide: n=900, Ø Betrag Δ 1.7° |
| hip_l | keins: n=0, Ø Betrag Δ –° | nur Rumpf l: n=0, Ø Betrag Δ –° | nur Oberschenkel l: n=0, Ø Betrag Δ –° | beide: n=900, Ø Betrag Δ 14.3° |
| hip_r | keins: n=0, Ø Betrag Δ –° | nur Rumpf r: n=0, Ø Betrag Δ –° | nur Oberschenkel r: n=0, Ø Betrag Δ –° | beide: n=900, Ø Betrag Δ 14.4° |
| knee_l | keins: n=0, Ø Betrag Δ –° | nur Oberschenkel l: n=900, Ø Betrag Δ 17.4° | nur Unterschenkel l: n=0, Ø Betrag Δ –° | beide: n=0, Ø Betrag Δ –° |
| knee_r | keins: n=0, Ø Betrag Δ –° | nur Oberschenkel r: n=860, Ø Betrag Δ 19.9° | nur Unterschenkel r: n=0, Ø Betrag Δ –° | beide: n=40, Ø Betrag Δ 28.8° |
| ankle_l | keins: n=0, Ø Betrag Δ –° | nur Unterschenkel l: n=0, Ø Betrag Δ –° | nur Fuss l: n=900, Ø Betrag Δ 1.3° | beide: n=0, Ø Betrag Δ –° |
| ankle_r | keins: n=0, Ø Betrag Δ –° | nur Unterschenkel r: n=0, Ø Betrag Δ –° | nur Fuss r: n=860, Ø Betrag Δ 1.9° | beide: n=40, Ø Betrag Δ 10.7° |

Konsistenzprüfung: Frames mit beiden Segmenten unkorrigiert und Betrag Δ > 0.5°: 0 (Soll 0).

**Die 8 Korrektur-Episoden mit der grössten Winkelwirkung** (alle Episoden: `korrekturen.csv`)
| Segment | Frames | Start [s] | Dauer [s] | Ø Abw. roh [%] | max Betrag Δ [°] | Winkel |
| --- | --- | --- | --- | --- | --- | --- |
| Oberarm r | 0–899 | 0.00 | 29.97 | -19.4 | 47.2 | elbow_r |
| Unterarm r | 63–337 | 2.10 | 9.16 | -8.4 | 47.2 | elbow_r |
| Unterarm r | 523–839 | 17.41 | 10.56 | -9.5 | 43.9 | elbow_r |
| Unterarm r | 424–510 | 14.12 | 2.90 | -10.0 | 43.2 | elbow_r |
| Unterarm r | 0–1 | 0.00 | 0.07 | 5.2 | 42.6 | elbow_r |
| Unterarm r | 355–421 | 11.82 | 2.23 | -8.4 | 41.2 | elbow_r |
| Unterarm r | 518–521 | 17.25 | 0.13 | -6.2 | 38.3 | elbow_r |
| Unterarm r | 841–895 | 28.00 | 1.83 | -7.8 | 36.1 | elbow_r |

## Segmentlängen gegen Messung (Modell roh)
| Segment | gemessen [m] | MediaPipe | InstantHMR |
| --- | --- | --- | --- |
| Rumpf l | 0.480 | 0.492 (+2.5 %) | 0.562 (+17.0 %) |
| Rumpf r | 0.480 | 0.556 (+15.8 %) | 0.565 (+17.7 %) |
| Oberarm l | 0.315 | 0.198 (-37.3 %) | 0.266 (-15.5 %) |
| Oberarm r | 0.315 | 0.250 (-20.5 %) | 0.254 (-19.4 %) |
| Unterarm l | 0.270 | 0.190 (-29.5 %) | 0.256 (-5.4 %) |
| Unterarm r | 0.270 | 0.230 (-14.9 %) | 0.249 (-7.8 %) |
| Oberschenkel l | 0.475 | 0.416 (-12.4 %) | 0.431 (-9.3 %) |
| Oberschenkel r | 0.475 | 0.342 (-28.1 %) | 0.429 (-9.7 %) |
| Unterschenkel l | 0.415 | 0.399 (-3.9 %) | 0.411 (-1.0 %) |
| Unterschenkel r | 0.415 | 0.432 (+4.1 %) | 0.402 (-3.1 %) |
| Fuss l | 0.200 | 0.062 (-68.8 %) | 0.187 (-6.3 %) |
| Fuss r | 0.200 | 0.120 (-40.0 %) | 0.186 (-7.0 %) |
| Hueftbreite | 0.310 | 0.195 (-37.2 %) | 0.157 (-49.3 %) |
| Schulterbreite | 0.400 | 0.334 (-16.5 %) | 0.321 (-19.7 %) |

## Hinweise
- Ohne Referenzmessung (Goniometer, Marker) beschreibt dieser Bericht Übereinstimmung und Stabilität, nicht Genauigkeit.
- Bias, MAE und Korrelation gelten nur für diese Aufnahme (eine Person, eine Kameraposition).
- Die Tiefenkorrektur übernimmt das Vorzeichen von dz vom Modell. Ein falsches Vorzeichen wird nicht korrigiert.
- Hüftbreite weicht bei beiden Modellen stark ab, vermutlich Definitionsunterschied (Gelenkzentrum gegen gemessene Breite). Hüft- und Schulterbreite werden nicht korrigiert.

## Dateien
01_verlauf.png, 02_uebereinstimmung.png, 03_band.png, 04_qualitaet.png, 05_korrektur.png, 06_korrektur_zeit.png, korrekturen.csv, winkelwirkung.csv, metrics.csv, vergleich.mp4