# Vergleich MediaPipe und InstantHMR: Reverse Butterfly mit Resistenzband

Video: REVERSEBUTTERFLY2 | Übung: reverse_butterfly | Winkelquelle: tiefenkorrigierte Winkel | gemeinsame Frames: 322 | erstellt: 2026-11-02T10:33:07

## Überblick
|  | MediaPipe | InstantHMR |
| --- | --- | --- |
| Frames (gesamt) | 900 | 322 |
| Dauer [s] | 29.9 | 10.7 |
| Pose erkannt [%] | 100.0 | 100.0 |
| Detektor-Stride (Meta) | – | 1 |
| Frisch ausgewertete Frames `fresh` [%] | 100.0 | 100.0 |
| Latenz Mittel / 95. Perz. [ms] | 91 / 91 | 165 / 187 |
| Gelenke unter Sichtbarkeit 0.5 [%] | 0.9 | – |
| Jitter Ø (RMS 2. Diff., relevante Winkel, gewählte Quelle) [°/Frame²] | 1.8 | 6.8 |
| Jitter Ø roh / korrigiert [°/Frame²] | 2.1 / 1.8 | 2.4 / 6.8 |
| Winkelsprünge > 20°/Frame (relevante Winkel) | 1 | 11 |
| Ø Betrag Segmentabweichung roh zu gemessen [%] | 23.7 | 13.7 |
| Wiederholungen erkannt | 10 | 2 |

## Übereinstimmung (InstantHMR minus MediaPipe, tiefenkorrigierte Winkel)
| Winkel | n | Bias [°] | MAE [°] | RMSE [°] | Korrelation |
| --- | --- | --- | --- | --- | --- |
| elbow_l | 322 | +8.3 | 10.0 | 11.9 | -0.13 |
| elbow_r | 322 | -22.2 | 30.3 | 37.0 | 0.35 |
| shoulder_l | 322 | -1.4 | 4.5 | 6.1 | 0.81 |
| shoulder_r | 322 | +15.0 | 15.2 | 16.5 | 0.66 |
| hip_l | 265 | -22.5 | 22.5 | 22.8 | -0.13 |
| hip_r | 260 | -30.3 | 30.3 | 30.8 | -0.26 |
| knee_l | 265 | +8.2 | 11.3 | 12.0 | -0.02 |
| knee_r | 260 | -24.8 | 24.8 | 24.9 | 0.11 |
| ankle_l | 265 | -6.2 | 6.5 | 8.0 | 0.21 |
| ankle_r | 260 | +20.7 | 20.7 | 21.0 | -0.17 |

Modell-zu-Modell-MAE über die relevanten Winkel: roh 19.71°, nach Tiefenkorrektur 14.96° (näher beieinander). Das ist ein Mass der Übereinstimmung, nicht der Genauigkeit.

## Wiederholungen
| Modell | Nr. | Frame | Zeit [s] | Spitzenwert (Handabstand / Schulterbreite) | bis nächste [s] | Dehnung an Spitze | Ellbogen Mittel an Spitze |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MediaPipe | 1 | 85 | 2.83 | 3.84 | 3.56 | 95 % | 144° |
| MediaPipe | 2 | 192 | 6.39 | 3.87 | 3.13 | 95 % | 144° |
| MediaPipe | 3 | 286 | 9.52 | 3.82 | 3.30 | 95 % | 143° |
| MediaPipe | 4 | 385 | 12.82 | 3.87 | 2.93 | 98 % | 141° |
| MediaPipe | 5 | 473 | 15.75 | 3.71 | 2.46 | 90 % | 139° |
| MediaPipe | 6 | 547 | 18.21 | 3.80 | 2.93 | 93 % | 147° |
| MediaPipe | 7 | 635 | 21.14 | 3.80 | 2.73 | 95 % | 144° |
| MediaPipe | 8 | 717 | 23.88 | 3.83 | 2.46 | 95 % | 145° |
| MediaPipe | 9 | 791 | 26.34 | 3.86 | 2.33 | 97 % | 140° |
| MediaPipe | 10 | 861 | 28.67 | 3.79 | – | 92 % | 138° |
| InstantHMR | 1 | 104 | 3.46 | 4.40 | 2.96 | 115 % | 127° |
| InstantHMR | 2 | 193 | 6.43 | 4.41 | – | 114 % | 128° |

- Wiederholungen erkannt: MediaPipe 10, InstantHMR 2; davon zeitlich zuordenbar (±20 Frames): 2.
- Mittlerer Zeitversatz der Wiederholungsspitzen zwischen den Modellen: 333 ms.

## Band (Länge = Griffpunktabstand, ab v6 modellabhängig)
| Modell | present [%] | Ø band_red_frac | r Handgelenkabstand | r PCA-Referenz | Versatz Griffpunkt minus PCA Ø ± SD [px] | Versatz Ø [% der Länge] | Versatz SD [% der Länge] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MediaPipe | 100.0 | 0.98 | 1.00 | 0.97 | 1.4 ± 38.4 | 0.2 | 6.3 |
| InstantHMR | 100.0 | 0.97 | 1.00 | 0.98 | 22.1 ± 42.2 | 3.5 | 6.6 |
Der Versatz Griffpunkt minus PCA-Referenz zeigt, ob die Fingerspitzen-Griffpunkte konstant gegenüber dem Band verschoben sind (kleine SD: Kalibrierung des Versatzes möglich) oder haltungsabhängig (grosse SD: Bandenden aus den Rotpixeln bestimmen). Die PCA-Referenz nimmt alle roten Pixel im Bild und ist bei rotem Hintergrund oder roter Kleidung verfälscht.

## Segmentkorrektur und Wirkung auf die Winkel

Δ Winkel = `angle_*_cor_raw` minus `angle_*_raw` (beide ungeglättet). Ein Segment gilt als korrigiert, wenn die Tiefenkorrektur in diesem Frame aktiv war (Abweichung zur Solllänge ab 5 %, Rückfall unter 3 %). Sind beide an einem Winkel beteiligten Segmente nicht korrigiert, bleibt die Richtung erhalten und Δ ist 0.

### MediaPipe
**Wann welche Segmente korrigiert wurden**
| Segment | korrigiert [% Frames] | Episoden | Ø Dauer [s] | Ø Betrag Abw. bei Korrektur [%] | Ø Betrag Abw. ohne Korrektur [%] | Fallback [% Frames] | gesperrt vis [% Frames] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Rumpf l | 0.0 | 0 | – | – | 2.5 | 0.0 | 0.0 |
| Rumpf r | 0.0 | 0 | – | – | 15.3 | 0.0 | 0.0 |
| Oberarm l | 99.0 | 1 | 29.40 | 37.8 | 33.7 | 0.0 | 0.0 |
| Oberarm r | 99.0 | 1 | 29.40 | 19.8 | 14.1 | 0.0 | 0.0 |
| Unterarm l | 99.0 | 1 | 29.40 | 30.1 | 25.7 | 39.8 | 0.0 |
| Unterarm r | 94.0 | 13 | 2.15 | 13.9 | 2.9 | 14.9 | 0.0 |
| Oberschenkel l | 0.0 | 0 | – | – | 12.2 | 0.0 | 6.3 |
| Oberschenkel r | 0.0 | 0 | – | – | 28.1 | 0.0 | 6.9 |
| Unterschenkel l | 0.0 | 0 | – | – | 4.3 | 0.0 | 6.3 |
| Unterschenkel r | 0.0 | 0 | – | – | 3.9 | 0.0 | 6.9 |
| Fuss l | 0.0 | 0 | – | – | 69.3 | 0.0 | 0.0 |
| Fuss r | 0.0 | 0 | – | – | 39.9 | 0.0 | 0.0 |

**Wirkung auf die Winkel (Δ = korrigiert minus roh)**
| Winkel | angrenzende Segmente | Ø Betrag Δ [°] | Median Betrag Δ [°] | P95 Betrag Δ [°] | Max Betrag Δ [°] | Ø Δ [°] | Betrag Δ > 1° [% Frames] | Betrag Δ > 5° [% Frames] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| elbow_l | Oberarm l + Unterarm l | 14.41 | 12.50 | 28.5 | 33.3 | -13.88 | 94 | 76 |
| elbow_r | Oberarm r + Unterarm r | 9.10 | 9.29 | 17.0 | 25.8 | 8.95 | 90 | 73 |
| shoulder_l | Rumpf l + Oberarm l | 3.42 | 3.35 | 6.9 | 9.5 | -2.50 | 88 | 19 |
| shoulder_r | Rumpf r + Oberarm r | 15.29 | 15.35 | 19.6 | 23.2 | 15.29 | 99 | 99 |
| hip_l | Rumpf l + Oberschenkel l | 0.00 | 0.00 | 0.0 | 0.0 | 0.00 | 0 | 0 |
| hip_r | Rumpf r + Oberschenkel r | 0.00 | 0.00 | 0.0 | 0.0 | 0.00 | 0 | 0 |
| knee_l | Oberschenkel l + Unterschenkel l | 0.00 | 0.00 | 0.0 | 0.0 | 0.00 | 0 | 0 |
| knee_r | Oberschenkel r + Unterschenkel r | 0.00 | 0.00 | 0.0 | 0.0 | 0.00 | 0 | 0 |
| ankle_l | Unterschenkel l + Fuss l | 0.00 | 0.00 | 0.0 | 0.0 | 0.00 | 0 | 0 |
| ankle_r | Unterschenkel r + Fuss r | 0.00 | 0.00 | 0.0 | 0.0 | 0.00 | 0 | 0 |

**Zuordnung: welche Segmentkorrektur erzeugt welche Winkeländerung**
| Winkel | kein Segment | nur Segment 1 | nur Segment 2 | beide |
| --- | --- | --- | --- | --- |
| elbow_l | keins: n=9, Ø Betrag Δ 0.0° | nur Oberarm l: n=0, Ø Betrag Δ –° | nur Unterarm l: n=0, Ø Betrag Δ –° | beide: n=891, Ø Betrag Δ 14.6° |
| elbow_r | keins: n=9, Ø Betrag Δ 0.0° | nur Oberarm r: n=45, Ø Betrag Δ 12.7° | nur Unterarm r: n=0, Ø Betrag Δ –° | beide: n=846, Ø Betrag Δ 9.0° |
| shoulder_l | keins: n=9, Ø Betrag Δ 0.0° | nur Rumpf l: n=0, Ø Betrag Δ –° | nur Oberarm l: n=891, Ø Betrag Δ 3.5° | beide: n=0, Ø Betrag Δ –° |
| shoulder_r | keins: n=9, Ø Betrag Δ 0.0° | nur Rumpf r: n=0, Ø Betrag Δ –° | nur Oberarm r: n=891, Ø Betrag Δ 15.4° | beide: n=0, Ø Betrag Δ –° |
| hip_l | keins: n=843, Ø Betrag Δ 0.0° | nur Rumpf l: n=0, Ø Betrag Δ –° | nur Oberschenkel l: n=0, Ø Betrag Δ –° | beide: n=0, Ø Betrag Δ –° |
| hip_r | keins: n=838, Ø Betrag Δ 0.0° | nur Rumpf r: n=0, Ø Betrag Δ –° | nur Oberschenkel r: n=0, Ø Betrag Δ –° | beide: n=0, Ø Betrag Δ –° |
| knee_l | keins: n=843, Ø Betrag Δ 0.0° | nur Oberschenkel l: n=0, Ø Betrag Δ –° | nur Unterschenkel l: n=0, Ø Betrag Δ –° | beide: n=0, Ø Betrag Δ –° |
| knee_r | keins: n=838, Ø Betrag Δ 0.0° | nur Oberschenkel r: n=0, Ø Betrag Δ –° | nur Unterschenkel r: n=0, Ø Betrag Δ –° | beide: n=0, Ø Betrag Δ –° |
| ankle_l | keins: n=843, Ø Betrag Δ 0.0° | nur Unterschenkel l: n=0, Ø Betrag Δ –° | nur Fuss l: n=0, Ø Betrag Δ –° | beide: n=0, Ø Betrag Δ –° |
| ankle_r | keins: n=838, Ø Betrag Δ 0.0° | nur Unterschenkel r: n=0, Ø Betrag Δ –° | nur Fuss r: n=0, Ø Betrag Δ –° | beide: n=0, Ø Betrag Δ –° |

Konsistenzprüfung: Frames mit beiden Segmenten unkorrigiert und Betrag Δ > 0.5°: 0 (Soll 0).

**Die 8 Korrektur-Episoden mit der grössten Winkelwirkung** (alle Episoden: `korrekturen.csv`)
| Segment | Frames | Start [s] | Dauer [s] | Ø Abw. roh [%] | max Betrag Δ [°] | Winkel |
| --- | --- | --- | --- | --- | --- | --- |
| Oberarm l | 9–899 | 0.30 | 29.67 | -37.8 | 33.3 | elbow_l |
| Unterarm l | 9–899 | 0.30 | 29.67 | -30.1 | 33.3 | elbow_l |
| Oberarm r | 9–899 | 0.30 | 29.67 | -19.8 | 25.8 | elbow_r |
| Unterarm r | 165–324 | 5.49 | 5.33 | -14.1 | 25.8 | elbow_r |
| Unterarm r | 698–899 | 23.24 | 6.73 | -15.2 | 23.4 | elbow_r |
| Unterarm r | 519–590 | 17.28 | 2.40 | -16.3 | 22.4 | elbow_r |
| Unterarm r | 157–161 | 5.23 | 0.17 | -7.7 | 22.1 | elbow_r |
| Unterarm r | 592–678 | 19.71 | 2.90 | -14.6 | 19.5 | elbow_r |

### InstantHMR
**Wann welche Segmente korrigiert wurden**
| Segment | korrigiert [% Frames] | Episoden | Ø Dauer [s] | Ø Betrag Abw. bei Korrektur [%] | Ø Betrag Abw. ohne Korrektur [%] | Fallback [% Frames] | gesperrt vis [% Frames] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Rumpf l | 100.0 | 1 | 10.63 | 16.7 | – | 83.5 | 0.0 |
| Rumpf r | 100.0 | 1 | 10.63 | 16.8 | – | 51.9 | 0.0 |
| Oberarm l | 100.0 | 1 | 10.63 | 14.9 | – | 0.0 | 0.0 |
| Oberarm r | 100.0 | 1 | 10.63 | 20.1 | – | 0.0 | 0.0 |
| Unterarm l | 88.8 | 4 | 2.36 | 7.3 | 3.9 | 0.0 | 0.0 |
| Unterarm r | 88.5 | 5 | 1.88 | 8.1 | 1.6 | 0.0 | 0.0 |
| Oberschenkel l | 100.0 | 1 | 10.63 | 9.4 | – | 0.0 | 0.0 |
| Oberschenkel r | 100.0 | 1 | 10.63 | 9.7 | – | 0.0 | 0.0 |
| Unterschenkel l | 0.0 | 0 | – | – | 1.4 | 0.0 | 0.0 |
| Unterschenkel r | 0.0 | 0 | – | – | 3.4 | 0.0 | 0.0 |
| Fuss l | 100.0 | 1 | 10.63 | 6.5 | – | 0.0 | 0.0 |
| Fuss r | 100.0 | 1 | 10.63 | 6.8 | – | 0.0 | 0.0 |

**Wirkung auf die Winkel (Δ = korrigiert minus roh)**
| Winkel | angrenzende Segmente | Ø Betrag Δ [°] | Median Betrag Δ [°] | P95 Betrag Δ [°] | Max Betrag Δ [°] | Ø Δ [°] | Betrag Δ > 1° [% Frames] | Betrag Δ > 5° [% Frames] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| elbow_l | Oberarm l + Unterarm l | 13.09 | 12.30 | 18.7 | 22.4 | 13.09 | 100 | 100 |
| elbow_r | Oberarm r + Unterarm r | 38.07 | 38.43 | 58.3 | 64.7 | -6.23 | 100 | 100 |
| shoulder_l | Rumpf l + Oberarm l | 5.40 | 5.18 | 8.0 | 10.8 | 5.24 | 100 | 62 |
| shoulder_r | Rumpf r + Oberarm r | 3.92 | 3.28 | 9.7 | 12.3 | 2.89 | 78 | 26 |
| hip_l | Rumpf l + Oberschenkel l | 25.39 | 26.01 | 28.4 | 35.0 | -25.39 | 100 | 100 |
| hip_r | Rumpf r + Oberschenkel r | 22.11 | 23.30 | 27.7 | 28.7 | -22.11 | 100 | 100 |
| knee_l | Oberschenkel l + Unterschenkel l | 14.47 | 8.17 | 39.5 | 40.2 | -14.29 | 100 | 96 |
| knee_r | Oberschenkel r + Unterschenkel r | 32.16 | 32.23 | 36.3 | 37.2 | -32.16 | 100 | 100 |
| ankle_l | Unterschenkel l + Fuss l | 8.97 | 8.89 | 10.7 | 10.9 | 8.86 | 100 | 97 |
| ankle_r | Unterschenkel r + Fuss r | 6.36 | 5.72 | 8.8 | 9.2 | 6.25 | 100 | 79 |

**Zuordnung: welche Segmentkorrektur erzeugt welche Winkeländerung**
| Winkel | kein Segment | nur Segment 1 | nur Segment 2 | beide |
| --- | --- | --- | --- | --- |
| elbow_l | keins: n=0, Ø Betrag Δ –° | nur Oberarm l: n=36, Ø Betrag Δ 16.6° | nur Unterarm l: n=0, Ø Betrag Δ –° | beide: n=286, Ø Betrag Δ 12.6° |
| elbow_r | keins: n=0, Ø Betrag Δ –° | nur Oberarm r: n=37, Ø Betrag Δ 35.9° | nur Unterarm r: n=0, Ø Betrag Δ –° | beide: n=285, Ø Betrag Δ 38.3° |
| shoulder_l | keins: n=0, Ø Betrag Δ –° | nur Rumpf l: n=0, Ø Betrag Δ –° | nur Oberarm l: n=0, Ø Betrag Δ –° | beide: n=322, Ø Betrag Δ 5.4° |
| shoulder_r | keins: n=0, Ø Betrag Δ –° | nur Rumpf r: n=0, Ø Betrag Δ –° | nur Oberarm r: n=0, Ø Betrag Δ –° | beide: n=322, Ø Betrag Δ 3.9° |
| hip_l | keins: n=0, Ø Betrag Δ –° | nur Rumpf l: n=0, Ø Betrag Δ –° | nur Oberschenkel l: n=0, Ø Betrag Δ –° | beide: n=322, Ø Betrag Δ 25.4° |
| hip_r | keins: n=0, Ø Betrag Δ –° | nur Rumpf r: n=0, Ø Betrag Δ –° | nur Oberschenkel r: n=0, Ø Betrag Δ –° | beide: n=322, Ø Betrag Δ 22.1° |
| knee_l | keins: n=0, Ø Betrag Δ –° | nur Oberschenkel l: n=322, Ø Betrag Δ 14.5° | nur Unterschenkel l: n=0, Ø Betrag Δ –° | beide: n=0, Ø Betrag Δ –° |
| knee_r | keins: n=0, Ø Betrag Δ –° | nur Oberschenkel r: n=322, Ø Betrag Δ 32.2° | nur Unterschenkel r: n=0, Ø Betrag Δ –° | beide: n=0, Ø Betrag Δ –° |
| ankle_l | keins: n=0, Ø Betrag Δ –° | nur Unterschenkel l: n=0, Ø Betrag Δ –° | nur Fuss l: n=322, Ø Betrag Δ 9.0° | beide: n=0, Ø Betrag Δ –° |
| ankle_r | keins: n=0, Ø Betrag Δ –° | nur Unterschenkel r: n=0, Ø Betrag Δ –° | nur Fuss r: n=322, Ø Betrag Δ 6.4° | beide: n=0, Ø Betrag Δ –° |

Konsistenzprüfung: Frames mit beiden Segmenten unkorrigiert und Betrag Δ > 0.5°: 0 (Soll 0).

**Die 8 Korrektur-Episoden mit der grössten Winkelwirkung** (alle Episoden: `korrekturen.csv`)
| Segment | Frames | Start [s] | Dauer [s] | Ø Abw. roh [%] | max Betrag Δ [°] | Winkel |
| --- | --- | --- | --- | --- | --- | --- |
| Oberarm r | 0–321 | 0.00 | 10.72 | -20.1 | 64.7 | elbow_r |
| Unterarm r | 63–321 | 2.10 | 8.62 | -8.5 | 64.7 | elbow_r |
| Unterarm r | 56–57 | 1.86 | 0.07 | 6.2 | 44.0 | elbow_r |
| Unterarm r | 0–1 | 0.00 | 0.07 | 5.2 | 42.6 | elbow_r |
| Oberschenkel l | 0–321 | 0.00 | 10.72 | -9.4 | 40.2 | knee_l |
| Unterarm r | 59–61 | 1.97 | 0.10 | -6.0 | 39.4 | elbow_r |
| Oberschenkel r | 0–321 | 0.00 | 10.72 | -9.7 | 37.2 | knee_r |
| Unterarm r | 13–31 | 0.43 | 0.63 | -4.5 | 37.0 | elbow_r |

## Segmentlängen gegen Messung (Modell roh)
| Segment | gemessen [m] | MediaPipe | InstantHMR |
| --- | --- | --- | --- |
| Rumpf l | 0.480 | 0.492 (+2.5 %) | 0.566 (+17.8 %) |
| Rumpf r | 0.480 | 0.556 (+15.8 %) | 0.568 (+18.3 %) |
| Oberarm l | 0.315 | 0.198 (-37.3 %) | 0.267 (-15.3 %) |
| Oberarm r | 0.315 | 0.250 (-20.5 %) | 0.253 (-19.8 %) |
| Unterarm l | 0.270 | 0.190 (-29.5 %) | 0.254 (-5.8 %) |
| Unterarm r | 0.270 | 0.230 (-14.9 %) | 0.250 (-7.3 %) |
| Oberschenkel l | 0.475 | 0.416 (-12.4 %) | 0.431 (-9.3 %) |
| Oberschenkel r | 0.475 | 0.342 (-28.1 %) | 0.429 (-9.7 %) |
| Unterschenkel l | 0.415 | 0.399 (-3.9 %) | 0.410 (-1.3 %) |
| Unterschenkel r | 0.415 | 0.432 (+4.1 %) | 0.401 (-3.5 %) |
| Fuss l | 0.200 | 0.062 (-68.8 %) | 0.187 (-6.7 %) |
| Fuss r | 0.200 | 0.120 (-40.0 %) | 0.187 (-6.6 %) |
| Hueftbreite | 0.310 | 0.195 (-37.2 %) | 0.155 (-49.9 %) |
| Schulterbreite | 0.400 | 0.334 (-16.5 %) | 0.318 (-20.5 %) |

## Hinweise
- Ohne Referenzmessung (Goniometer, Marker) beschreibt dieser Bericht Übereinstimmung und Stabilität, nicht Genauigkeit.
- Bias, MAE und Korrelation gelten nur für diese Aufnahme (eine Person, eine Kameraposition).
- Die Tiefenkorrektur übernimmt das Vorzeichen von dz vom Modell. Ein falsches Vorzeichen wird nicht korrigiert.
- Hüftbreite weicht bei beiden Modellen stark ab, vermutlich Definitionsunterschied (Gelenkzentrum gegen gemessene Breite). Hüft- und Schulterbreite werden nicht korrigiert.

## Dateien
01_verlauf.png, 02_uebereinstimmung.png, 03_band.png, 04_qualitaet.png, 05_korrektur.png, 06_korrektur_zeit.png, korrekturen.csv, winkelwirkung.csv, metrics.csv, vergleich.mp4