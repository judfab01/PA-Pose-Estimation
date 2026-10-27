# Vergleich MediaPipe und InstantHMR: Band Reverse Fly Jacks

Video: REVERSEBUTTERFLY | Übung: reverse_fly_jacks | gemeinsame Frames: 273 | erstellt: 2026-10-27T09:20:41

## Überblick
|  | MediaPipe | InstantHMR |
| --- | --- | --- |
| Frames (gesamt) | 2661 | 273 |
| Dauer [s] | 44.3 | 4.5 |
| Pose erkannt [%] | 100.0 | 100.0 |
| Latenz Mittel / 95. Perz. [ms] | 52 / 53 | 75 / 117 |
| Gelenke unter Sichtbarkeit 0.5 [%] | 18.3 | – |
| Jitter Ø (RMS 2. Diff., relevante Winkel) [°/Frame²] | 1.3 | 22.7 |
| Winkelsprünge > 20°/Frame (relevante Winkel) | 1 | 142 |
| Ø |Segmentabweichung| zu gemessen [%] | 23.5 | 12.9 |
| Wiederholungen erkannt | 11 | 0 |

## Übereinstimmung (InstantHMR minus MediaPipe, rohe Winkel)
| Winkel | n | Bias [°] | MAE [°] | RMSE [°] | Korrelation |
| --- | --- | --- | --- | --- | --- |
| elbow_l | 273 | -35.4 | 35.6 | 47.1 | 0.25 |
| elbow_r | 273 | -14.6 | 23.7 | 31.3 | 0.44 |
| shoulder_l | 273 | -6.0 | 8.2 | 10.5 | 0.97 |
| shoulder_r | 273 | +18.8 | 18.8 | 20.5 | 0.97 |
| hip_l | 118 | +1.7 | 2.7 | 3.5 | -0.28 |
| hip_r | 126 | -10.9 | 10.9 | 11.3 | 0.61 |
| knee_l | 118 | +12.5 | 12.5 | 12.8 | 0.17 |
| knee_r | 126 | +8.7 | 8.7 | 9.0 | 0.44 |
| ankle_l | 118 | -9.5 | 15.4 | 20.0 | -0.27 |
| ankle_r | 126 | +7.9 | 8.7 | 10.3 | 0.08 |

## Wiederholungen
| Modell | Nr. | Frame | Zeit [s] | Spitzenwert (Handabstand / Schulterbreite) | bis nächste [s] | Dehnung an Spitze | Ellbogen Mittel an Spitze |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MediaPipe | 1 | 245 | 4.08 | 3.90 | 4.02 | 56 % | 148° |
| MediaPipe | 2 | 486 | 8.10 | 3.85 | 3.48 | 58 % | 146° |
| MediaPipe | 3 | 695 | 11.59 | 3.85 | 3.52 | 60 % | 147° |
| MediaPipe | 4 | 906 | 15.10 | 3.85 | 4.05 | 58 % | 147° |
| MediaPipe | 5 | 1149 | 19.16 | 3.90 | 3.60 | 59 % | 147° |
| MediaPipe | 6 | 1365 | 22.76 | 3.76 | 3.27 | 56 % | 141° |
| MediaPipe | 7 | 1561 | 26.02 | 3.86 | 3.57 | 55 % | 148° |
| MediaPipe | 8 | 1775 | 29.59 | 3.88 | 3.15 | 58 % | 147° |
| MediaPipe | 9 | 1964 | 32.74 | 3.85 | 3.13 | 55 % | 144° |
| MediaPipe | 10 | 2152 | 35.88 | 3.86 | 4.77 | 55 % | 146° |
| MediaPipe | 11 | 2438 | 40.64 | 3.47 | – | 52 % | 137° |

- Wiederholungen erkannt: MediaPipe 11, InstantHMR 0; davon zeitlich zuordenbar (±20 Frames): 0.
- Synchronität Arme/Beine (MediaPipe): Beinspreizung folgt der Armspreizung mit +15 Frames (+250 ms bei 60 fps), Korrelation -0.13.
- Synchronität Arme/Beine (InstantHMR): Beinspreizung folgt der Armspreizung mit -1 Frames (-17 ms bei 60 fps), Korrelation 0.38.

## Band
Bandlänge gegen Handabstand aus der Pose (Korrelation): MediaPipe 0.95, InstantHMR 0.93
Die Bandmessung ist modellunabhängig; eine niedrige Korrelation deutet auf Verdeckung, Fehlerkennung oder ein Band, das nicht zwischen den Händen liegt.

## Segmentlängen gegen Messung
| Segment | gemessen [m] | MediaPipe | InstantHMR |
| --- | --- | --- | --- |
| Rumpf l | 0.480 | 0.477 (-0.6 %) | 0.568 (+18.4 %) |
| Rumpf r | 0.480 | 0.532 (+10.8 %) | 0.564 (+17.5 %) |
| Oberarm l | 0.315 | 0.187 (-40.6 %) | 0.262 (-16.7 %) |
| Oberarm r | 0.315 | 0.243 (-22.9 %) | 0.271 (-14.0 %) |
| Unterarm l | 0.270 | 0.180 (-33.4 %) | 0.251 (-7.2 %) |
| Unterarm r | 0.270 | 0.223 (-17.4 %) | 0.249 (-7.8 %) |
| Oberschenkel l | 0.475 | 0.422 (-11.2 %) | 0.442 (-6.9 %) |
| Oberschenkel r | 0.475 | 0.352 (-26.0 %) | 0.436 (-8.1 %) |
| Unterschenkel l | 0.415 | 0.387 (-6.7 %) | 0.406 (-2.2 %) |
| Unterschenkel r | 0.415 | 0.439 (+5.8 %) | 0.405 (-2.3 %) |
| Fuss l | 0.200 | 0.077 (-61.6 %) | 0.195 (-2.7 %) |
| Fuss r | 0.200 | 0.134 (-33.2 %) | 0.187 (-6.4 %) |
| Hüftbreite | 0.310 | 0.190 (-38.7 %) | 0.161 (-48.0 %) |
| Schulterbreite | 0.400 | 0.318 (-20.4 %) | 0.309 (-22.8 %) |

## Hinweise
- Ohne Referenzmessung (Goniometer, Marker) beschreibt dieser Bericht Übereinstimmung und Stabilität, nicht Genauigkeit.
- Bias, MAE und Korrelation gelten nur für diese Aufnahme (eine Person, eine Kameraposition).
- Winkelsprünge bei InstantHMR hängen mit `detector_stride` zusammen (siehe Auswertung vom 26.10.); Vergleich mit `--detector-stride 1` prüfen.
- Hüftbreite weicht bei beiden Modellen stark ab, vermutlich Definitionsunterschied (Gelenkzentrum gegen gemessene Breite).

## Dateien
01_verlauf.png, 02_uebereinstimmung.png, 03_band.png, 04_qualitaet.png, metrics.csv, vergleich.mp4