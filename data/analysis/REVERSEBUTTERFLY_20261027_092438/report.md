# Vergleich MediaPipe und InstantHMR: Band Reverse Fly Jacks

Video: REVERSEBUTTERFLY | Übung: reverse_fly_jacks | gemeinsame Frames: 2661 | erstellt: 2026-10-27T09:24:54

## Überblick
|  | MediaPipe | InstantHMR |
| --- | --- | --- |
| Frames (gesamt) | 2661 | 2661 |
| Dauer [s] | 44.3 | 44.3 |
| Pose erkannt [%] | 100.0 | 100.0 |
| Latenz Mittel / 95. Perz. [ms] | 52 / 53 | 74 / 116 |
| Gelenke unter Sichtbarkeit 0.5 [%] | 18.3 | – |
| Jitter Ø (RMS 2. Diff., relevante Winkel) [°/Frame²] | 1.3 | 17.7 |
| Winkelsprünge > 20°/Frame (relevante Winkel) | 1 | 836 |
| Ø |Segmentabweichung| zu gemessen [%] | 23.5 | 12.6 |
| Wiederholungen erkannt | 11 | 11 |

## Übereinstimmung (InstantHMR minus MediaPipe, rohe Winkel)
| Winkel | n | Bias [°] | MAE [°] | RMSE [°] | Korrelation |
| --- | --- | --- | --- | --- | --- |
| elbow_l | 2661 | -23.7 | 23.8 | 31.7 | 0.52 |
| elbow_r | 2652 | -5.9 | 14.3 | 18.8 | 0.61 |
| shoulder_l | 2661 | -3.8 | 7.5 | 8.7 | 0.93 |
| shoulder_r | 2661 | +18.9 | 18.9 | 20.5 | 0.90 |
| hip_l | 1238 | +2.2 | 2.5 | 3.2 | -0.08 |
| hip_r | 1231 | -9.1 | 9.1 | 9.3 | 0.28 |
| knee_l | 1238 | +14.2 | 14.2 | 14.3 | 0.30 |
| knee_r | 1231 | +9.2 | 9.2 | 9.5 | 0.25 |
| ankle_l | 1238 | -6.2 | 7.8 | 11.5 | -0.12 |
| ankle_r | 1231 | +8.6 | 9.0 | 10.0 | -0.08 |

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
| InstantHMR | 1 | 293 | 4.88 | 4.50 | 4.12 | 59 % | 145° |
| InstantHMR | 2 | 540 | 9.00 | 4.34 | 3.10 | 60 % | 139° |
| InstantHMR | 3 | 726 | 12.10 | 4.36 | 3.99 | 63 % | 143° |
| InstantHMR | 4 | 965 | 16.09 | 4.40 | 3.57 | 60 % | 144° |
| InstantHMR | 5 | 1179 | 19.66 | 4.39 | 3.50 | 62 % | 140° |
| InstantHMR | 6 | 1389 | 23.16 | 4.26 | 3.45 | 58 % | 134° |
| InstantHMR | 7 | 1596 | 26.61 | 4.43 | 3.55 | 58 % | 140° |
| InstantHMR | 8 | 1809 | 30.16 | 4.43 | 3.25 | 58 % | 142° |
| InstantHMR | 9 | 2004 | 33.41 | 4.44 | 3.08 | 55 % | 142° |
| InstantHMR | 10 | 2189 | 36.49 | 4.46 | 4.12 | 57 % | 148° |
| InstantHMR | 11 | 2436 | 40.61 | 4.14 | – | 56 % | 132° |

- Wiederholungen erkannt: MediaPipe 11, InstantHMR 11; davon zeitlich zuordenbar (±20 Frames): 1.
- Mittlerer Zeitversatz der Wiederholungsspitzen zwischen den Modellen: 33 ms.
- Synchronität Arme/Beine (MediaPipe): Beinspreizung folgt der Armspreizung mit +15 Frames (+250 ms bei 60 fps), Korrelation -0.13.
- Synchronität Arme/Beine (InstantHMR): Beinspreizung folgt der Armspreizung mit +1 Frames (+17 ms bei 60 fps), Korrelation 0.58.

## Band
Bandlänge gegen Handabstand aus der Pose (Korrelation): MediaPipe 0.95, InstantHMR 0.96
Die Bandmessung ist modellunabhängig; eine niedrige Korrelation deutet auf Verdeckung, Fehlerkennung oder ein Band, das nicht zwischen den Händen liegt.

## Segmentlängen gegen Messung
| Segment | gemessen [m] | MediaPipe | InstantHMR |
| --- | --- | --- | --- |
| Rumpf l | 0.480 | 0.477 (-0.6 %) | 0.568 (+18.3 %) |
| Rumpf r | 0.480 | 0.532 (+10.8 %) | 0.561 (+17.0 %) |
| Oberarm l | 0.315 | 0.187 (-40.6 %) | 0.262 (-16.8 %) |
| Oberarm r | 0.315 | 0.243 (-22.9 %) | 0.267 (-15.4 %) |
| Unterarm l | 0.270 | 0.180 (-33.4 %) | 0.251 (-7.0 %) |
| Unterarm r | 0.270 | 0.223 (-17.4 %) | 0.251 (-7.0 %) |
| Oberschenkel l | 0.475 | 0.422 (-11.2 %) | 0.441 (-7.2 %) |
| Oberschenkel r | 0.475 | 0.352 (-26.0 %) | 0.437 (-8.0 %) |
| Unterschenkel l | 0.415 | 0.387 (-6.7 %) | 0.408 (-1.6 %) |
| Unterschenkel r | 0.415 | 0.439 (+5.8 %) | 0.407 (-2.0 %) |
| Fuss l | 0.200 | 0.077 (-61.6 %) | 0.195 (-2.4 %) |
| Fuss r | 0.200 | 0.134 (-33.2 %) | 0.189 (-5.4 %) |
| Hüftbreite | 0.310 | 0.190 (-38.7 %) | 0.163 (-47.5 %) |
| Schulterbreite | 0.400 | 0.318 (-20.4 %) | 0.315 (-21.3 %) |

## Hinweise
- Ohne Referenzmessung (Goniometer, Marker) beschreibt dieser Bericht Übereinstimmung und Stabilität, nicht Genauigkeit.
- Bias, MAE und Korrelation gelten nur für diese Aufnahme (eine Person, eine Kameraposition).
- Winkelsprünge bei InstantHMR hängen mit `detector_stride` zusammen (siehe Auswertung vom 26.10.); Vergleich mit `--detector-stride 1` prüfen.
- Hüftbreite weicht bei beiden Modellen stark ab, vermutlich Definitionsunterschied (Gelenkzentrum gegen gemessene Breite).

## Dateien
01_verlauf.png, 02_uebereinstimmung.png, 03_band.png, 04_qualitaet.png, metrics.csv, vergleich.mp4