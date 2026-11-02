"""
run_mediapipe.py (v7.1)
MediaPipe Pose Landmarker (Heavy, CPU) + rotes Theraband (Griffpunkt zu Griffpunkt) -> CSV, SegDev-CSV, Meta-JSON, HUD.
Gleicher Aufbau und gleiches Ausgabeformat wie run_instanthmr.py. Eigenständig, keine weiteren Dateien nötig.

Änderungen v7.1 gegenüber v7 (gemessen an REVERSEBUTTERFLY2: Knie 148 Grad roh, 106 Grad mit v7):
  - Tiefenkorrektur nur noch an Oberarm und Unterarm (--depth-segs arms, Standard). Rumpf, Beine, Füsse
    werden nur skaliert, ihre Winkel bleiben die des Modells. --depth-segs all = Verhalten von v7.
  - Skala s = Median von Pixellänge / Solllänge über Rumpf und Beine (vorher Maximum): Eine einzelne zu kurze
    Tabellenlänge hebt s nicht mehr an. Pixel/Soll aller Segmente stehen in der Meta-JSON (scale).

Änderungen v7 gegenüber v6 (Segmentkorrektur nach Taylor, "Reconstruction of Articulated Objects from Point
Correspondences in a Single Uncalibrated Image", Gl. 2 und 3):
  - DepthCorrector, Modus taylor (Standard, --segment-mode): dx, dy der Tiefenkorrektur kommen aus Bildpixeln
    geteilt durch die Bildskala s (px/m) statt aus den Modell-Weltkoordinaten. s: siehe v7.1. Ein falscher
    Modellmassstab (MediaPipe Arme, InstantHMR Rumpf) wird so nicht mehr als Tiefe gelesen.
    Hysterese 5 % / 3 %, Vorzeichen-Gedächtnis, Visibility-Sperre, Fallback bleiben. --segment-mode depth = v6.
  - CSV (am Ende angehängt): scale_px_per_m, band_len_m (Griffabstand / s), stretch_abs_pct
    (absolute Dehnung gegen --band-rest-m, unabhängig von L0), angle_*_v6_raw (v6-Korrektur als Referenz).
  - Meta: depth_correction.mode / scale (s und Pixel/Soll je Segment), band_rest_m.
  - Abfrage und HUD: freie Bandlänge, Segmentmodus, s.

Änderungen v6 gegenüber v5:
  - Tiefenkorrektur immer an (DepthCorrector, Hysterese 5 % / 3 %): Segmente mit Abweichung zur
    Solllänge bekommen neues z (dx, dy bleiben), übrige Segmente werden richtungserhaltend skaliert.
    Gelenke mit Visibility < 0.5 werden nie tiefenkorrigiert.
  - CSV: angle_*_raw (Rohkoordinaten), angle_*_cor_raw (korrigiert, ungeglättet), angle_*
    (korrigiert, EMA). Neue Spalten fresh, conf, grip_*, band_len_pca_px, band_red_frac.
  - <stem>_segdev.csv: Segmentabweichung pro Frame (Massstab- gegen Tiefenfehler) und Korrektur-Status.
  - Band: Länge = Abstand der Griffpunkte (Mittelwert Pinky, Zeigefinger, Daumen je Hand, nur sichtbare
    Punkte), rote Pixel im Korridor nur als Validierung, kein PCA-Fallback.
    Gilt für die Reverse Butterfly (Hand zu Hand).

CSV-Spalte fresh: 1 bei erkannter Pose. conf = mittlere Visibility der 14 Gelenke.

Start ohne Argumente (Run-Button der IDE): interaktive Abfrage von Quelle, Pfad,
Übung und Band. Letzte Eingaben werden gemerkt.

Beispiele (Kommandozeile):
  python run_mediapipe.py --video "/Users/fabrice/Desktop/PA Data/REVERSEBUTTERFLY.MOV" --exercise reversebutterfly
  python run_mediapipe.py --camera 0 --exercise reversebutterfly
"""
import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# =============================================================================
# COMMON START (in beiden Dateien byte-identisch, nicht einzeln ändern)
# =============================================================================
NAN = float("nan")

# ---------------------------------------------------------------- Skelett
JOINTS = [
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
    "l_foot", "r_foot",
]
JI = {n: i for i, n in enumerate(JOINTS)}

# (Name, A, B, C): Winkel bei B zwischen B->A und B->C (3D)
ANGLES = [
    ("elbow_l", "l_shoulder", "l_elbow", "l_wrist"),
    ("elbow_r", "r_shoulder", "r_elbow", "r_wrist"),
    ("shoulder_l", "l_hip", "l_shoulder", "l_elbow"),
    ("shoulder_r", "r_hip", "r_shoulder", "r_elbow"),
    ("hip_l", "l_shoulder", "l_hip", "l_knee"),
    ("hip_r", "r_shoulder", "r_hip", "r_knee"),
    ("knee_l", "l_hip", "l_knee", "l_ankle"),
    ("knee_r", "r_hip", "r_knee", "r_ankle"),
    ("ankle_l", "l_knee", "l_ankle", "l_foot"),
    ("ankle_r", "r_knee", "r_ankle", "r_foot"),
]

BONES = [
    ("l_shoulder", "r_shoulder"), ("l_hip", "r_hip"),
    ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"),
    ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("l_ankle", "l_foot"),
    ("r_hip", "r_knee"), ("r_knee", "r_ankle"), ("r_ankle", "r_foot"),
]

VIS_MIN = 0.5
SMOOTH_ALPHA = 0.4

# Tiefenkorrektur (Startwerte)
DEV_ON = 0.05    # Segment wird tiefenkorrigiert ab 5 % Abweichung zur Solllänge
DEV_OFF = 0.03   # und wieder richtungserhaltend skaliert unter 3 % (Hysterese)
DZ_EPS = 0.02    # m: unter |dz| < eps bleibt das dz-Vorzeichen vom Vorframe
S_WINDOW = 90    # Frames des gleitenden Medians der Bildskala s (px/m)
S_MIN_N = 10     # ab so vielen gültigen Frames wird s verwendet (davor Modus depth)


@dataclass
class Pose:
    detected: bool
    w: np.ndarray    # (14, 3) Weltkoordinaten in m
    img: np.ndarray  # (14, 2) Bild, normiert 0..1
    vis: np.ndarray  # (14,)  Sichtbarkeit 0..1
    grip: np.ndarray = field(default_factory=lambda: np.full((2, 2), NAN))   # (links, rechts) Griffpunkt, normiert 0..1
    grip_ok: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=bool))
    fresh: int = 0   # 1 = Modell/Detektor hat diesen Frame frisch ausgewertet
    conf: float = NAN  # MediaPipe: mittlere Visibility der 14 Gelenke; InstantHMR: Detektor-Konfidenz


def empty_pose():
    return Pose(False, np.full((len(JOINTS), 3), NAN),
                np.full((len(JOINTS), 2), NAN), np.zeros(len(JOINTS)))


# ---------------------------------------------------------------- Segmentlängen
# Reale Segmentlängen von Fabrice in Meter (gemittelt, aus PE_MediaPipe_3D_BAND.py, 2.2).
# Gelten nur für diesen Probanden. Reihenfolge ist funktional: proximal vor distal
# (Rumpf, Oberarm, Unterarm, Oberschenkel, Unterschenkel, Fuss). Umsortieren bricht die Kette.
SEGMENT_LENGTHS = {
    ("l_hip", "l_shoulder"): 0.480,    # Rumpf: Hüftgelenk bis Schultergelenk
    ("r_hip", "r_shoulder"): 0.480,
    ("l_shoulder", "l_elbow"): 0.315,  # Oberarm
    ("r_shoulder", "r_elbow"): 0.315,
    ("l_elbow", "l_wrist"): 0.270,     # Unterarm
    ("r_elbow", "r_wrist"): 0.270,
    ("l_hip", "l_knee"): 0.475,        # Oberschenkel
    ("r_hip", "r_knee"): 0.475,
    ("l_knee", "l_ankle"): 0.415,      # Unterschenkel
    ("r_knee", "r_ankle"): 0.415,
    ("l_ankle", "l_foot"): 0.200,      # Fuss: Knöchel bis Fussspitze
    ("r_ankle", "r_foot"): 0.200,
}
BODY_WIDTHS = {                        # nur Vergleichsreport, werden nicht erzwungen
    ("l_hip", "r_hip"): 0.310,         # Hüftbreite
    ("l_shoulder", "r_shoulder"): 0.400,  # Schulterbreite
}
CHECK_PAIRS = {**SEGMENT_LENGTHS, **BODY_WIDTHS}
CHECK_KEYS = list(CHECK_PAIRS)
SEG_KEYS = list(SEGMENT_LENGTHS)
S_SEGS = [k for k in SEG_KEYS if k[0].endswith(("hip", "knee"))]    # Skala aus Rumpf, Oberschenkel, Unterschenkel
R_SEGS = [k for k in SEG_KEYS if not k[1].endswith("_foot")]  # Diagnose Pixel/Soll (ohne Fuss)
ARM_SEGS = [k for k in SEG_KEYS if k[0].endswith(("shoulder", "elbow"))]   # Oberarm, Unterarm

# Kinematische Kette: Gelenk -> direkte Nachfolger (für die Korrektur)
CHILDREN = {
    "l_shoulder": ["l_elbow"], "r_shoulder": ["r_elbow"],
    "l_elbow": ["l_wrist"], "r_elbow": ["r_wrist"],
    "l_hip": ["l_knee"], "r_hip": ["r_knee"],
    "l_knee": ["l_ankle"], "r_knee": ["r_ankle"],
    "l_ankle": ["l_foot"], "r_ankle": ["r_foot"],
}


def descendants(name):
    out = []
    for ch in CHILDREN.get(name, []):
        out.append(ch)
        out += descendants(ch)
    return out


class DepthCorrector:
    """
    Setzt jedes Segment auf die reale Länge (proximal vor distal, Nachfolger werden
    starr mitverschoben). Pro Segment zwei Zustände, entschieden mit Hysterese:

      aus (Abweichung <= DEV_OFF): Skalierung entlang der Modellrichtung.
                                   Richtung bleibt, Gelenkwinkel ändern sich NICHT.
      an  (Abweichung >= DEV_ON):  Tiefenkorrektur nach Taylor (2000), Gl. 2:
                                   dz = sign * sqrt(L^2 - dx^2 - dy^2).
                                   Richtung ändert sich, Gelenkwinkel ändern sich.
                                   Ist dx^2 + dy^2 > L^2, ist z nicht lösbar: dz = 0
                                   und xy wird auf L skaliert (Fallback, wird gezählt).

    Modus (mode), nur für Zustand "an":
      "taylor" (Standard): dx, dy = Bildpixel-Differenz / s. Die Bildskala s (px/m) ist der
          Median von Pixellänge / Solllänge über Rumpf, Oberschenkel und Unterschenkel (S_SEGS,
          stehen in der Übung in der Bildebene, dort gilt Gleichheit, Taylor Gl. 3), gleitender
          Median über S_WINDOW Frames. Ein falscher Modellmassstab wird so nicht als Tiefe gelesen.
          Die Achsenvorzeichen Bild <-> Modell werden aus den Segmentrichtungen bestimmt,
          die Hüften werden aus Pixelbreite / s gesetzt. Vor S_MIN_N gültigen Frames gilt "depth".
      "depth" (v6): dx, dy aus den Modell-Weltkoordinaten.

    Tiefenkorrigiert werden nur die Segmente in depth_segs (Standard: Oberarm, Unterarm = ARM_SEGS).
    Alle anderen Segmente (Rumpf, Beine, Füsse) werden nur richtungserhaltend skaliert: Ihre
    Winkel bleiben die des Modells. Grund: Eine Tabellenlänge, die nicht zum Körper im Bild passt,
    würde dort als Tiefe gelesen (gemessen: Knie 148 Grad roh, 106 Grad nach Tiefenkorrektur).
    Segmente mit Sichtbarkeit < VIS_MIN (MediaPipe) werden nie tiefenkorrigiert
    (ihre Tiefe ist geraten), sondern richtungserhaltend skaliert.
    Das dz-Vorzeichen kommt vom Modell und wird nicht korrigiert. Bei |dz| < DZ_EPS
    bleibt das Vorzeichen des letzten Frames, damit es nicht um 0 flackert.
    Achsenannahme: z = Tiefe (MediaPipe-Welt und InstantHMR joints_3d_local, beide
    kameraparallel, Y-down). s wird bei Posenverlust nicht zurückgesetzt.
    """

    def __init__(self, mode="taylor", depth_segs=ARM_SEGS):
        self.mode = mode
        self.depth_segs = set(depth_segs)
        self.active = {k: False for k in SEG_KEYS}
        self.sign = {k: 0.0 for k in SEG_KEYS}
        self.last_on = {k: 0 for k in SEG_KEYS}
        self.stats = {k: dict(n=0, depth_on=0, fallback=0, gated=0) for k in SEG_KEYS}
        self.s, self.s_hist, self.s_trace = NAN, [], []
        self.ratio = {k: [] for k in R_SEGS}      # Pixellänge / Solllänge je Segment (Diagnose)

    def reset(self):
        """Pose verloren: Hysterese-Zustand und Vorzeichen-Gedächtnis zurücksetzen."""
        for k in SEG_KEYS:
            self.active[k] = False
            self.sign[k] = 0.0
            self.last_on[k] = 0

    def _scale(self, px, vis):
        """Bildskala s (px/m): gleitender Median (S_WINDOW Frames) des Medians der Pixellänge / Solllänge von Rumpf und Beinen."""
        r = []
        for k in R_SEGS:
            i, j = JI[k[0]], JI[k[1]]
            if min(vis[i], vis[j]) >= VIS_MIN:
                v = float(np.linalg.norm(px[j] - px[i])) / SEGMENT_LENGTHS[k]
                if np.isfinite(v):
                    self.ratio[k].append(v)
                    if k in S_SEGS:
                        r.append(v)
        if len(r) >= 3:
            self.s_hist.append(float(np.median(r)))
            del self.s_hist[:-S_WINDOW]
        if len(self.s_hist) >= S_MIN_N:
            self.s = float(np.median(self.s_hist))
        self.s_trace.append(self.s)

    @staticmethod
    def _axes(w, px, vis):
        """Vorzeichen (x, y) zwischen Modellachsen und Bildachsen aus den Segmentrichtungen."""
        acc = np.zeros(2)
        for a, b in SEG_KEYS:
            i, j = JI[a], JI[b]
            if min(vis[i], vis[j]) >= VIS_MIN:
                acc += (w[j, :2] - w[i, :2]) * (px[j] - px[i])
        return np.where(np.nan_to_num(acc) >= 0, 1.0, -1.0)

    def _roots(self, w, px, vis, m):
        """Hüftbreite aus Pixeln / s. Jede Körperseite wird starr verschoben (Segmentvektoren bleiben)."""
        li, ri = JI["l_hip"], JI["r_hip"]
        if min(vis[li], vis[ri]) < VIS_MIN or not np.all(np.isfinite(px[[li, ri]])):
            return
        mid_w, mid_p = (w[li, :2] + w[ri, :2]) / 2.0, (px[li] + px[ri]) / 2.0
        for side, i in (("l", li), ("r", ri)):
            delta = np.zeros(3)
            delta[:2] = mid_w + m * (px[i] - mid_p) / self.s - w[i, :2]
            for n in ("hip", "shoulder", "elbow", "wrist", "knee", "ankle", "foot"):
                w[JI[f"{side}_{n}"]] += delta

    def __call__(self, w, vis, px=None):
        """w: (14, 3) Modellwelt, vis: (14,), px: (14, 2) Bildpixel (nur Modus taylor)."""
        w = w.copy()
        m = None
        if self.mode == "taylor" and px is not None:
            self._scale(px, vis)
            if np.isfinite(self.s):
                m = self._axes(w, px, vis)
                self._roots(w, px, vis, m)
        for key, length in SEGMENT_LENGTHS.items():
            parent, child = key
            p, c = JI[parent], JI[child]
            st = self.stats[key]
            d = w[c] - w[p]
            n = float(np.linalg.norm(d))
            if not np.isfinite(n) or n < 1e-9:
                self.last_on[key] = 0
                continue
            st["n"] += 1
            dev = abs(n - length) / length
            if dev >= DEV_ON:
                self.active[key] = True
            elif dev <= DEV_OFF:
                self.active[key] = False
            gated = min(vis[p], vis[c]) < VIS_MIN
            if gated:
                st["gated"] += 1
            use_depth = self.active[key] and not gated and key in self.depth_segs
            if use_depth:
                dx, dy, dz = d
                if m is not None and np.all(np.isfinite(px[[p, c]])):
                    dx, dy = m * (px[c] - px[p]) / self.s
                rxy2 = float(dx * dx + dy * dy)
                if rxy2 > length * length:
                    s = length / np.sqrt(rxy2)
                    nd = np.array([dx * s, dy * s, 0.0])
                    st["fallback"] += 1
                else:
                    if abs(dz) >= DZ_EPS:
                        self.sign[key] = 1.0 if dz > 0 else -1.0
                    elif self.sign[key] == 0.0:
                        self.sign[key] = 1.0 if dz >= 0 else -1.0
                    nd = np.array([dx, dy, self.sign[key] * np.sqrt(length * length - rxy2)])
                st["depth_on"] += 1
            else:
                nd = d / n * length
            self.last_on[key] = int(use_depth)
            delta = (w[p] + nd) - w[c]
            for name in [child] + descendants(child):
                w[JI[name]] += delta
        return w

    def report(self):
        out = {}
        for k, st in self.stats.items():
            n = max(st["n"], 1)
            out[f"{k[0]}-{k[1]}"] = dict(
                frames=st["n"], depth_on_pct=round(100.0 * st["depth_on"] / n, 1),
                fallback_pct=round(100.0 * st["fallback"] / n, 1),
                gated_pct=round(100.0 * st["gated"] / n, 1))
        return out

    def scale_report(self):
        tr = np.array([v for v in self.s_trace if np.isfinite(v)])
        out = dict(mode=self.mode, window_frames=S_WINDOW, min_frames=S_MIN_N, frames_with_scale=int(tr.size))
        if tr.size:
            out.update(px_per_m_median=round(float(np.median(tr)), 1),
                       px_per_m_p5=round(float(np.percentile(tr, 5)), 1),
                       px_per_m_p95=round(float(np.percentile(tr, 95)), 1))
        out["segment_px_per_m_median"] = {f"{a}-{b}": round(float(np.median(v)), 1)
                                          for (a, b), v in self.ratio.items() if v}
        return out


def print_scale_report(r):
    print(f"\nSegmentmodus: {r['mode']}")
    if "px_per_m_median" in r:
        print(f"  Bildskala s: Median {r['px_per_m_median']:.0f} px/m "
              f"(P5 {r['px_per_m_p5']:.0f}, P95 {r['px_per_m_p95']:.0f}) | Frames mit s: {r['frames_with_scale']}")
        print("  Pixellänge / Solllänge je Segment (Median, px/m). Segmente in der Bildebene liegen nahe s,")
        print("  deutlich darunter = Segment zur Kamera geneigt oder Solllänge zu gross:")
        for k, v in r["segment_px_per_m_median"].items():
            print(f"    {k:24s} {v:7.0f}")
    else:
        print("  Keine Skala geschätzt (Modus depth oder zu wenige gültige Frames).")


def print_depth_report(report):
    print("\nTiefenkorrektur pro Segment (Anteil Frames):")
    for name, r in report.items():
        print(f"  {name:24s} an {r['depth_on_pct']:5.1f} % | Fallback {r['fallback_pct']:5.1f} % "
              f"| gesperrt (vis) {r['gated_pct']:5.1f} % | n={r['frames']}")


def segment_distances(w, vis):
    """Rohe Abstände aller CHECK_KEYS (NaN, wenn ein Gelenk nicht sichtbar ist)."""
    out = []
    for (a, b) in CHECK_KEYS:
        ia, ib = JI[a], JI[b]
        if vis[ia] >= VIS_MIN and vis[ib] >= VIS_MIN:
            out.append(float(np.linalg.norm(w[ia] - w[ib])))
        else:
            out.append(NAN)
    return out


def collect_segment_samples(samples, dists):
    for key, d in zip(CHECK_KEYS, dists):
        if np.isfinite(d):
            samples[key].append(d)


def segment_check_report(samples, measured):
    """Median der rohen Modell-Abstände gegen gemessene Längen. deviation_pct = (Modell - gemessen) / gemessen."""
    out = {}
    for key, length in measured.items():
        vals = samples.get(key, [])
        name = f"{key[0]}-{key[1]}"
        if vals:
            med = float(np.median(vals))
            out[name] = {"measured_m": length, "model_median_m": round(med, 4),
                         "deviation_pct": round((med - length) / length * 100.0, 1), "n": len(vals)}
        else:
            out[name] = {"measured_m": length, "model_median_m": None, "deviation_pct": None, "n": 0}
    return out


def print_segment_report(report):
    print("\nSegmentvergleich (gemessen vs. Modell-Median, roh):")
    for name, r in report.items():
        if r["n"] == 0:
            print(f"  {name:24s} keine gültigen Frames")
        else:
            print(f"  {name:24s} gemessen {r['measured_m']:.3f} m | Modell {r['model_median_m']:.3f} m "
                  f"| Abw. {r['deviation_pct']:+.1f} % | n={r['n']}")


# ---------------------------------------------------------------- Winkel
def joint_angle(a, b, c):
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if not np.isfinite(n) or n < 1e-9:
        return NAN
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def compute_angles(pose, w=None):
    w = pose.w if w is None else w
    out = []
    for _, a, b, c in ANGLES:
        ia, ib, ic = JI[a], JI[b], JI[c]
        if pose.detected and min(pose.vis[ia], pose.vis[ib], pose.vis[ic]) >= VIS_MIN:
            out.append(joint_angle(w[ia], w[ib], w[ic]))
        else:
            out.append(NAN)
    return out


class Ema:
    """EMA pro Kanal. NaN setzt den Zustand zurück (keine Glättung über Lücken)."""

    def __init__(self, alpha, n):
        self.a = alpha
        self.s = [NAN] * n

    def __call__(self, vals):
        out = []
        for i, v in enumerate(vals):
            if not np.isfinite(v):
                self.s[i] = NAN
            elif not np.isfinite(self.s[i]):
                self.s[i] = v
            else:
                self.s[i] = self.a * v + (1 - self.a) * self.s[i]
            out.append(self.s[i])
        return out


# ---------------------------------------------------------------- Band (Griffpunkt zu Griffpunkt)
BAND_WORK_W = 640          # Band-Erkennung auf Bild dieser Breite (Tempo)
BAND_RED_MIN = 0.3         # Mindestanteil der Achse A->B, der im Korridor rote Pixel hat
BAND_MIN_L0_WORK = 20.0    # Mindest-L0 im Arbeitsbild (px), sonst keine Kalibrierung
CORR_MARGIN_WORK = 5.0     # Korridor reicht um diesen Rand über A und B hinaus (px, Arbeitsbild)
CORR_HW_MIN_WORK = 10.0    # Korridor-Halbbreite, untere Grenze (px, Arbeitsbild)
CORR_HW_MAX_WORK = 60.0    # Korridor-Halbbreite, obere Grenze (px, Arbeitsbild)
CORR_HW_DEFAULT = 30.0     # Halbbreite (Arbeitsbild), falls Schulterbreite fehlt
CORR_FRAC = 0.25           # Halbbreite = CORR_FRAC * Schulterbreite (Bildpixel)
# Nur für die Referenzspalte band_len_pca_px (alte Methode, ohne Landmarks)
BAND_MIN_FRAC = 0.0003
BAND_MERGE_FRAC = 0.15
LEN_PCT = (1.0, 99.0)


class BandTracker:
    """
    Rotes Band zwischen den Griffpunkten beider Hände (Reverse Butterfly).

    Länge L = Abstand der Griffpunkte A->B im Bild (2D, Pixel). Die Roterkennung
    validiert nur: Entlang der festen Achse A->B wird ein Korridor (Halbbreite
    CORR_FRAC * Schulterbreite) gelegt. Der Anteil der Achsenlänge mit roten Pixeln
    im Korridor ist band_red_frac. present = 1 nur bei band_red_frac >= BAND_RED_MIN.
    Kein PCA-Fallback: Fehlt das Band im Korridor, ist present = 0 und stretch leer.
    Die alte PCA-Länge (alle roten Pixel im Bild) steht nur als Referenz in
    band_len_pca_px.

    Dehnung % = (L / L0 - 1) * 100. L0 = Median von L in den ersten calib_frames
    Frames (Startposition der Übung, nicht das entspannte Band) oder per Taste c.
    L0 muss mindestens BAND_MIN_L0_WORK (Arbeitsbild) betragen.
    """

    def __init__(self, calib_frames=30, s_min=120, v_min=70):
        self.calib_frames = calib_frames
        self.s_min, self.v_min = s_min, v_min
        self.n = 0
        self.buf = []
        self.l0 = NAN
        self.scale = 1.0
        self.ema = Ema(SMOOTH_ALPHA, 1)
        self.k5 = np.ones((5, 5), np.uint8)
        self.k9 = np.ones((9, 9), np.uint8)

    @staticmethod
    def none():
        """Detektionsergebnis 'kein Band'."""
        return dict(present=0, cx=NAN, cy=NAN, area=NAN, len_px=NAN, width_px=NAN,
                    angle=NAN, p0=None, p1=None, box=None, pca_len_px=NAN, red_frac=NAN)

    def _mask(self, small):
        hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV)
        m = (cv2.inRange(hsv, (0, self.s_min, self.v_min), (10, 255, 255))
             | cv2.inRange(hsv, (170, self.s_min, self.v_min), (180, 255, 255)))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.k5)
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.k9)

    @staticmethod
    def _pca_len(m, sw, sh):
        """Alte Methode (Referenz): Ausdehnung aller roten Regionen entlang der PCA-Hauptachse, Arbeitsbild-px."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            return NAN
        areas = stats[1:, cv2.CC_STAT_AREA]
        amax = areas.max()
        if amax < BAND_MIN_FRAC * sw * sh:
            return NAN
        keep = 1 + np.where(areas >= BAND_MERGE_FRAC * amax)[0]
        ys, xs = np.nonzero(np.isin(labels, keep))
        pts = np.column_stack([xs, ys]).astype(np.float64)
        d = pts - pts.mean(axis=0)
        evals, evecs = np.linalg.eigh(np.cov(d.T))
        t = d @ evecs[:, 1]
        t0, t1 = np.percentile(t, LEN_PCT)
        return float(t1 - t0)

    def detect(self, frame, pa, pb, half_w):
        """pa, pb: Griffpunkte in Bildpixeln (oder None). half_w: Korridor-Halbbreite in Bildpixeln."""
        res = self.none()
        h, w = frame.shape[:2]
        s = min(1.0, BAND_WORK_W / w)
        self.scale = s
        inv = 1.0 / s
        small = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame
        sh, sw = small.shape[:2]
        m = self._mask(small)
        pca = self._pca_len(m, sw, sh)
        if np.isfinite(pca):
            res["pca_len_px"] = pca * inv
        if pa is None or pb is None:
            return res
        a = np.asarray(pa, float) * s
        b = np.asarray(pb, float) * s
        ab = b - a
        L = float(np.linalg.norm(ab))
        res["p0"], res["p1"] = tuple(a * inv), tuple(b * inv)
        res["angle"] = float(np.degrees(np.arctan2(ab[1], ab[0])) % 180.0)
        if not np.isfinite(L) or L < 2.0:
            return res
        u = ab / L
        nrm = np.array([-u[1], u[0]])
        hw_in = half_w * s if np.isfinite(half_w) else CORR_HW_DEFAULT
        hw = float(np.clip(hw_in, CORR_HW_MIN_WORK, CORR_HW_MAX_WORK))
        mg = CORR_MARGIN_WORK
        corners = np.array([a - u * mg + nrm * hw, a - u * mg - nrm * hw,
                            b + u * mg - nrm * hw, b + u * mg + nrm * hw])
        res["box"] = (corners * inv).astype(np.int32)
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            res["red_frac"] = 0.0
            return res
        xy = np.column_stack([xs, ys]).astype(np.float64)
        rel = xy - a
        t, q = rel @ u, rel @ nrm
        sel = (t >= -mg) & (t <= L + mg) & (np.abs(q) <= hw)
        if not sel.any():
            res["red_frac"] = 0.0
            return res
        ts = np.clip(t[sel], 0.0, L - 1e-6)
        nb = max(int(np.ceil(L)), 1)
        frac = min(np.unique(ts.astype(int)).size / nb, 1.0)
        res["red_frac"] = float(frac)
        if frac < BAND_RED_MIN:
            return res
        mean = xy[sel].mean(axis=0)
        area = float(sel.sum())
        res.update(present=1, cx=mean[0] * inv, cy=mean[1] * inv, area=area * inv * inv,
                   len_px=L * inv, width_px=area / L * inv)
        return res

    def stretch(self, det, calibrate_now=False):
        """Rückgabe: (len_smooth_px, stretch_raw_pct, stretch_pct)"""
        self.n += 1
        min_l0 = BAND_MIN_L0_WORK / self.scale
        L = det["len_px"] if det["present"] else NAN
        Ls = self.ema([L])[0]
        if np.isfinite(Ls) and self.n <= self.calib_frames:
            self.buf.append(Ls)
        if self.calib_frames and self.n == self.calib_frames and len(self.buf) >= 5:
            med = float(np.median(self.buf))
            if med >= min_l0:
                self.l0 = med
        if calibrate_now and np.isfinite(Ls) and Ls >= min_l0:
            self.l0 = float(Ls)
        ok = np.isfinite(self.l0) and self.l0 > 0
        raw = (L / self.l0 - 1.0) * 100.0 if ok and np.isfinite(L) else NAN
        sm = (Ls / self.l0 - 1.0) * 100.0 if ok and np.isfinite(Ls) else NAN
        return Ls, raw, sm


def corridor_half_width(pose, w, h, frac):
    """Halbbreite des Band-Korridors in Bildpixeln: frac * Schulterbreite (aus Pose.img)."""
    px = pose.img * np.array([w, h])
    sw = float(np.linalg.norm(px[JI["l_shoulder"]] - px[JI["r_shoulder"]]))
    return frac * sw if np.isfinite(sw) else NAN


# ---------------------------------------------------------------- Ausgabe
def csv_header():
    h = ["frame", "timestamp_ms", "pose_detected", "exercise", "subject_id",
         "rep_id", "model", "infer_ms", "fresh", "conf"]
    for j in JOINTS:
        h += [f"{j}_{s}" for s in ("wx", "wy", "wz", "wx_raw", "wy_raw", "wz_raw", "ix", "iy", "vis")]
    h += [f"angle_{a[0]}_raw" for a in ANGLES]       # aus Rohkoordinaten, ungeglättet
    h += [f"angle_{a[0]}_cor_raw" for a in ANGLES]   # aus korrigierten Koordinaten, ungeglättet
    h += [f"angle_{a[0]}" for a in ANGLES]           # aus korrigierten Koordinaten, EMA
    h += ["grip_l_ix", "grip_l_iy", "grip_r_ix", "grip_r_iy"]
    h += ["band_present", "band_cx", "band_cy", "band_area", "band_len_px",
          "band_len_smooth_px", "band_width_px", "band_angle_deg", "band_ref_len_px",
          "stretch_raw_pct", "stretch_pct", "band_len_pca_px", "band_red_frac"]
    h += ["scale_px_per_m", "band_len_m", "stretch_abs_pct"]   # Bildskala s, Bandlänge in m, absolute Dehnung
    h += [f"angle_{a[0]}_v6_raw" for a in ANGLES]   # Referenz: v6-Tiefenkorrektur (Modus depth), ungeglättet
    return h


def seg_header():
    h = ["frame", "timestamp_ms"]
    h += [f"dev_pct_{a}-{b}" for a, b in CHECK_KEYS]   # (Modell roh / gemessen - 1) * 100
    h += [f"depth_on_{a}-{b}" for a, b in SEG_KEYS]    # 1 = Tiefenkorrektur in diesem Frame aktiv
    return h


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{v:.5f}"
    return str(v)


class Recorder:
    def __init__(self, outdir, stem):
        outdir.mkdir(parents=True, exist_ok=True)
        self.csv_path = outdir / f"{stem}.csv"
        self.seg_path = outdir / f"{stem}_segdev.csv"
        self.meta_path = outdir / f"{stem}_meta.json"
        self.f = open(self.csv_path, "w", newline="")
        self.sf = open(self.seg_path, "w", newline="")
        self.header = csv_header()
        self.sheader = seg_header()
        self.f.write(",".join(self.header) + "\n")
        self.sf.write(",".join(self.sheader) + "\n")
        self.n = 0

    def write(self, row):
        assert len(row) == len(self.header), (len(row), len(self.header))
        self.f.write(",".join(_fmt(v) for v in row) + "\n")
        self.n += 1

    def write_seg(self, row):
        assert len(row) == len(self.sheader), (len(row), len(self.sheader))
        self.sf.write(",".join(_fmt(v) for v in row) + "\n")

    def close(self, meta):
        self.f.close()
        self.sf.close()
        meta["frames_written"] = self.n
        meta["csv_columns"] = len(self.header)
        meta["segdev_csv"] = self.seg_path.name
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)


# ---------------------------------------------------------------- HUD
GREEN, GREY, YELLOW, CYAN, MAGENTA = (0, 220, 0), (140, 140, 140), (0, 255, 255), (255, 255, 0), (255, 0, 255)
ORANGE = (0, 165, 255)


def _txt(img, s, org, scale=0.6, color=(255, 255, 255), th=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def draw_hud(frame, model_name, pose, ang_sm, band, len_s, l0, pct, info):
    h, w = frame.shape[:2]
    if pose.detected:
        px = pose.img * np.array([w, h])
        ok = pose.vis >= VIS_MIN
        for a, b in BONES:
            ia, ib = JI[a], JI[b]
            if np.all(np.isfinite(px[[ia, ib]])):
                cv2.line(frame, tuple(px[ia].astype(int)), tuple(px[ib].astype(int)),
                         GREEN if ok[ia] and ok[ib] else GREY, 2, cv2.LINE_AA)
        for i in range(len(JOINTS)):
            if np.all(np.isfinite(px[i])):
                cv2.circle(frame, tuple(px[i].astype(int)), 4, GREEN if ok[i] else GREY, -1)
        for (name, a, b, c), v in zip(ANGLES, ang_sm):
            ib = JI[b]
            if np.isfinite(v) and np.all(np.isfinite(px[ib])):
                _txt(frame, f"{v:.0f}", (int(px[ib][0]) + 6, int(px[ib][1]) - 6), 0.5, YELLOW)
        for k in range(2):                                   # Griffpunkte
            if pose.grip_ok[k] and np.all(np.isfinite(pose.grip[k])):
                g = (pose.grip[k] * np.array([w, h])).astype(int)
                cv2.circle(frame, tuple(g), 9, ORANGE, 2, cv2.LINE_AA)

    if band["box"] is not None:                              # Korridor A->B
        cv2.polylines(frame, [band["box"]], True, CYAN if band["present"] else GREY, 2, cv2.LINE_AA)
    if band["p0"] is not None:
        p0 = tuple(int(round(v)) for v in band["p0"])
        p1 = tuple(int(round(v)) for v in band["p1"])
        col = MAGENTA if band["present"] else GREY
        cv2.line(frame, p0, p1, col, 2, cv2.LINE_AA)

    if np.isfinite(pct):
        _txt(frame, f"Dehnung {pct:+.1f} %", (12, 44), 1.1, YELLOW, 2)
        _txt(frame, f"L {len_s:.0f} px  L0 {l0:.0f} px", (12, 68), 0.5)
    if np.isfinite(info["abs_pct"]):
        _txt(frame, f"absolut {info['abs_pct']:+.1f} %  ({info['len_m']:.2f} m)", (12, 112), 0.5, YELLOW)
    elif np.isfinite(info["len_m"]):
        _txt(frame, f"Band {info['len_m']:.2f} m", (12, 112), 0.5)
    elif band["present"]:
        _txt(frame, "Dehnung: kalibrieren (c)", (12, 44), 0.8, YELLOW, 2)
    else:
        _txt(frame, "Dehnung: kein Band", (12, 44), 0.8, GREY, 2)
    if np.isfinite(band["red_frac"]):
        _txt(frame, f"rot im Korridor {band['red_frac']:.2f} (min {BAND_RED_MIN:.2f})", (12, 90), 0.5)
    _txt(frame, f"{model_name}  frame {info['frame']}  rep {info['rep']}  "
                f"{info['infer_ms']:.0f} ms  {info['fps']:.0f} fps", (12, h - 14), 0.55)
    _txt(frame, "Space=rep  c=kalibrieren  q=Ende", (12, h - 36), 0.5)
    s_txt = f"{info['s']:.0f} px/m" if np.isfinite(info["s"]) else "-"
    _txt(frame, f"Segment {info['mode']}  s {s_txt}", (12, h - 58), 0.5)


# ---------------------------------------------------------------- Interaktive Abfrage
CFG_PATH = Path.home() / ".pose_pipeline.json"   # merkt sich letzte Eingaben (ausserhalb des Repos)


def load_cfg():
    try:
        return json.loads(CFG_PATH.read_text())
    except Exception:
        return {}


def save_cfg(**kw):
    try:
        cfg = load_cfg()
        cfg.update(kw)
        CFG_PATH.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass


def find_repo_root():
    """Nächster übergeordneter Ordner mit .git oder data/ (sonst aktueller Ordner)."""
    for p in Path(__file__).resolve().parents:
        if (p / ".git").exists() or (p / "data").is_dir():
            return p
    return Path.cwd()


def default_out_dir():
    root = find_repo_root()
    return root / "data" / "output" if (root / "data").is_dir() else root / "output"


def clean_path(s):
    """Bereinigt Pfade aus Drag-and-Drop ('...' oder Backslash vor Leerzeichen)."""
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        s = s[1:-1]
    return str(Path(s.replace("\\ ", " ")).expanduser())


def ask(prompt, default=""):
    d = f" [{default}]" if default != "" else ""
    try:
        v = input(f"{prompt}{d}: ").strip()
    except EOFError:
        v = ""
    return v if v else str(default)


def ask_yes_no(prompt, default):
    hint = "J/n" if default else "j/N"
    while True:
        v = ask(f"{prompt} ({hint})", "").lower()
        if v == "":
            return default
        if v in ("j", "ja", "y", "yes"):
            return True
        if v in ("n", "nein", "no"):
            return False
        print("Bitte j oder n eingeben.")


def ask_int(prompt, default, minimum=0):
    while True:
        v = ask(prompt, default)
        if v.lstrip("-").isdigit() and int(v) >= minimum:
            return int(v)
        print("Bitte eine ganze Zahl eingeben.")


def ask_float(prompt, default):
    while True:
        v = ask(prompt, default).replace(",", ".")
        try:
            return float(v)
        except ValueError:
            print("Bitte eine Zahl eingeben.")


def interactive_common(args):
    cfg = load_cfg()
    print(f"=== Pose-Pipeline: {MODEL_NAME} ===")
    src = ask("Quelle: 1 = Video-Datei, 2 = Live-Kamera", cfg.get("source", "1"))
    while src not in ("1", "2"):
        src = ask("Bitte 1 oder 2 eingeben", "1")
    if src == "1":
        while True:
            p = clean_path(ask("Pfad zur Videodatei (Datei ins Terminal ziehen geht)",
                               cfg.get("video", "")))
            if Path(p).is_file():
                break
            print(f"Datei nicht gefunden: {p}")
        args.video, args.camera = p, None
        save_cfg(source="1", video=p)
    else:
        args.camera = ask_int("Kamera-Index (0 = Standard, 1 = oft iPhone/extern)",
                              cfg.get("camera", 0))
        args.video = None
        save_cfg(source="2", camera=args.camera)
    args.exercise = ask("Übung (Label für die CSV, z.B. reversebutterfly)", cfg.get("exercise", ""))
    args.subject = ask("Probanden-ID (Enter = leer)", cfg.get("subject", ""))
    band = ask_yes_no("Theraband im Video", cfg.get("band", False))
    args.no_band = not band
    if band:
        cm = ask_float("Freie Bandlänge zwischen den Griffen, Band gerade straff und ungedehnt, in cm (0 = aus)",
                       cfg.get("band_rest_cm", 0))
        args.band_rest_m = cm / 100.0
        save_cfg(band_rest_cm=cm)
    if args.camera is None:
        args.no_show = not ask_yes_no("Fenster mit HUD anzeigen (Beenden mit q)", True)
        args.max_frames = ask_int("Nur die ersten N Frames verarbeiten (0 = alle)", 0)
    else:
        args.no_show, args.max_frames = False, 0
        print("Live-Kamera: Beenden mit q im Fenster.")
    if band:
        print("Band: erste 30 Frames in Startposition zeigen (L0) oder im Fenster c drücken.")
    save_cfg(exercise=args.exercise, subject=args.subject, band=band)


# ---------------------------------------------------------------- Hauptschleife
def add_common_args(p):
    s = p.add_mutually_exclusive_group(required=False)
    s.add_argument("--video", help="Pfad zu .mov/.mp4")
    s.add_argument("--camera", type=int, help="Kamera-Index (0, 1, ...)")
    p.add_argument("--out", default=None, help="Ausgabeordner (Standard: <Repo>/data/output)")
    p.add_argument("--exercise", default="", help="Label für CSV-Spalte exercise")
    p.add_argument("--subject", default="", help="Label für CSV-Spalte subject_id")
    p.add_argument("--calib-frames", type=int, default=30,
                   help="Erste N Frames (Startposition) als Referenzlänge L0, 0 = nur Taste c")
    p.add_argument("--red-s-min", type=int, default=120, help="HSV Sättigung min für Rot")
    p.add_argument("--red-v-min", type=int, default=70, help="HSV Helligkeit min für Rot")
    p.add_argument("--band-anchor", default="hands", choices=["hands"],
                   help="Band-Endpunkte: hands = Griffpunkt links zu Griffpunkt rechts")
    p.add_argument("--corridor-frac", type=float, default=CORR_FRAC,
                   help="Korridor-Halbbreite als Anteil der Schulterbreite (Bildpixel)")
    p.add_argument("--depth-segs", default="arms", choices=["arms", "all"],
                   help="Segmente mit Tiefenkorrektur: arms = Oberarm, Unterarm (Standard), all = alle (v6)")
    p.add_argument("--segment-mode", default="taylor", choices=["taylor", "depth"],
                   help="taylor = Tiefenkorrektur mit Bildpixeln/Skala s (Standard), depth = v6 (Modell-xy)")
    p.add_argument("--band-rest-m", type=float, default=0.0,
                   help="Freie Bandlänge zwischen den Griffen in m (straff, ungedehnt). >0 aktiviert stretch_abs_pct")
    p.add_argument("--no-band", action="store_true",
                   help="Band-Erkennung aus (Video ohne Band): Bandspalten bleiben leer")
    p.add_argument("--no-show", action="store_true", help="Kein Fenster (Batch)")
    p.add_argument("--max-frames", type=int, default=0)


def run_pipeline(model_name, infer, args, meta_extra=None):
    is_video = args.video is not None
    cap = cv2.VideoCapture(args.video if is_video else args.camera)
    if not cap.isOpened():
        sys.exit("Quelle nicht öffenbar.")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_name = Path(args.video).stem if is_video else f"cam{args.camera}"
    t_start = datetime.now()
    stem = f"{model_name}_{src_name}_{t_start:%Y%m%d_%H%M%S}".replace(" ", "_")
    outdir = Path(args.out) if args.out else default_out_dir()
    rec = Recorder(outdir, stem)
    band = BandTracker(args.calib_frames, args.red_s_min, args.red_v_min)
    ema = Ema(SMOOTH_ALPHA, len(ANGLES))
    corrector = DepthCorrector(args.segment_mode, ARM_SEGS if args.depth_segs == "arms" else SEG_KEYS)
    ref_corrector = DepthCorrector("depth", SEG_KEYS)           # v6-Referenz, nur für die Spalten angle_*_v6_raw
    check_samples = {k: [] for k in CHECK_PAIRS}   # rohe Segmentabstände für den Vergleichsreport

    rep, frame_idx, last_ts, calib = 0, 0, -1, False
    fps_s, infer_sum = 30.0, 0.0
    t0 = time.perf_counter()
    try:
        while True:
            t_loop = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            fh, fw = frame.shape[:2]
            ts = int(round(frame_idx * 1000 / src_fps)) if is_video else int((t_loop - t0) * 1000)
            ts = max(ts, last_ts + 1)
            last_ts = ts

            t1 = time.perf_counter()
            pose = infer(frame, ts)
            infer_ms = (time.perf_counter() - t1) * 1000
            infer_sum += infer_ms

            if pose.detected:
                w_raw = pose.w
                w_cor = corrector(w_raw, pose.vis, pose.img * np.array([fw, fh]))   # Kopie, Rohwerte bleiben erhalten
                w_v6 = ref_corrector(w_raw, pose.vis)
                dists = segment_distances(w_raw, pose.vis)
                collect_segment_samples(check_samples, dists)
                dev = [(d / CHECK_PAIRS[k] - 1.0) * 100.0 for k, d in zip(CHECK_KEYS, dists)]
                depth_on = [corrector.last_on[k] for k in SEG_KEYS]
            else:
                corrector.reset()
                ref_corrector.reset()
                w_raw = w_cor = w_v6 = pose.w
                dev = [NAN] * len(CHECK_KEYS)
                depth_on = [NAN] * len(SEG_KEYS)
            raw = compute_angles(pose)                         # aus rohen Koordinaten, ungeglättet
            cor_raw = compute_angles(pose, w_cor)              # aus korrigierten Koordinaten, ungeglättet
            sm = ema(cor_raw)                                  # aus korrigierten Koordinaten, EMA
            v6_raw = compute_angles(pose, w_v6)                # Referenz: v6-Korrektur

            if args.no_band or not pose.detected or not pose.grip_ok.all():
                pa = pb = None
            else:
                pa, pb = pose.grip[0] * np.array([fw, fh]), pose.grip[1] * np.array([fw, fh])
            half_w = corridor_half_width(pose, fw, fh, args.corridor_frac) if pose.detected else NAN
            b = BandTracker.none() if args.no_band else band.detect(frame, pa, pb, half_w)
            len_s, pct_raw, pct = band.stretch(b, calib)
            calib = False
            s_px = corrector.s
            len_m = len_s / s_px if s_px > 0 else NAN          # NaN > 0 ist False
            abs_pct = (len_m / args.band_rest_m - 1.0) * 100.0 if args.band_rest_m > 0 else NAN

            row = [frame_idx, ts, int(pose.detected), args.exercise, args.subject,
                   rep, model_name, infer_ms, pose.fresh, pose.conf]
            for i in range(len(JOINTS)):
                row += [*w_cor[i], *w_raw[i], *pose.img[i], pose.vis[i]]
            row += raw + cor_raw + sm
            row += [*pose.grip[0], *pose.grip[1]]
            row += [b["present"], b["cx"], b["cy"], b["area"], b["len_px"], len_s,
                    b["width_px"], b["angle"], band.l0, pct_raw, pct, b["pca_len_px"], b["red_frac"]]
            row += [s_px, len_m, abs_pct] + v6_raw
            rec.write(row)
            rec.write_seg([frame_idx, ts] + dev + depth_on)

            if not args.no_show:
                draw_hud(frame, model_name, pose, sm, b, len_s, band.l0, pct,
                         dict(frame=frame_idx, rep=rep, infer_ms=infer_ms, fps=fps_s,
                              mode=args.segment_mode, s=s_px, len_m=len_m, abs_pct=abs_pct))
                cv2.imshow(model_name, frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    break
                if k == ord(" "):
                    rep += 1
                if k == ord("c"):
                    calib = True
            frame_idx += 1
            fps_s = 0.9 * fps_s + 0.1 / max(time.perf_counter() - t_loop, 1e-6)
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        report = segment_check_report(check_samples, CHECK_PAIRS)
        depth_report = corrector.report()
        scale_report = corrector.scale_report()
        meta = dict(
            model=model_name, source=args.video or f"camera {args.camera}",
            source_fps=src_fps, width=W, height=H, started=t_start.isoformat(),
            finished=datetime.now().isoformat(), mean_infer_ms=infer_sum / max(frame_idx, 1),
            vis_min=VIS_MIN, smooth_alpha=SMOOTH_ALPHA, calib_frames=args.calib_frames,
            no_band=args.no_band, band_ref_len_px=band.l0, red_s_min=args.red_s_min, red_v_min=args.red_v_min,
            band_anchor=args.band_anchor, corridor_frac=args.corridor_frac,
            band_red_min=BAND_RED_MIN, band_min_l0_work=BAND_MIN_L0_WORK,
            corridor_hw_work_limits=[CORR_HW_MIN_WORK, CORR_HW_MAX_WORK],
            corridor_margin_work=CORR_MARGIN_WORK, band_work_width=BAND_WORK_W,
            band_len_percentiles_ref_pca=LEN_PCT,
            depth_correction=dict(mode=args.segment_mode, dev_on=DEV_ON, dev_off=DEV_OFF, dz_eps=DZ_EPS,
                                  per_segment=depth_report, scale=scale_report),
            depth_segs=args.depth_segs, band_rest_m=args.band_rest_m,
            joints=JOINTS, angles=[a[0] for a in ANGLES],
            segment_lengths_m={f"{p}-{c}": v for (p, c), v in SEGMENT_LENGTHS.items()},
            body_widths_m={f"{p}-{c}": v for (p, c), v in BODY_WIDTHS.items()},
            segment_check=report,
            python=platform.python_version(), opencv=cv2.__version__,
        )
        meta.update(meta_extra or {})
        rec.close(meta)
        print_segment_report(report)
        print_depth_report(depth_report)
        print_scale_report(scale_report)
        print(f"CSV   : {rec.csv_path}\nSegDev: {rec.seg_path}\nMeta  : {rec.meta_path}")
# =============================================================================
# COMMON END
# =============================================================================

# =============================================================================
# MODEL START (nur dieser Block unterscheidet sich zwischen den Dateien)
# =============================================================================
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_NAME = "mediapipe"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")

# Gemeinsames Gelenk -> MediaPipe-Landmark-Index
MP_IDX = {
    "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16, "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26, "l_ankle": 27, "r_ankle": 28,
    "l_foot": 31, "r_foot": 32,  # foot_index
}
IDX = [MP_IDX[j] for j in JOINTS]

# Griffpunkt pro Hand = Mittelwert aus Pinky, Zeigefinger, Daumen (MediaPipe-Landmarks
# 17/19/21 links, 18/20/22 rechts), nur Punkte mit visibility >= VIS_MIN.
GRIP_IDX = {"l": [17, 19, 21], "r": [18, 20, 22]}


def add_model_args(ap):
    ap.add_argument("--model", default=None,
                    help="Pfad zur .task-Datei (Standard: <Repo>/Models/models/pose_landmarker_heavy.task)")


def interactive_model(args):
    pass                                   # MediaPipe: keine Zusatzfragen


def load_model(args):
    """Rückgabe: (infer(frame_bgr, ts_ms) -> Pose, meta_dict, close())"""
    name = "pose_landmarker_heavy.task"
    if args.model:
        path = Path(args.model).expanduser()
    else:
        root = find_repo_root()
        cands = [root / "Models" / "models" / name, root / "models" / name, Path.cwd() / "models" / name]
        path = next((c for c in cands if c.exists()), cands[0])
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print("Lade Modell ...")
        urllib.request.urlretrieve(MODEL_URL, path)
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(path), delegate=mp_python.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    lm = vision.PoseLandmarker.create_from_options(opts)

    def infer(frame_bgr, ts_ms):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
        if not res.pose_world_landmarks:
            return empty_pose()
        wl, il = res.pose_world_landmarks[0], res.pose_landmarks[0]
        n = len(JOINTS)
        w, img, vis = np.zeros((n, 3)), np.zeros((n, 2)), np.zeros(n)
        for k, i in enumerate(IDX):
            w[k] = (wl[i].x, wl[i].y, wl[i].z)
            img[k] = (il[i].x, il[i].y)
            vis[k] = il[i].visibility
        grip, grip_ok = np.full((2, 2), NAN), np.zeros(2, dtype=bool)
        for k, side in enumerate(("l", "r")):
            pts = [(il[i].x, il[i].y) for i in GRIP_IDX[side] if il[i].visibility >= VIS_MIN]
            if pts:
                grip[k] = np.mean(pts, axis=0)
                grip_ok[k] = True
        return Pose(True, w, img, vis, grip=grip, grip_ok=grip_ok,
                    fresh=1, conf=float(np.mean(vis)))

    meta = dict(model_file=path.name, delegate="CPU", mediapipe=mp.__version__, mp_idx=MP_IDX,
                grip_idx=GRIP_IDX,
                fresh_note="fresh = 1 bei erkannter Pose; conf = mittlere Visibility der 14 Gelenke")
    return infer, meta, lm.close
# =============================================================================
# MODEL END
# =============================================================================


# =============================================================================
# ENTRY START (in beiden Dateien byte-identisch)
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    add_model_args(ap)
    if len(sys.argv) == 1:                 # Start ohne Argumente (Run-Button): Abfrage
        args = ap.parse_args([])
        interactive_common(args)
        interactive_model(args)
        print(f"Ausgabe nach: {Path(args.out) if args.out else default_out_dir()}")
    else:
        args = ap.parse_args()
        if args.video is None and args.camera is None:
            ap.error("--video oder --camera angeben (oder ohne Argumente starten für die Abfrage)")
    infer, meta, close = load_model(args)
    try:
        run_pipeline(MODEL_NAME, infer, args, meta)
    finally:
        close()


if __name__ == "__main__":
    main()
# =============================================================================
# ENTRY END
# =============================================================================
