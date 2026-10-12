"""
MediaPipe Pose + Fitnessband Pipeline (eine Datei)

Installation:
    pip install mediapipe opencv-python numpy

Start:
    python pose_band_pipeline.py                      # fragt Quelle im Terminal ab
    python pose_band_pipeline.py --cam 0              # Webcam live
    python pose_band_pipeline.py --video clip.mp4     # Video aus Speicher
    python pose_band_pipeline.py --video clip.mp4 --bands red,blue --out out.csv
    python pose_band_pipeline.py --cam 0 --exercise squat --subject P01

Tasten im Fenster:
    q oder ESC = beenden
    LEERTASTE  = naechste Wiederholung beginnt (rep_id + 1; 0 = vor der ersten Markierung)

Ausgabe: <out>.csv und <out>_meta.json (Lauf-Metadaten).
Standardpfade sind relativ zum Arbeitsverzeichnis: aus dem Repo-Root starten.
    Modell: models/pose_landmarker_heavy.task   CSV: data/output/pose_band_data.csv

Das Modell (pose_landmarker_heavy.task) wird beim ersten Start automatisch geladen.
"""

import argparse
import csv
import json
import os
import time
import urllib.request
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ----------------------------------------------------------------------------
# KONFIGURATION
# ----------------------------------------------------------------------------
MODEL_PATH = "models/pose_landmarker_heavy.task"  # relativ zum Arbeitsverzeichnis (Repo-Root)
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")

VIS_THRESHOLD = 0.5  # Landmarks darunter: im HUD grau, Winkel mit so einem Landmark ungueltig
SMOOTH_ALPHA = 0.4   # EMA-Glaettung der Winkel (0..1; 1.0 = aus; kleiner = staerker). 0.4 = Wert aus v1

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer", "left_ear", "right_ear",
    "mouth_left", "mouth_right", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_pinky", "right_pinky", "left_index", "right_index",
    "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]
IDX = {n: i for i, n in enumerate(LANDMARK_NAMES)}

# Gelenkwinkel: Name -> (Punkt A, Gelenk B, Punkt C). Winkel B aufgespannt durch A und C wird berechnet
ANGLES = {
    "l_elbow":    ("left_shoulder", "left_elbow", "left_wrist"),
    "r_elbow":    ("right_shoulder", "right_elbow", "right_wrist"),
    "l_shoulder": ("left_hip", "left_shoulder", "left_elbow"),
    "r_shoulder": ("right_hip", "right_shoulder", "right_elbow"),
    "l_hip":      ("left_shoulder", "left_hip", "left_knee"),
    "r_hip":      ("right_shoulder", "right_hip", "right_knee"),
    "l_knee":     ("left_hip", "left_knee", "left_ankle"),
    "r_knee":     ("right_hip", "right_knee", "right_ankle"),
    "l_ankle":    ("left_knee", "left_ankle", "left_foot_index"),
    "r_ankle":    ("right_knee", "right_ankle", "right_foot_index"),
    "l_wrist":    ("left_elbow", "left_wrist", "left_index"),
    "r_wrist":    ("right_elbow", "right_wrist", "right_index"),
}

# Skelett-Verbindungen fuers HUD
CONNECTIONS = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("left_heel", "left_foot_index"),
    ("left_ankle", "left_foot_index"), ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"), ("right_ankle", "right_foot_index"),
    ("left_wrist", "left_index"), ("right_wrist", "right_index"),
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
]

# 2.2 Reale Segmentlaengen von Fabrice in Meter (gemittelt)

SEGMENT_LENGTHS = {
    ("left_hip", "left_shoulder"): 0.480,    # Rumpf: Hueftgelenk bis Schultergelenk
    ("right_hip", "right_shoulder"): 0.480,
    ("left_shoulder", "left_elbow"): 0.315,  # Oberarme
    ("right_shoulder", "right_elbow"): 0.315,
    ("left_elbow", "left_wrist"): 0.270,     # Unterarme
    ("right_elbow", "right_wrist"): 0.270,
    ("left_hip", "left_knee"): 0.475,        # Oberschenkel
    ("right_hip", "right_knee"): 0.475,
    ("left_knee", "left_ankle"): 0.415,      # Unterschenkel
    ("right_knee", "right_ankle"): 0.415,
    ("left_ankle", "left_foot_index"): 0.200,   # Fuss: Knoechel bis Fussspitze
    ("right_ankle", "right_foot_index"): 0.200,
}

BODY_WIDTHS = {
    ("left_hip", "right_hip"): 0.310,            # Hueftbreite
    ("left_shoulder", "right_shoulder"): 0.400,  # Schulterbreite
}

# Kinematische Kette: Gelenk -> direkte Nachfolger für Korrektur mit Segmentlängen
CHILDREN = {
    "left_shoulder": ["left_elbow"], "right_shoulder": ["right_elbow"],
    "left_elbow": ["left_wrist"], "right_elbow": ["right_wrist"],
    "left_wrist": ["left_pinky", "left_index", "left_thumb"],
    "right_wrist": ["right_pinky", "right_index", "right_thumb"],
    "left_hip": ["left_knee"], "right_hip": ["right_knee"],
    "left_knee": ["left_ankle"], "right_knee": ["right_ankle"],
    "left_ankle": ["left_heel", "left_foot_index"],
    "right_ankle": ["right_heel", "right_foot_index"],
}


def descendants(name):
    """Alle Nachfolger eines Gelenks (rekursiv) in der Kette CHILDREN."""
    out = []
    for ch in CHILDREN.get(name, []):
        out.append(ch)
        out += descendants(ch)
    return out

def segment_check_report(samples, measured):
    """
    Vergleicht Median der rohen MediaPipe-Abstaende mit den gemessenen Laengen.
    samples: {(a, b): [Abstand_m, ...]}, measured: {(a, b): Laenge_m}.
    Rueckgabe: {"a-b": {"measured_m", "mediapipe_median_m", "deviation_pct", "n"}}
    deviation_pct = (MediaPipe - gemessen) / gemessen * 100. Ohne Samples: None-Werte.
    """
    out = {}
    for key, length in measured.items():
        vals = samples.get(key, [])
        if vals:
            med = float(np.median(vals))
            out[f"{key[0]}-{key[1]}"] = {
                "measured_m": length, "mediapipe_median_m": round(med, 4),
                "deviation_pct": round((med - length) / length * 100.0, 1), "n": len(vals)}
        else:
            out[f"{key[0]}-{key[1]}"] = {
                "measured_m": length, "mediapipe_median_m": None, "deviation_pct": None, "n": 0}
    return out

# 3 Fitnessband-Farben: HSV-Bereiche (OpenCV: H 0-179, S 0-255, V 0-255).
# Pro Farbe eine Liste von (untere Grenze, obere Grenze). Rot wrappt ueber H=0.
BAND_COLORS = {
    "red":    [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "green":  [((40, 80, 50), (85, 255, 255))],
    "blue":   [((95, 120, 50), (130, 255, 255))],
    "yellow": [((20, 110, 100), (35, 255, 255))],
    "orange": [((10, 140, 120), (20, 255, 255))],
    "pink":   [((140, 70, 100), (169, 255, 255))],
}
BAND_DRAW_BGR = {
    "red": (0, 0, 255), "green": (0, 200, 0), "blue": (255, 100, 0),
    "yellow": (0, 230, 255), "orange": (0, 140, 255), "pink": (200, 0, 255),
}
BAND_MIN_AREA = 800  # Pixel; kleinere Konturen werden ignoriert


# ----------------------------------------------------------------------------
# HILFSFUNKTIONEN
# ----------------------------------------------------------------------------
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
        print(f"Lade Modell nach {MODEL_PATH} ...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as e:
            raise SystemExit(
                f"Modell-Download fehlgeschlagen ({type(e).__name__}: {e}).\n"
                f"Manuell laden: {MODEL_URL}\nund speichern als: {MODEL_PATH}")


class EmaFilter:
    """
    Exponentielle Glaettung pro Schluessel (Gelenkwinkel).
    Ungueltiger Wert (NaN) setzt den Zustand dieses Schluessels zurueck, damit
    nach einer Luecke nicht ueber veraltete Werte geglaettet wird.
    """
    def __init__(self, alpha):
        self.alpha = alpha
        self.state = {}

    def update(self, key, value):
        if value is None or np.isnan(value):
            self.state.pop(key, None)
            return float("nan")
        prev = self.state.get(key)
        new = value if prev is None else self.alpha * value + (1.0 - self.alpha) * prev
        self.state[key] = new
        return new

    def reset(self):
        self.state.clear()


def visible_angle(world, vis, a, b, c):
    """
    Gelenkwinkel nur, wenn alle drei beteiligten Landmarks sichtbar genug sind
    (Sichtbarkeit >= VIS_THRESHOLD, wie v1), sonst NaN.
    a, b, c sind Landmark-Namen, world (33,3), vis (33,).
    """
    ia, ib, ic = IDX[a], IDX[b], IDX[c]
    if min(vis[ia], vis[ib], vis[ic]) < VIS_THRESHOLD:
        return float("nan")
    return joint_angle(world[ia], world[ib], world[ic])


def fmt(v):
    """CSV-Wert: NaN wird zu leerer Zelle."""
    return "" if (isinstance(v, float) and np.isnan(v)) else v


def joint_angle(a, b, c):
    """Winkel in Grad bei b zwischen Strecken b->a und b->c (3D)."""
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-9:
        return float("nan")
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def apply_segment_lengths(world):
    """
    2.2 Skaliert Knochen auf reale Laengen.
    Eltern-Punkt bleibt, Kind-Punkt wird entlang der aktuellen Knochenrichtung
    auf die reale Laenge gesetzt. Die Verschiebung (neu minus alt) wird auf alle
    Nachfolger addiert, damit die Kette starr bleibt und die Richtungen der
    Folgesegmente erhalten bleiben.
    SEGMENT_LENGTHS muss von proximal nach distal geordnet sein.
    world: (33,3) Array. Gibt korrigierte Kopie zurueck.
    """
    if not SEGMENT_LENGTHS:
        return world
    w = world.copy()
    for (parent, child), length in SEGMENT_LENGTHS.items():
        p, c = IDX[parent], IDX[child]
        d = w[c] - w[p]
        n = np.linalg.norm(d)
        if n > 1e-9:
            delta = (w[p] + d / n * length) - w[c]
            for name in [child] + descendants(child):
                w[IDX[name]] += delta
    return w


def detect_bands(frame_bgr, color_names):
    """
    3 Erkennt je Farbe das groesste Band-Objekt.
    Rueckgabe: dict name -> dict(contour, cx, cy, area, long_px, short_px) oder None.
    """
    blurred = cv2.GaussianBlur(frame_bgr, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    result = {}
    for name in color_names:
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in BAND_COLORS[name]:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= BAND_MIN_AREA]
        if not contours:
            result[name] = None
            continue
        c = max(contours, key=cv2.contourArea)
        m = cv2.moments(c)
        cx = m["m10"] / m["m00"] if m["m00"] else 0.0
        cy = m["m01"] / m["m00"] if m["m00"] else 0.0
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        result[name] = dict(contour=c, cx=cx, cy=cy, area=cv2.contourArea(c),
                            long_px=max(rw, rh), short_px=min(rw, rh))
    return result


def csv_header():
    cols = ["frame", "timestamp_ms", "pose_detected", "exercise", "subject_id", "rep_id"]
    for n in LANDMARK_NAMES:
        cols += [f"{n}_wx", f"{n}_wy", f"{n}_wz",              # World korrigiert (Meter, Hueftmitte = Ursprung)
                 f"{n}_wx_raw", f"{n}_wy_raw", f"{n}_wz_raw",  # World roh (MediaPipe-Ausgabe)
                 f"{n}_ix", f"{n}_iy", f"{n}_vis"]             # Bild normiert 0..1, Sichtbarkeit
    cols += [f"angle_{a}_raw" for a in ANGLES]  # Winkel aus rohen World-Koordinaten
    cols += [f"angle_{a}" for a in ANGLES]      # Winkel aus korrigierten World-Koordinaten
    # Ein Band, feste Spalten. band_color enthaelt die erkannte Farbe (z.B. "red").
    cols += ["band_present", "band_color", "band_cx", "band_cy",
             "band_area", "band_long_px", "band_short_px",
             "band_stretch"]  # stretch: Platzhalter fuer spaeter (3.1)
    return cols


def draw_hud(frame, img_lm, angles, bands, fps):
    h, w = frame.shape[:2]
    # Baender: Umriss
    for name, b in bands.items():
        if b is None:
            continue
        col = BAND_DRAW_BGR.get(name, (255, 255, 255))
        cv2.drawContours(frame, [b["contour"]], -1, col, 2)
        cv2.putText(frame, f"band {name}", (int(b["cx"]), int(b["cy"])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    # Skelett
    if img_lm is not None:
        pts = {n: (int(img_lm[i].x * w), int(img_lm[i].y * h)) for n, i in IDX.items()}
        vis = {n: img_lm[i].visibility for n, i in IDX.items()}
        # Wie v1: alle Punkte und Linien werden gezeichnet. Gruen = sichtbar, grau = unter der
        # Sichtbarkeitsschwelle. Das macht Framing-Probleme live im Bild sichtbar.
        GREEN, GRAY = (0, 255, 0), (120, 120, 120)
        for a, b in CONNECTIONS:
            color = GREEN if (vis[a] >= VIS_THRESHOLD and vis[b] >= VIS_THRESHOLD) else GRAY
            cv2.line(frame, pts[a], pts[b], color, 2)
        for n, p in pts.items():
            color = GREEN if vis[n] >= VIS_THRESHOLD else GRAY
            cv2.circle(frame, p, 7, color, -1)
            cv2.circle(frame, p, 7, (0, 0, 0), 1)  # dunkler Rand fuer Kontrast
        # Winkel am Gelenk (nur gueltige Winkel, Kontur + Text wie v1)
        for aname, (_, bn, _) in ANGLES.items():
            v = angles.get(aname)
            if v is not None and not np.isnan(v):
                pos = (pts[bn][0] + 10, pts[bn][1] - 10)
                label = f"{v:.0f}"
                cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS {fps:.0f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="Pfad zu Videodatei")
    ap.add_argument("--cam", type=int, help="Webcam-Index (z.B. 0)")
    ap.add_argument("--bands", default=",".join(BAND_COLORS),
                    help=f"Kandidatenfarben, kommagetrennt aus: {','.join(BAND_COLORS)}. "
                         "Erkannt wird ein Band, die Farbe landet in der CSV-Spalte band_color.")
    ap.add_argument("--out", default="data/output/pose_band_data.csv")
    ap.add_argument("--exercise", default="unspecified",
                    help="Uebung, z.B. squat oder seitzug (landet in jeder CSV-Zeile)")
    ap.add_argument("--subject", default="unspecified",
                    help="Personen-/Probanden-ID, z.B. P01 (landet in jeder CSV-Zeile)")
    args = ap.parse_args()

    # 1) Quelle waehlen
    if args.video is None and args.cam is None:
        choice = input("Quelle: [1] Video aus Speicher  [2] Webcam live  > ").strip()
        if choice == "1":
            args.video = input("Pfad zur Videodatei > ").strip().strip('"')
        else:
            args.cam = 0

    is_video = args.video is not None
    cap = cv2.VideoCapture(args.video if is_video else args.cam)
    if not cap.isOpened():
        raise SystemExit("Quelle konnte nicht geoeffnet werden.")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    color_names = [c.strip() for c in args.bands.split(",") if c.strip() in BAND_COLORS]

    # 2) MediaPipe Heavy
    ensure_model()
    options = vision.PoseLandmarkerOptions(
        # Expliziter CPU-Delegate: Fix fuer Absturz von mediapipe>=1.0 auf Apple Silicon (wie in v1)
        base_options=mp_python.BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=mp_python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    t0 = time.time()
    frame_idx = 0
    last_ts = -1
    prev_t = time.time()
    fps = 0.0
    rep_id = 0
    ema_raw = EmaFilter(SMOOTH_ALPHA)  # Winkel ohne Segmentkorrektur
    ema_cor = EmaFilter(SMOOTH_ALPHA)  # Winkel mit Segmentkorrektur
    check_pairs = {**SEGMENT_LENGTHS, **BODY_WIDTHS}
    check_samples = {k: [] for k in check_pairs}  # rohe MediaPipe-Abstaende fuer den Laengenvergleich

    # Lauf-Metadaten (Sidecar-Datei), am Ende mit Frame- und Wiederholungszahl ergaenzt
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    meta = {
        "exercise": args.exercise,
        "subject_id": args.subject,
        "source": args.video if is_video else f"webcam:{args.cam}",
        "source_fps": src_fps,
        "resolution": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))],
        "model": MODEL_PATH,
        "band_candidates": color_names,
        "segment_lengths_m": {f"{p}-{c}": v for (p, c), v in SEGMENT_LENGTHS.items()},
        "body_widths_m": {f"{p}-{c}": v for (p, c), v in BODY_WIDTHS.items()},
        "smooth_alpha": SMOOTH_ALPHA,
        "vis_threshold": VIS_THRESHOLD,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header())

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Zeitstempel (ms, streng monoton steigend)
            ts = int(frame_idx * 1000.0 / src_fps) if is_video else int((time.time() - t0) * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts

            # Pose
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(mp_img, ts)

            img_lm = None
            angles = {}
            row = [frame_idx, ts, 0, args.exercise, args.subject, rep_id]
            if res.pose_landmarks and res.pose_world_landmarks:
                img_lm = res.pose_landmarks[0]
                wl = res.pose_world_landmarks[0]
                world_raw = np.array([[p.x, p.y, p.z] for p in wl])
                world = apply_segment_lengths(world_raw)  # 2.2 (Kopie, Rohwerte bleiben erhalten)
                vis = np.array([lm.visibility for lm in img_lm])

                                # Rohe Segmentabstaende sammeln (nur wenn beide Landmarks sichtbar)
                for (a, b) in check_pairs:
                    if vis[IDX[a]] >= VIS_THRESHOLD and vis[IDX[b]] >= VIS_THRESHOLD:
                        check_samples[(a, b)].append(float(np.linalg.norm(world_raw[IDX[a]] - world_raw[IDX[b]])))

                # 2.1 alle Gelenkwinkel, ohne und mit Segmentkorrektur.
                # Gueltig nur bei Sichtbarkeit >= VIS_THRESHOLD aller drei Landmarks, danach EMA-Glaettung.
                angles_raw = {}
                for aname, (a, b, c) in ANGLES.items():
                    angles_raw[aname] = ema_raw.update(aname, visible_angle(world_raw, vis, a, b, c))
                    angles[aname] = ema_cor.update(aname, visible_angle(world, vis, a, b, c))

                row[2] = 1
                for i in range(33):
                    row += [world[i, 0], world[i, 1], world[i, 2],
                            world_raw[i, 0], world_raw[i, 1], world_raw[i, 2],
                            img_lm[i].x, img_lm[i].y, img_lm[i].visibility]
                row += [fmt(angles_raw[a]) for a in ANGLES]
                row += [fmt(angles[a]) for a in ANGLES]
            else:
                ema_raw.reset()
                ema_cor.reset()
                row += [""] * (33 * 9 + 2 * len(ANGLES))

            # 3) Baender
            # Ein Band im Bild: unter den Kandidatenfarben gewinnt die mit der groessten Flaeche.
            bands = detect_bands(frame, color_names)
            found = {n: b for n, b in bands.items() if b is not None}
            if found:
                best = max(found, key=lambda n: found[n]["area"])
                b = found[best]
                row += [1, best, b["cx"], b["cy"], b["area"], b["long_px"], b["short_px"], ""]
                bands = {best: b}
            else:
                row += [0, "", "", "", "", "", "", ""]
                bands = {}

            writer.writerow(row)  # 4 / 4.1

            # 5) HUD
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - prev_t, 1e-6)
            prev_t = now
            draw_hud(frame, img_lm, angles, bands, fps)
            cv2.putText(frame, f"{args.exercise} | rep {rep_id} (LEERTASTE = naechste)", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Pose + Band HUD", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == 32:  # Leertaste
                rep_id += 1

            frame_idx += 1

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    
    # Laengenvergleich: gemessen gegen MediaPipe-Median (Hinweis auf Punktdefinition/Skalierung)
    meta["segment_check"] = segment_check_report(check_samples, check_pairs)
    print("\nSegmentvergleich (gemessen vs. MediaPipe-Median, roh):")
    for name, r in meta["segment_check"].items():
        if r["n"] == 0:
            print(f"  {name:32s} keine gueltigen Frames")
        else:
            print(f"  {name:32s} gemessen {r['measured_m']:.3f} m | MediaPipe {r['mediapipe_median_m']:.3f} m "
                  f"| Abw. {r['deviation_pct']:+.1f} % | n={r['n']}")
            
    meta["frames"] = frame_idx
    meta["reps_marked"] = rep_id
    meta["ended_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2, ensure_ascii=False)
    print(f"Fertig. {frame_idx} Frames, CSV: {args.out}, Meta: {meta_path}")


if __name__ == "__main__":
    main()
