
from __future__ import annotations
import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

# notwendige MediePipe BlazePose-33-Landmark-IDs fest definiert
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

# Offizielle BlazePose-33-Namen in Indexreihenfolge (0..32) für CSV
LANDMARK_NAMES: List[str] = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

ANGLE_COLUMNS: List[str] = ["knee_left", "knee_right", "hip_left", "hip_right", "ankle_left", "ankle_right"]

# Skelett-Verbindungslinien fuer das Overlay (Ober- und Unterkoerper)
POSE_CONNECTIONS: List[Tuple[int, int]] = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL), (LEFT_ANKLE, LEFT_FOOT_INDEX),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    (RIGHT_ANKLE, RIGHT_HEEL), (RIGHT_ANKLE, RIGHT_FOOT_INDEX),
]
# Welcher Landmark-Index ist der Scheitelpunkt (vertex) je berechnetem
# Gelenkwinkel - dort wird im Overlay die Winkelzahl platziert.
JOINT_VERTEX_INDEX: Dict[str, int] = {
    "knee_left": LEFT_KNEE, "knee_right": RIGHT_KNEE,
    "hip_left": LEFT_HIP, "hip_right": RIGHT_HIP,
    "ankle_left": LEFT_ANKLE, "ankle_right": RIGHT_ANKLE,
}

# --------------------------------------------------------------------------
# 1) Konfiguration
# --------------------------------------------------------------------------
@dataclass
class Config:
    model_path: str = "models/pose_landmarker_full.task"
    camera_index: Optional[int] = None              # None = automatisch suchen (find_working_camera)
    csv_path: Optional[str] = "data/output/pose_data.csv"  # None = kein CSV-Export
    min_visibility: float = 0.5                     #je höher desto strenger der Ausschuss wenn Gelenk nicht sichtbar
    up_threshold_deg: float = 160.0                 # Kniewinkel > 160 Grad = UP
    down_threshold_deg: float = 90.0                # Kniewinkel < 90 Grad = DOWN
    fatigue_duration_increase: float = 0.20         # +20% Rep-Dauer
    fatigue_min_angle_increase_deg: float = 10.0    # Knie weniger tief um 10 Grad
    fatigue_hip_deviation_deg: float = 10.0         # Hueftwinkel-Abweichung 10 Grad
    fatigue_baseline_reps: int = 3                  # basierend auf !ersten! 3 Reps wird Mittelwert berechnet, dies ist der Baseline für Fatigue-Detection

# --------------------------------------------------------------------------
# 2) Kamera finden
# --------------------------------------------------------------------------

def find_working_camera(max_index: int = 3) -> int:

    import cv2
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, _ = cap.read()
            cap.release()
            if ok:
                return idx
    raise RuntimeError(
        "Keine funktionierende Kamera gefunden. Falls 'OpenCV: not authorized "
        "to capture video' erscheint: Terminal-Kamerazugriff in "
        "Systemeinstellungen -> Datenschutz & Sicherheit -> Kamera pruefen "
        "und Terminal komplett neu starten."
    )

# --------------------------------------------------------------------------
# 3) MediaPipe-Estimator aufsetzen
# --------------------------------------------------------------------------
def create_pose_landmarker(model_path: str):
    """Erstellt den PoseLandmarker mit explizitem CPU-Delegate.
    Ohne diesen Fix stuerzt mediapipe>=1.0 auf macOS Apple Silicon beim
    Erstellen ab (abort(), EXC_CRASH SIGABRT), unabhaengig vom delegate-
    Setting, weil TensorsToDetectionsCalculator einen im Wheel fehlenden
    Metal-GPU-Service initialisieren will."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    base_options = mp_python.BaseOptions(
        model_asset_path=model_path,
        delegate=mp_python.BaseOptions.Delegate.CPU,
    )
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def extract_world_landmarks(detection_result) -> Optional[List]:
    """Holt die 3D-Weltkoordinaten-Landmarks (x_world/y_world/z_world,
    robust gegen Kamerawinkel) aus einem MediaPipe-Ergebnis, oder None,
    wenn keine Person erkannt wurde. Wird fuer die Winkelberechnung benutzt
    (genauer, kamerawinkel-unabhaengig)."""
    if not detection_result.pose_world_landmarks:
        return None
    return detection_result.pose_world_landmarks[0]


def extract_image_landmarks(detection_result) -> Optional[List]:
    """Holt die Bildraum-Landmarks (x/y normalisiert auf [0,1] relativ zum
    Frame, z relative Tiefe) aus einem MediaPipe-Ergebnis, oder None. Diese
    werden NICHT fuer die Winkelberechnung benutzt (siehe extract_world_
    landmarks), sondern nur zum Zeichnen des Overlays - dafuer braucht es
    Pixelkoordinaten im aktuellen Frame, nicht kamerawinkel-unabhaengige
    Weltkoordinaten."""
    if not detection_result.pose_landmarks:
        return None
    return detection_result.pose_landmarks[0]


def to_pixel(landmark, frame_width: int, frame_height: int) -> Tuple[int, int]:
    """Normalisierte Bildraum-Koordinate (0..1) in Pixelkoordinate umrechnen."""
    return int(landmark.x * frame_width), int(landmark.y * frame_height)


# --------------------------------------------------------------------------
# 4) Winkel-Engine (3D-Weltkoordinaten)
# --------------------------------------------------------------------------
def calculate_angle(a: Sequence[float], vertex: Sequence[float], c: Sequence[float]) -> float:
    """Winkel in Grad am Punkt 'vertex' zwischen den Punkten a und c."""
    v1 = np.asarray(a, dtype=float) - np.asarray(vertex, dtype=float)
    v2 = np.asarray(c, dtype=float) - np.asarray(vertex, dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
    cos_angle = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _point(lm) -> Sequence[float]:
    return (lm.x, lm.y, lm.z)


def _visible(*landmarks, min_visibility: float) -> bool:
    return all(lm.visibility >= min_visibility for lm in landmarks)


def compute_joint_angles(landmarks, min_visibility: float = 0.5) -> Dict[str, Optional[float]]:
    """Berechnet Knie-, Huefte- und Knoechelwinkel fuer beide Seiten.
    Liefert None fuer ein Gelenk, wenn eine der drei Landmarks unter der
    min_visibility-Schwelle liegt (typischer Fall: schlechtes Kamera-Framing)."""

    def angle_or_none(a_idx, v_idx, c_idx):
        a, v, c = landmarks[a_idx], landmarks[v_idx], landmarks[c_idx]
        if not _visible(a, v, c, min_visibility=min_visibility):
            return None
        return calculate_angle(_point(a), _point(v), _point(c))

    return {
        "knee_left": angle_or_none(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
        "knee_right": angle_or_none(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
        "hip_left": angle_or_none(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
        "hip_right": angle_or_none(RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
        "ankle_left": angle_or_none(LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT_INDEX),
        "ankle_right": angle_or_none(RIGHT_KNEE, RIGHT_ANKLE, RIGHT_FOOT_INDEX),
    }


def average_knee_angle(angles: Dict[str, Optional[float]]) -> Optional[float]:
    values = [angles["knee_left"], angles["knee_right"]]
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


# --------------------------------------------------------------------------
# 5) Rep-Counter & Fatigue-Detection (Zustandsmaschine UP/DOWN)
# --------------------------------------------------------------------------
@dataclass
class FatigueStatus:
    is_fatigued: bool = False
    reasons: List[str] = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


class RepCounter:
    """Zaehlt Wiederholungen anhand des gemittelten Kniewinkels.

    Fatigue-Baseline: die ersten `fatigue_baseline_reps` abgeschlossenen Reps
    der Session dienen als dynamische Referenz (separat von den festen
    Zaehl-Schwellwerten). Kriterien sind ODER-verknuepft."""

    def __init__(self, config: Config):
        self.config = config
        self.state = "UP"
        self.rep_count = 0
        self._rep_start_time: Optional[float] = None
        self._rep_min_knee_angle: Optional[float] = None
        self._rep_min_hip_angle: Optional[float] = None
        self._completed_reps: List[dict] = []  # fuer Fatigue-Baseline

    def update(self, angles: Dict[str, Optional[float]], timestamp: float) -> Optional[FatigueStatus]:
        """Fuettert einen Frame in die Zustandsmaschine. Gibt einen
        FatigueStatus zurueck, wenn genau in diesem Frame ein Rep
        abgeschlossen wurde, sonst None."""
        knee = average_knee_angle(angles)
        hip_values = [v for v in (angles["hip_left"], angles["hip_right"]) if v is not None]
        hip = float(np.mean(hip_values)) if hip_values else None
        if knee is None:
            return None  # kein verlaesslicher Winkel -> Frame ueberspringen

        if self.state == "UP" and knee < self.config.down_threshold_deg:
            self.state = "DOWN"
            self._rep_start_time = timestamp
            self._rep_min_knee_angle = knee
            self._rep_min_hip_angle = hip
        elif self.state == "DOWN":
            if self._rep_min_knee_angle is None or knee < self._rep_min_knee_angle:
                self._rep_min_knee_angle = knee
            if hip is not None and (self._rep_min_hip_angle is None or hip < self._rep_min_hip_angle):
                self._rep_min_hip_angle = hip
            if knee > self.config.up_threshold_deg:
                self.state = "UP"
                self.rep_count += 1
                duration = timestamp - self._rep_start_time if self._rep_start_time else 0.0
                rep_data = {
                    "duration": duration,
                    "min_knee_angle": self._rep_min_knee_angle,
                    "min_hip_angle": self._rep_min_hip_angle,
                }
                fatigue = self._evaluate_fatigue(rep_data)
                self._completed_reps.append(rep_data)
                return fatigue
        return None

    def _evaluate_fatigue(self, rep_data: dict) -> FatigueStatus:
        if len(self._completed_reps) < self.config.fatigue_baseline_reps:
            return FatigueStatus(is_fatigued=False)  # Baseline noch nicht voll

        baseline = self._completed_reps[: self.config.fatigue_baseline_reps]
        baseline_duration = float(np.mean([r["duration"] for r in baseline]))
        baseline_min_knee = float(np.mean([r["min_knee_angle"] for r in baseline]))
        baseline_min_hip_vals = [r["min_hip_angle"] for r in baseline if r["min_hip_angle"] is not None]
        baseline_min_hip = float(np.mean(baseline_min_hip_vals)) if baseline_min_hip_vals else None

        reasons = []
        if rep_data["duration"] > baseline_duration * (1 + self.config.fatigue_duration_increase):
            reasons.append("duration")
        if rep_data["min_knee_angle"] > baseline_min_knee + self.config.fatigue_min_angle_increase_deg:
            reasons.append("knee_depth")
        if (
            baseline_min_hip is not None
            and rep_data["min_hip_angle"] is not None
            and abs(rep_data["min_hip_angle"] - baseline_min_hip) > self.config.fatigue_hip_deviation_deg
        ):
            reasons.append("hip_deviation")

        return FatigueStatus(is_fatigued=len(reasons) > 0, reasons=reasons)


# --------------------------------------------------------------------------
# 6) CSV-Export
# --------------------------------------------------------------------------
def csv_header() -> List[str]:
    """Spaltennamen, Reihenfolge fest an csv_row() gekoppelt.

    Enthaelt alles, was pro Frame ohnehin schon berechnet wird:
    - timestamp_s, frame_index: zum Rekonstruieren des zeitlichen Verlaufs
    - rep_count, state: um Frames spaeter einzelnen Wiederholungen zuzuordnen
    - *_deg: alle sechs Gelenkwinkel (None/leer, wenn unter min_visibility)
    - fatigue_*: nur auf der Zeile gesetzt, in der ein Rep abgeschlossen wurde
    - <landmark>_x_world/_y_world/_z_world/_visibility: alle 33 Punkte,
      3D-Weltkoordinaten (kamerawinkel-robust) + Visibility zum nachtraeglichen
      Filtern unzuverlaessiger Frames."""
    cols = ["timestamp_s", "frame_index", "rep_count", "state"]
    cols += [f"{name}_deg" for name in ANGLE_COLUMNS]
    cols += ["fatigue_is_fatigued", "fatigue_reasons"]
    for name in LANDMARK_NAMES:
        cols += [f"{name}_x_world", f"{name}_y_world", f"{name}_z_world", f"{name}_visibility"]
    return cols


def csv_row(
    timestamp_s: float,
    frame_index: int,
    rep_count: int,
    state: str,
    angles: Dict[str, Optional[float]],
    fatigue: Optional[FatigueStatus],
    world_landmarks,
) -> List:
    """Eine Zeile passend zu csv_header(). fatigue wird nur auf der Zeile
    mitgegeben, in der genau in diesem Frame ein Rep abgeschlossen wurde
    (siehe RepCounter.update() - liefert sonst None)."""
    row: List = [timestamp_s, frame_index, rep_count, state]
    row += [angles.get(name) for name in ANGLE_COLUMNS]
    row += [fatigue.is_fatigued if fatigue else "", ",".join(fatigue.reasons) if fatigue else ""]

    if world_landmarks is None:
        row += [""] * (len(LANDMARK_NAMES) * 4)
    else:
        # Die MediaPipe-Weltkoordinaten-Objekte heissen intern .x/.y/.z (nicht
        # .x_world/.y_world/.z_world) - "world" beschreibt nur, aus welcher
        # Liste sie stammen (pose_world_landmarks), nicht den Attributnamen.
        for lm in world_landmarks:
            row += [lm.x, lm.y, lm.z, lm.visibility]
    return row


def open_csv_writer(path: str):
    """Legt den Zielordner an (falls noetig) und oeffnet die CSV-Datei mit
    bereits geschriebenem Header. Rueckgabe: (file_handle, writer) - beide
    werden in run() gehalten und am Ende geschlossen."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    file_handle = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(file_handle)
    writer.writerow(csv_header())
    return file_handle, writer


# --------------------------------------------------------------------------
# 7) Overlay (reine OpenCV-Zeichenfunktionen, keine MediaPipe-Abhaengigkeit)
# --------------------------------------------------------------------------
def draw_skeleton(frame, image_landmarks, min_visibility: float, point_radius: int = 7):
    """Zeichnet alle 33 Punkte (gross, gut sichtbar) und die Verbindungs-
    linien aus POSE_CONNECTIONS. Punkte/Linien unter der visibility-Schwelle
    werden grau statt gruen gezeichnet - das macht Framing-Probleme live im
    Bild sichtbar."""
    import cv2

    if image_landmarks is None:
        return frame

    h, w = frame.shape[:2]
    pixels = [to_pixel(lm, w, h) for lm in image_landmarks]
    visible_flags = [lm.visibility >= min_visibility for lm in image_landmarks]

    GREEN, GRAY = (0, 255, 0), (120, 120, 120)

    for idx_a, idx_b in POSE_CONNECTIONS:
        if idx_a >= len(pixels) or idx_b >= len(pixels):
            continue
        color = GREEN if (visible_flags[idx_a] and visible_flags[idx_b]) else GRAY
        cv2.line(frame, pixels[idx_a], pixels[idx_b], color, 2)

    for idx, (px, py) in enumerate(pixels):
        color = GREEN if visible_flags[idx] else GRAY
        cv2.circle(frame, (px, py), point_radius, color, -1)
        cv2.circle(frame, (px, py), point_radius, (0, 0, 0), 1)  # dunkler Rand fuer Kontrast

    return frame


def draw_joint_angles(frame, angles: Dict[str, Optional[float]], image_landmarks):
    """Schreibt die berechnete Winkelzahl direkt neben den zugehoerigen
    Gelenkpunkt im Bild (statt nur als Liste in der Ecke)."""
    import cv2

    if image_landmarks is None:
        return frame
    h, w = frame.shape[:2]

    for joint_name, value in angles.items():
        if value is None:
            continue
        vertex_idx = JOINT_VERTEX_INDEX.get(joint_name)
        if vertex_idx is None or vertex_idx >= len(image_landmarks):
            continue
        px, py = to_pixel(image_landmarks[vertex_idx], w, h)
        label = f"{value:.0f}"
        cv2.putText(frame, label, (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)   # Kontur
        cv2.putText(frame, label, (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # Text

    return frame


def draw_hud(frame, rep_count: int, state: str, fatigued: bool):
    """Kompakte Statuszeile oben links, mit dunklem Hintergrund-Rechteck
    hinter dem Text - sonst ist weisser Text bei hellem/weissem Kamerabild
    kaum lesbar."""
    import cv2

    lines = [f"Reps: {rep_count}", f"Status: {state}"]
    if fatigued:
        lines.append("FATIGUE WARNING")

    x, y0 = 10, 15
    line_height = 30
    box_w = 260
    box_h = line_height * len(lines) + 10

    # Halbtransparentes Rechteck: Kopie des Frames nehmen, Box drauf zeichnen,
    # dann mit dem Original ueberblenden (addWeighted) statt hart zu ueberschreiben.
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, y0 - 5), (x - 5 + box_w, y0 - 5 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        ty = y0 + line_height * (i + 1) - 8
        color = (0, 0, 255) if line == "FATIGUE WARNING" else (255, 255, 255)
        cv2.putText(frame, line, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame


def draw_overlay(frame, image_landmarks, angles: Dict[str, Optional[float]], rep_count: int,
                  state: str, fatigued: bool, min_visibility: float = 0.5):
    """Verdrahtet alle Overlay-Teilfunktionen: Skelett -> Gelenkwinkel -> HUD."""
    frame = draw_skeleton(frame, image_landmarks, min_visibility)
    frame = draw_joint_angles(frame, angles, image_landmarks)
    frame = draw_hud(frame, rep_count, state, fatigued)
    return frame


# --------------------------------------------------------------------------
# 8) Hauptschleife
# --------------------------------------------------------------------------
def run(config: Config):
    import cv2

    camera_index = config.camera_index if config.camera_index is not None else find_working_camera()

    landmarker = create_pose_landmarker(config.model_path)
    rep_counter = RepCounter(config)

    csv_file, csv_writer_obj = (None, None)
    if config.csv_path:
        csv_file, csv_writer_obj = open_csv_writer(config.csv_path)
        print(f"CSV-Export aktiv: {config.csv_path}")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera-Index {camera_index} liess sich nicht oeffnen.")

    window = "Pose Tracking"
    cv2.namedWindow(window)
    print(
        "Live-Betrieb gestartet. Zum Beenden entweder im Videofenster (nicht "
        "im Terminal!) die Taste 'q' druecken, oder im Terminal Ctrl+C."
    )
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            import mediapipe as mp

            timestamp_ms = int(time.time() * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            world_landmarks = extract_world_landmarks(result)
            image_landmarks = extract_image_landmarks(result)
            angles = {}
            fatigue = None
            state = rep_counter.state
            if world_landmarks is not None:
                # Winkel werden aus Weltkoordinaten berechnet (robust gegen
                # Kamerawinkel), das Overlay zeichnet aber mit den Bildraum-
                # Landmarks (image_landmarks), weil dafuer Pixelkoordinaten
                # im aktuellen Frame gebraucht werden - beide Landmark-Saetze
                # haben dieselbe anatomische Indexzuordnung.
                angles = compute_joint_angles(world_landmarks, config.min_visibility)
                fatigue = rep_counter.update(angles, timestamp_ms / 1000.0)
                state = rep_counter.state

            if csv_writer_obj is not None:
                csv_writer_obj.writerow(
                    csv_row(timestamp_ms / 1000.0, frame_index, rep_counter.rep_count, state,
                            angles, fatigue, world_landmarks)
                )

            frame = draw_overlay(
                frame,
                image_landmarks,
                angles,
                rep_counter.rep_count,
                state,
                fatigued=bool(fatigue and fatigue.is_fatigued),
                min_visibility=config.min_visibility,
            )
            cv2.putText(frame, "'q' zum Beenden (Fenster muss fokussiert sein)",
                        (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow(window, frame)
            # cv2.waitKey() liest Tastendruecke nur, wenn DIESES Fenster den
            # OS-Fokus hat - Klick ins Videofenster, nicht ins Terminal.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_index += 1
    except KeyboardInterrupt:
        # Zuverlaessiger Fallback, falls das Videofenster den Tastendruck aus
        # irgendeinem Grund nicht empfaengt (z.B. Fenstermanager-Eigenheiten).
        print("Per Ctrl+C beendet.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if csv_file is not None:
            csv_file.close()
            print(f"CSV geschrieben: {config.csv_path} ({frame_index} Frames)")


# --------------------------------------------------------------------------
# Selbsttests ohne pytest, ohne Kamera/Modell (reine Funktionen pruefbar)
# --------------------------------------------------------------------------
def _run_selftests():
    from types import SimpleNamespace

    def lm(x, y, z=0.0, vis=1.0):
        return SimpleNamespace(x=x, y=y, z=z, visibility=vis)

    # 1) 90-Grad-Winkel-Test
    assert abs(calculate_angle((0, 1, 0), (0, 0, 0), (1, 0, 0)) - 90.0) < 1e-6

    # 2) compute_joint_angles liefert None bei zu geringer visibility
    landmarks = [lm(0, 0)] * 33
    landmarks[LEFT_HIP] = lm(0, 1, vis=0.1)  # unter Schwelle
    landmarks[LEFT_KNEE] = lm(0, 0.5)
    landmarks[LEFT_ANKLE] = lm(0, 0)
    angles = compute_joint_angles(landmarks, min_visibility=0.5)
    assert angles["knee_left"] is None

    # 3) RepCounter: eine volle Kniebeuge zaehlt genau 1 Rep
    cfg = Config()
    counter = RepCounter(cfg)
    t = 0.0
    for knee_angle in [170, 150, 100, 70, 100, 150, 170]:
        angles = {"knee_left": knee_angle, "knee_right": knee_angle, "hip_left": 170, "hip_right": 170}
        counter.update(angles, t)
        t += 0.1
    assert counter.rep_count == 1, counter.rep_count

    # 4) Fatigue: baseline erst nach 3 Reps aktiv
    cfg2 = Config(fatigue_baseline_reps=1)
    counter2 = RepCounter(cfg2)
    t = 0.0
    for knee_angle in [170, 60, 170]:  # ein tiefer Rep als Baseline (Knie bis 60 Grad)
        counter2.update({"knee_left": knee_angle, "knee_right": knee_angle, "hip_left": 170, "hip_right": 170}, t)
        t += 0.2
    fatigue = None
    for knee_angle in [170, 85, 170]:  # zweiter Rep: nur bis 85 statt 60 Grad -> weniger tief -> Fatigue
        fatigue = counter2.update({"knee_left": knee_angle, "knee_right": knee_angle, "hip_left": 170, "hip_right": 170}, t)
        t += 0.2
    assert fatigue is not None and fatigue.is_fatigued, fatigue

    # 5) CSV: Header- und Zeilenlaenge muessen uebereinstimmen, Landmark-
    # Werte muessen an der richtigen Spalte landen. lm_world() bildet die
    # echten MediaPipe-Weltkoordinaten-Objekte nach: Attribute heissen
    # .x/.y/.z/.visibility, NICHT .x_world/.y_world/.z_world - nur die
    # CSV-Spalten heissen so, zur Klarheit beim Auswerten.
    def lm_world(x, y, z, vis=1.0):
        return SimpleNamespace(x=x, y=y, z=z, visibility=vis)

    header = csv_header()
    world_landmarks = [lm_world(0, 0, 0, vis=0.9) for _ in range(33)]
    world_landmarks[LEFT_KNEE] = lm_world(1.5, 2.5, 3.5, vis=0.9)
    row = csv_row(12.34, 5, 2, "DOWN", {"knee_left": 95.0}, None, world_landmarks)
    assert len(header) == len(row), (len(header), len(row))
    knee_x_idx = header.index("left_knee_x_world")
    assert row[knee_x_idx] == 1.5 and row[knee_x_idx + 1] == 2.5 and row[knee_x_idx + 2] == 3.5

    # CSV ohne erkannte Person (world_landmarks=None) darf nicht crashen
    row_none = csv_row(12.34, 5, 2, "UP", {}, None, None)
    assert len(row_none) == len(header)

    print("Selbsttests bestanden.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/pose_landmarker_full.task")
    parser.add_argument("--camera", type=int, default=None, help="Kamera-Index direkt setzen (Default: automatisch)")
    parser.add_argument("--csv", default="data/output/pose_data.csv", help="Zielpfad fuer CSV-Export")
    parser.add_argument("--no-csv", action="store_true", help="CSV-Export deaktivieren")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        _run_selftests()
    else:
        run(Config(model_path=args.model, camera_index=args.camera, csv_path=None if args.no_csv else args.csv))
