
from __future__ import annotations
import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Unterschiede zur MediaPipe-3D-Pipeline (pose_tracking_mediapipe_V5.py):
# - YOLO-Pose liefert COCO-17-Keypoints (nicht BlazePose-33). Es gibt KEINE
#   heel/foot_index-Punkte, daher lassen sich die Knoechelwinkel (ankle_left/
#   ankle_right) nicht analog berechnen -> ANGLE_COLUMNS enthaelt nur noch
#   Knie- und Hueftwinkel.
# - YOLO-Pose liefert direkt Pixelkoordinaten (x, y) im Originalbild, keine
#   kamerawinkel-robusten 3D-Weltkoordinaten wie MediaPipe -> Winkel werden
#   hier aus 2D-Bildkoordinaten berechnet (weniger robust gegen Kamerawinkel,
#   siehe Diskussion im Bericht: Kap. 3 "Vorgehen/Methoden").
# - Statt "visibility" (0..1, MediaPipe) liefert YOLO-Pose "confidence"
#   (0..1) pro Keypoint - gleiche Rolle, anderer Name.
# - YOLO-Pose erkennt standardmaessig MEHRERE Personen pro Frame. Fuer die
#   Vergleichbarkeit mit der Single-Person-MediaPipe-Pipeline wird hier die
#   Person mit der groessten Bounding-Box (= vermutlich naeheste/Hauptperson
#   im Bild) ausgewaehlt (siehe select_main_person()).
# Quelle Modell/API: Ultralytics-Dokumentation (docs.ultralytics.com/tasks/pose).
# --------------------------------------------------------------------------

# COCO-17-Keypoint-IDs (Ultralytics-Standardreihenfolge)
NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

# Offizielle COCO-17-Namen in Indexreihenfolge (0..16) fuer CSV
LANDMARK_NAMES: List[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Keine Knoechelwinkel - siehe Kommentarblock oben (kein foot_index in COCO-17)
ANGLE_COLUMNS: List[str] = ["knee_left", "knee_right", "hip_left", "hip_right"]

# Skelett-Verbindungslinien fuer das Overlay (ohne Fuss-Segmente)
POSE_CONNECTIONS: List[Tuple[int, int]] = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
]
# Welcher Landmark-Index ist der Scheitelpunkt (vertex) je berechnetem
# Gelenkwinkel - dort wird im Overlay die Winkelzahl platziert.
JOINT_VERTEX_INDEX: Dict[str, int] = {
    "knee_left": LEFT_KNEE, "knee_right": RIGHT_KNEE,
    "hip_left": LEFT_HIP, "hip_right": RIGHT_HIP,
}

# --------------------------------------------------------------------------
# 1) Konfiguration
# --------------------------------------------------------------------------
@dataclass
class Config:
    model_path: str = "yolo11n-pose.pt"    # "n"=nano (schnell); fuer mehr Genauigkeit z.B. "yolo11x-pose.pt" (langsamer)
    device: str = "cpu"                    # "cpu" | "mps" (Apple-GPU) | "cuda"; cpu ist der stabilste Default
    camera_index: Optional[int] = None     # None = automatisch suchen (find_working_camera)
    csv_path: Optional[str] = "data/output/pose_data_yolo.csv"  # None = kein CSV-Export
    min_confidence: float = 0.5            # analog zu min_visibility bei MediaPipe
    up_threshold_deg: float = 160.0        # Kniewinkel > 160 Grad = UP
    down_threshold_deg: float = 90.0       # Kniewinkel < 90 Grad = DOWN
    fatigue_duration_increase: float = 0.20        # +20% Rep-Dauer
    fatigue_min_angle_increase_deg: float = 10.0   # Knie weniger tief um 10 Grad
    fatigue_hip_deviation_deg: float = 10.0        # Hueftwinkel-Abweichung 10 Grad
    fatigue_baseline_reps: int = 3         # basierend auf ersten 3 Reps wird Mittelwert berechnet (Fatigue-Baseline)

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
# 3) YOLO-Pose-Estimator aufsetzen
# --------------------------------------------------------------------------
def create_pose_model(model_path: str):
    """Laedt das YOLO-Pose-Gewicht (wird bei erstem Aufruf automatisch von
    Ultralytics heruntergeladen, falls model_path ein Standardname wie
    'yolo11n-pose.pt' ist und lokal noch nicht existiert)."""
    from ultralytics import YOLO
    return YOLO(model_path)


class MainPerson:
    """Container fuer die ausgewaehlte Zielperson eines Frames: 17x2
    Pixelkoordinaten (xy) und 17 Konfidenzwerte (conf), COCO-17-Reihenfolge."""

    def __init__(self, xy: np.ndarray, conf: np.ndarray):
        self.xy = xy      # shape (17, 2), Pixelkoordinaten im Originalbild
        self.conf = conf  # shape (17,)


def select_main_person(result, min_confidence: float) -> Optional[MainPerson]:
    """YOLO-Pose erkennt potenziell mehrere Personen; hier wird die Person
    mit der groessten Bounding-Box-Flaeche gewaehlt (Annahme: Hauptperson
    steht am naechsten zur Kamera). Gibt None zurueck, wenn niemand erkannt
    wurde oder keine Keypoints vorliegen."""
    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        return None

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()          # (N, 4)
    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    main_idx = int(np.argmax(areas))

    xy_all = result.keypoints.xy.cpu().numpy()             # (N, 17, 2)
    conf_all = result.keypoints.conf.cpu().numpy()         # (N, 17)
    return MainPerson(xy=xy_all[main_idx], conf=conf_all[main_idx])


def to_pixel(x: float, y: float) -> Tuple[int, int]:
    """YOLO-Pose liefert bereits Pixelkoordinaten im Originalbild - anders
    als bei MediaPipe (dort normalisiert 0..1) muss hier nur gerundet werden."""
    return int(x), int(y)


# --------------------------------------------------------------------------
# 4) Winkel-Engine (2D-Pixelkoordinaten)
# --------------------------------------------------------------------------
def calculate_angle(a: Sequence[float], vertex: Sequence[float], c: Sequence[float]) -> float:
    """Winkel in Grad am Punkt 'vertex' zwischen den Punkten a und c."""
    v1 = np.asarray(a, dtype=float) - np.asarray(vertex, dtype=float)
    v2 = np.asarray(c, dtype=float) - np.asarray(vertex, dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
    cos_angle = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _visible(person: MainPerson, *indices: int, min_confidence: float) -> bool:
    return all(person.conf[i] >= min_confidence for i in indices)


def compute_joint_angles(person: MainPerson, min_confidence: float = 0.5) -> Dict[str, Optional[float]]:
    """Berechnet Knie- und Hueftwinkel fuer beide Seiten aus 2D-Pixel-
    koordinaten. Liefert None fuer ein Gelenk, wenn eine der drei Landmarks
    unter der min_confidence-Schwelle liegt."""

    def angle_or_none(a_idx, v_idx, c_idx):
        if not _visible(person, a_idx, v_idx, c_idx, min_confidence=min_confidence):
            return None
        return calculate_angle(person.xy[a_idx], person.xy[v_idx], person.xy[c_idx])

    return {
        "knee_left": angle_or_none(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
        "knee_right": angle_or_none(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
        "hip_left": angle_or_none(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
        "hip_right": angle_or_none(RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
    }


def average_knee_angle(angles: Dict[str, Optional[float]]) -> Optional[float]:
    values = [angles["knee_left"], angles["knee_right"]]
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


# --------------------------------------------------------------------------
# 5) Rep-Counter & Fatigue-Detection (Zustandsmaschine UP/DOWN)
# identisch zur MediaPipe-Pipeline - die Logik arbeitet nur auf dem
# angles-Dict und ist unabhaengig davon, ob die Winkel aus 2D- oder
# 3D-Koordinaten stammen.
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
    der Session dienen als dynamische Referenz. Kriterien sind ODER-verknuepft."""

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

    Unterschied zu MediaPipe: pro Landmark nur x_px/y_px/confidence (3 statt
    4 Spalten) - es gibt keine robuste 3D-Tiefeninformation bei YOLO-Pose."""
    cols = ["timestamp_s", "frame_index", "rep_count", "state"]
    cols += [f"{name}_deg" for name in ANGLE_COLUMNS]
    cols += ["fatigue_is_fatigued", "fatigue_reasons"]
    for name in LANDMARK_NAMES:
        cols += [f"{name}_x_px", f"{name}_y_px", f"{name}_confidence"]
    return cols


def csv_row(
    timestamp_s: float,
    frame_index: int,
    rep_count: int,
    state: str,
    angles: Dict[str, Optional[float]],
    fatigue: Optional[FatigueStatus],
    person: Optional[MainPerson],
) -> List:
    """Eine Zeile passend zu csv_header()."""
    row: List = [timestamp_s, frame_index, rep_count, state]
    row += [angles.get(name) for name in ANGLE_COLUMNS]
    row += [fatigue.is_fatigued if fatigue else "", ",".join(fatigue.reasons) if fatigue else ""]

    if person is None:
        row += [""] * (len(LANDMARK_NAMES) * 3)
    else:
        for i in range(len(LANDMARK_NAMES)):
            x, y = person.xy[i]
            row += [float(x), float(y), float(person.conf[i])]
    return row


def open_csv_writer(path: str):
    """Legt den Zielordner an (falls noetig) und oeffnet die CSV-Datei mit
    bereits geschriebenem Header."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    file_handle = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(file_handle)
    writer.writerow(csv_header())
    return file_handle, writer


# --------------------------------------------------------------------------
# 7) Overlay (reine OpenCV-Zeichenfunktionen, keine YOLO-Abhaengigkeit)
# --------------------------------------------------------------------------
def draw_skeleton(frame, person: Optional[MainPerson], min_confidence: float, point_radius: int = 7):
    """Zeichnet alle 17 Punkte und die Verbindungslinien aus POSE_CONNECTIONS.
    Punkte/Linien unter der confidence-Schwelle werden grau statt gruen
    gezeichnet."""
    import cv2

    if person is None:
        return frame

    pixels = [to_pixel(x, y) for x, y in person.xy]
    visible_flags = [c >= min_confidence for c in person.conf]

    GREEN, GRAY = (0, 255, 0), (120, 120, 120)

    for idx_a, idx_b in POSE_CONNECTIONS:
        color = GREEN if (visible_flags[idx_a] and visible_flags[idx_b]) else GRAY
        cv2.line(frame, pixels[idx_a], pixels[idx_b], color, 2)

    for idx, (px, py) in enumerate(pixels):
        color = GREEN if visible_flags[idx] else GRAY
        cv2.circle(frame, (px, py), point_radius, color, -1)
        cv2.circle(frame, (px, py), point_radius, (0, 0, 0), 1)  # dunkler Rand fuer Kontrast

    return frame


def draw_joint_angles(frame, angles: Dict[str, Optional[float]], person: Optional[MainPerson]):
    """Schreibt die berechnete Winkelzahl direkt neben den zugehoerigen
    Gelenkpunkt im Bild."""
    import cv2

    if person is None:
        return frame

    for joint_name, value in angles.items():
        if value is None:
            continue
        vertex_idx = JOINT_VERTEX_INDEX.get(joint_name)
        if vertex_idx is None:
            continue
        px, py = to_pixel(*person.xy[vertex_idx])
        label = f"{value:.0f}"
        cv2.putText(frame, label, (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)   # Kontur
        cv2.putText(frame, label, (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # Text

    return frame


def draw_hud(frame, rep_count: int, state: str, fatigued: bool):
    """Kompakte Statuszeile oben links, mit dunklem Hintergrund-Rechteck."""
    import cv2

    lines = [f"Reps: {rep_count}", f"Status: {state}"]
    if fatigued:
        lines.append("FATIGUE WARNING")

    x, y0 = 10, 15
    line_height = 30
    box_w = 260
    box_h = line_height * len(lines) + 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, y0 - 5), (x - 5 + box_w, y0 - 5 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        ty = y0 + line_height * (i + 1) - 8
        color = (0, 0, 255) if line == "FATIGUE WARNING" else (255, 255, 255)
        cv2.putText(frame, line, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame


def draw_overlay(frame, person: Optional[MainPerson], angles: Dict[str, Optional[float]], rep_count: int,
                  state: str, fatigued: bool, min_confidence: float = 0.5):
    """Verdrahtet alle Overlay-Teilfunktionen: Skelett -> Gelenkwinkel -> HUD."""
    frame = draw_skeleton(frame, person, min_confidence)
    frame = draw_joint_angles(frame, angles, person)
    frame = draw_hud(frame, rep_count, state, fatigued)
    return frame


# --------------------------------------------------------------------------
# 8) Hauptschleife
# --------------------------------------------------------------------------
def run(config: Config):
    import cv2

    camera_index = config.camera_index if config.camera_index is not None else find_working_camera()

    model = create_pose_model(config.model_path)
    rep_counter = RepCounter(config)

    csv_file, csv_writer_obj = (None, None)
    if config.csv_path:
        csv_file, csv_writer_obj = open_csv_writer(config.csv_path)
        print(f"CSV-Export aktiv: {config.csv_path}")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera-Index {camera_index} liess sich nicht oeffnen.")

    window = "Pose Tracking (YOLO-Pose)"
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

            timestamp_ms = int(time.time() * 1000)
            results = model.predict(source=frame, device=config.device, verbose=False)
            person = select_main_person(results[0], config.min_confidence)

            angles = {}
            fatigue = None
            state = rep_counter.state
            if person is not None:
                angles = compute_joint_angles(person, config.min_confidence)
                fatigue = rep_counter.update(angles, timestamp_ms / 1000.0)
                state = rep_counter.state

            if csv_writer_obj is not None:
                csv_writer_obj.writerow(
                    csv_row(timestamp_ms / 1000.0, frame_index, rep_counter.rep_count, state,
                            angles, fatigue, person)
                )

            frame = draw_overlay(
                frame,
                person,
                angles,
                rep_counter.rep_count,
                state,
                fatigued=bool(fatigue and fatigue.is_fatigued),
                min_confidence=config.min_confidence,
            )
            cv2.putText(frame, "'q' zum Beenden (Fenster muss fokussiert sein)",
                        (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_index += 1
    except KeyboardInterrupt:
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

    # 1) 90-Grad-Winkel-Test
    assert abs(calculate_angle((0, 1), (0, 0), (1, 0)) - 90.0) < 1e-6

    # 2) compute_joint_angles liefert None bei zu geringer confidence
    xy = np.zeros((17, 2))
    conf = np.ones(17)
    xy[LEFT_HIP] = [0, 1]
    conf[LEFT_HIP] = 0.1  # unter Schwelle
    xy[LEFT_KNEE] = [0, 0.5]
    xy[LEFT_ANKLE] = [0, 0]
    person = MainPerson(xy=xy, conf=conf)
    angles = compute_joint_angles(person, min_confidence=0.5)
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

    # 4) Fatigue: baseline erst nach 1 Rep aktiv (Testkonfiguration)
    cfg2 = Config(fatigue_baseline_reps=1)
    counter2 = RepCounter(cfg2)
    t = 0.0
    for knee_angle in [170, 60, 170]:
        counter2.update({"knee_left": knee_angle, "knee_right": knee_angle, "hip_left": 170, "hip_right": 170}, t)
        t += 0.2
    fatigue = None
    for knee_angle in [170, 85, 170]:
        fatigue = counter2.update({"knee_left": knee_angle, "knee_right": knee_angle, "hip_left": 170, "hip_right": 170}, t)
        t += 0.2
    assert fatigue is not None and fatigue.is_fatigued, fatigue

    # 5) CSV: Header- und Zeilenlaenge muessen uebereinstimmen
    header = csv_header()
    xy2 = np.zeros((17, 2))
    conf2 = np.full(17, 0.9)
    xy2[LEFT_KNEE] = [1.5, 2.5]
    person2 = MainPerson(xy=xy2, conf=conf2)
    row = csv_row(12.34, 5, 2, "DOWN", {"knee_left": 95.0}, None, person2)
    assert len(header) == len(row), (len(header), len(row))
    knee_x_idx = header.index("left_knee_x_px")
    assert row[knee_x_idx] == 1.5 and row[knee_x_idx + 1] == 2.5

    # CSV ohne erkannte Person (person=None) darf nicht crashen
    row_none = csv_row(12.34, 5, 2, "UP", {}, None, None)
    assert len(row_none) == len(header)

    print("Selbsttests bestanden.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--camera", type=int, default=None, help="Kamera-Index direkt setzen (Default: automatisch)")
    parser.add_argument("--csv", default="data/output/pose_data_yolo.csv", help="Zielpfad fuer CSV-Export")
    parser.add_argument("--no-csv", action="store_true", help="CSV-Export deaktivieren")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        _run_selftests()
    else:
        run(Config(model_path=args.model, device=args.device, camera_index=args.camera,
                    csv_path=None if args.no_csv else args.csv))
