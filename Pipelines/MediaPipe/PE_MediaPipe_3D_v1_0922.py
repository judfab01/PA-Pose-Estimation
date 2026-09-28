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

# Offizielle BlazePose-33-Namen in Indexreihenfolge (0..32) fuer CSV
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

# Farben (BGR)
COLOR_GOOD = (0, 200, 0)
COLOR_WARN = (0, 220, 255)
COLOR_NEUTRAL = (150, 150, 150)
COLOR_FATIGUE = (0, 165, 255)

# --------------------------------------------------------------------------
# 1) Konfiguration
# --------------------------------------------------------------------------
@dataclass
class Config:
    model_path: str = "Models/models/pose_landmarker_full.task"
    camera_index: Optional[int] = None              # None = automatisch suchen (find_working_camera)
    csv_path: Optional[str] = "data/output/pose_data.csv"        # Frame-CSV, None = aus
    rep_csv_path: Optional[str] = "data/output/rep_results.csv"  # Rep-CSV, None = aus
    min_visibility: float = 0.5                     # je hoeher desto strenger der Ausschuss wenn Gelenk nicht sichtbar

    # --- Rep-Zustandsmaschine (Kniewinkel: 180 = gestreckt) ---
    up_threshold_deg: float = 160.0                 # Kniewinkel >= 160 = Rep abgeschlossen
    attempt_start_deg: float = 150.0                # Kniewinkel < 150 = Rep-Versuch beginnt
    target_knee_deg: float = 90.0                   # Zielwinkel (individuell durch Physio festlegen)
    target_tolerance_deg: float = 10.0              # Ziel gilt als erreicht bei <= target + tolerance
    min_attempt_progress: float = 0.25              # Versuche unter 25% Zieltiefe werden verworfen (kein Feedback)
    turn_hysteresis_deg: float = 5.0                # Umkehrpunkt: Winkel muss 5 Grad ueber Minimum steigen

    # --- Formkriterien (pro Rep) ---
    min_phase_duration_s: float = 0.4               # Senken bzw. Heben schneller als 0.4 s = "Zu schnell"
    max_asymmetry_deg: float = 15.0                 # |Knie links - Knie rechts| am tiefsten Punkt
    hip_min_allowed_deg: Optional[float] = None     # optional: Huefte am tiefsten Punkt mindestens (Rumpfneigung)

    # --- Signalqualitaet ---
    smoothing_alpha: float = 0.4                    # EMA-Glaettung, 1.0 = aus
    max_gap_s: float = 0.3                          # Tracking-Luecke im Rep laenger = Rep ungueltig
    abort_gap_s: float = 2.0                        # Luecke laenger = Versuch abgebrochen
    min_quality: float = 0.8                        # Anteil gueltiger Frames im Rep

    # --- UX ---
    feedback_hold_s: float = 1.5                    # Anzeigedauer des Rep-Ergebnisses
    reps_per_set: int = 10                          # nur Anzeige "Gut x/y"

    # --- Ermuedung ueber Velocity Loss (Sanchez-Medina & Gonzalez-Badillo, 2011) ---
    min_omega_phase_s: float = 0.25                 # Senkphase kuerzer = keine Geschwindigkeitsberechnung
    vl_warn_pct: float = 20.0                       # ANNAHME aus Kraftsport-Literatur, fuer Rehab nicht validiert
    fatigue_min_valid_reps: int = 3                 # mind. so viele gueltige Reps im Satz
    fatigue_consecutive_reps: int = 2               # so viele Reps in Folge ueber vl_warn_pct

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
# 4) Winkel-Engine (3D-Weltkoordinaten) + Glaettung
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


def _mean_valid(*values: Optional[float]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def average_knee_angle(angles: Dict[str, Optional[float]]) -> Optional[float]:
    return _mean_valid(angles.get("knee_left"), angles.get("knee_right"))


class AngleSmoother:
    """Exponentielle Glaettung (EMA) je Gelenkwinkel. alpha=1.0 = keine
    Glaettung. Faellt ein Winkel aus (None), wird sein Zustand zurueckgesetzt,
    damit nach der Luecke nicht mit einem veralteten Wert gemischt wird."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self._state: Dict[str, Optional[float]] = {}

    def reset(self):
        self._state = {}

    def smooth(self, angles: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {}
        for name, value in angles.items():
            if value is None:
                self._state[name] = None
                out[name] = None
                continue
            prev = self._state.get(name)
            smoothed = value if prev is None else prev + self.alpha * (value - prev)
            self._state[name] = smoothed
            out[name] = smoothed
        return out


# --------------------------------------------------------------------------
# 5) Rep-Counter, Bewertung und Ermuedung (Velocity Loss)
# --------------------------------------------------------------------------
@dataclass
class RepResult:
    """Ergebnis eines abgeschlossenen Rep-Versuchs.
    status: GOOD (Ziel erreicht, Form ok), FORM (Ziel erreicht, Formhinweis),
            PARTIAL (Ziel nicht erreicht), INVALID (Tracking unzureichend)."""
    index: int
    set_index: int
    status: str
    reasons: List[str]
    timestamp_end: float
    min_knee: float
    depth_ratio: float
    t_down: float
    t_up: float
    omega_deg_s: Optional[float]
    vl_pct: Optional[float]
    asymmetry_deg: Optional[float]
    min_hip: Optional[float]
    quality: float


class RepCounter:
    """Zustandsmaschine STANDING -> DESCENDING -> ASCENDING -> STANDING auf dem
    gemittelten Kniewinkel.

    - Jeder Versuch unterhalb attempt_start_deg wird bewertet (nicht nur tiefe Reps).
    - Ermuedung: Velocity Loss VL = 100 * (w_best - w_i) / w_best mit
      w_i = (theta_start - theta_min) / t_down [Grad/s]
      (Sanchez-Medina & Gonzalez-Badillo 2011, Winkelgeschwindigkeit als Proxy).
      Ermuedung aktiv, wenn VL mehrere Reps in Folge >= vl_warn_pct."""

    def __init__(self, config: Config):
        self.config = config
        self.state = "STANDING"
        self.knee: Optional[float] = None       # aktueller (geglaetteter) Kniewinkel
        self.progress = 0.0                     # 0 = Startwinkel, 1 = Zielwinkel
        self.tracking_ok = False
        self.good_count = 0                     # gute Reps im aktuellen Satz
        self.attempt_count = 0                  # alle bewerteten Versuche (gesamt)
        self.set_index = 1
        self.fatigue_active = False
        self.last_vl: Optional[float] = None
        self.results: List[RepResult] = []
        self._last_valid_ts: Optional[float] = None
        self._omegas: List[float] = []
        self._consecutive_slow = 0
        self._reset_attempt()

    # ---- Hilfsfunktionen -------------------------------------------------
    def _reset_attempt(self):
        self._t_start: Optional[float] = None
        self._theta_start: Optional[float] = None
        self._min_knee: Optional[float] = None
        self._t_min: Optional[float] = None
        self._min_hip: Optional[float] = None
        self._min_sides: Tuple[Optional[float], Optional[float]] = (None, None)
        self._frames_total = 0
        self._frames_valid = 0
        self._gap_flag = False

    def _depth_ratio(self, knee_min: float) -> float:
        cfg = self.config
        return (cfg.attempt_start_deg - knee_min) / (cfg.attempt_start_deg - cfg.target_knee_deg)

    def _track_min(self, knee, hip, angles, timestamp):
        self._min_knee = knee
        self._t_min = timestamp
        self._min_hip = hip
        self._min_sides = (angles.get("knee_left"), angles.get("knee_right"))

    def reset_set(self):
        """Neuer Satz: Zaehler und Ermuedungs-Referenz zuruecksetzen."""
        self.set_index += 1
        self.good_count = 0
        self._omegas = []
        self._consecutive_slow = 0
        self.fatigue_active = False
        self.last_vl = None

    # ---- Hauptupdate -----------------------------------------------------
    def update(self, angles: Dict[str, Optional[float]], timestamp: float) -> Optional[RepResult]:
        """Fuettert einen Frame ein (auch Frames ohne Person: angles={}).
        Gibt ein RepResult zurueck, wenn genau in diesem Frame ein Versuch
        bewertet wurde, sonst None."""
        cfg = self.config
        knee = average_knee_angle(angles)
        hip = _mean_valid(angles.get("hip_left"), angles.get("hip_right"))
        self.knee = knee

        if knee is None:
            if self._last_valid_ts is None:
                self.tracking_ok = False
            else:
                self.tracking_ok = (timestamp - self._last_valid_ts) <= cfg.max_gap_s
        else:
            self.tracking_ok = True

        if self.state != "STANDING":
            self._frames_total += 1

        if knee is None:
            if (
                self.state != "STANDING"
                and self._last_valid_ts is not None
                and timestamp - self._last_valid_ts > cfg.abort_gap_s
            ):
                return self._abort(timestamp)
            return None

        if self.state != "STANDING":
            self._frames_valid += 1
            if self._last_valid_ts is not None and timestamp - self._last_valid_ts > cfg.max_gap_s:
                self._gap_flag = True
        self._last_valid_ts = timestamp

        self.progress = float(np.clip(
            (cfg.attempt_start_deg - knee) / (cfg.attempt_start_deg - cfg.target_knee_deg), 0.0, 1.25))

        if self.state == "STANDING":
            if knee < cfg.attempt_start_deg:
                self.state = "DESCENDING"
                self._t_start = timestamp
                self._theta_start = knee
                self._track_min(knee, hip, angles, timestamp)
                self._frames_total = 1
                self._frames_valid = 1
            return None

        if self.state == "DESCENDING":
            if knee < self._min_knee:
                self._track_min(knee, hip, angles, timestamp)
            elif knee > self._min_knee + cfg.turn_hysteresis_deg:
                self.state = "ASCENDING"
        elif self.state == "ASCENDING":
            if knee < self._min_knee:  # Patient geht nochmals tiefer
                self.state = "DESCENDING"
                self._track_min(knee, hip, angles, timestamp)

        if knee >= cfg.up_threshold_deg:
            return self._finish(timestamp)
        return None

    # ---- Bewertung -------------------------------------------------------
    def _finish(self, timestamp: float) -> Optional[RepResult]:
        cfg = self.config
        depth_ratio = self._depth_ratio(self._min_knee)
        self.state = "STANDING"
        if depth_ratio < cfg.min_attempt_progress:
            self._reset_attempt()
            return None  # zu kleine Bewegung: kein Rep-Versuch

        t_down = self._t_min - self._t_start
        t_up = timestamp - self._t_min
        quality = self._frames_valid / max(self._frames_total, 1)
        left, right = self._min_sides
        asym = abs(left - right) if (left is not None and right is not None) else None

        if self._gap_flag or quality < cfg.min_quality:
            status, reasons = "INVALID", ["tracking"]
        elif self._min_knee > cfg.target_knee_deg + cfg.target_tolerance_deg:
            status, reasons = "PARTIAL", ["depth"]
        else:
            reasons = []
            if t_down < cfg.min_phase_duration_s or t_up < cfg.min_phase_duration_s:
                reasons.append("tempo")
            if asym is not None and asym > cfg.max_asymmetry_deg:
                reasons.append("asymmetry")
            if cfg.hip_min_allowed_deg is not None and self._min_hip is not None \
                    and self._min_hip < cfg.hip_min_allowed_deg:
                reasons.append("trunk_lean")
            status = "FORM" if reasons else "GOOD"

        return self._make_result(timestamp, status, reasons, t_down, t_up, depth_ratio, quality, asym)

    def _abort(self, timestamp: float) -> Optional[RepResult]:
        """Person laenger als abort_gap_s verschwunden: Versuch beenden."""
        depth_ratio = self._depth_ratio(self._min_knee)
        self.state = "STANDING"
        if depth_ratio < self.config.min_attempt_progress:
            self._reset_attempt()
            return None
        quality = self._frames_valid / max(self._frames_total, 1)
        return self._make_result(timestamp, "INVALID", ["tracking"],
                                 self._t_min - self._t_start, 0.0, depth_ratio, quality, None)

    def _make_result(self, timestamp, status, reasons, t_down, t_up, depth_ratio, quality, asym) -> RepResult:
        self.attempt_count += 1
        result = RepResult(
            index=self.attempt_count, set_index=self.set_index, status=status, reasons=reasons,
            timestamp_end=timestamp, min_knee=self._min_knee, depth_ratio=depth_ratio,
            t_down=t_down, t_up=t_up, omega_deg_s=None, vl_pct=None,
            asymmetry_deg=asym, min_hip=self._min_hip, quality=quality,
        )
        if status == "GOOD":
            self.good_count += 1
        if status != "INVALID":
            self._apply_velocity_loss(result)
        self.results.append(result)
        self._reset_attempt()
        return result

    def _apply_velocity_loss(self, result: RepResult):
        """VL_i = 100 * (w_best - w_i) / w_best, w_i in Grad/s (Senkphase).
        w_best = hoechstes w aller gueltigen Reps des Satzes (inkl. aktuellem).
        Alle nicht ungueltigen Reps zaehlen, da w die Tiefe herausnormiert."""
        cfg = self.config
        if result.t_down < cfg.min_omega_phase_s:
            return
        omega = (self._theta_start - result.min_knee) / result.t_down
        if omega <= 0:
            return
        result.omega_deg_s = omega
        self._omegas.append(omega)
        best = max(self._omegas)
        result.vl_pct = 100.0 * (best - omega) / best
        self.last_vl = result.vl_pct
        if result.vl_pct >= cfg.vl_warn_pct:
            self._consecutive_slow += 1
        else:
            self._consecutive_slow = 0
        self.fatigue_active = (
            len(self._omegas) >= cfg.fatigue_min_valid_reps
            and self._consecutive_slow >= cfg.fatigue_consecutive_reps
        )


# --------------------------------------------------------------------------
# 6) Feedback-Banner (zeitbasiert statt Ein-Frame-Anzeige)
# --------------------------------------------------------------------------
REASON_TEXT: Dict[str, str] = {
    "tempo": "Zu schnell",
    "asymmetry": "Seiten ungleich",
    "trunk_lean": "Oberkoerper aufrechter",
    "tracking": "Koerper nicht sichtbar",
}


def feedback_text(result: RepResult, config: Config) -> Tuple[str, Tuple[int, int, int]]:
    """Text und Farbe des Rep-Ergebnisses. ASCII, da cv2.putText keine Umlaute zeichnet."""
    if result.status == "GOOD":
        return "Gut", COLOR_GOOD
    if result.status == "PARTIAL":
        return f"Zu flach: {result.min_knee:.0f} Grad, Ziel {config.target_knee_deg:.0f}", COLOR_WARN
    if result.status == "FORM":
        return " / ".join(REASON_TEXT.get(r, r) for r in result.reasons), COLOR_WARN
    return "Nicht gewertet: Koerper nicht sichtbar", COLOR_NEUTRAL


class FeedbackBanner:
    """Haelt das letzte Rep-Ergebnis fuer hold_s Sekunden sichtbar."""

    def __init__(self, hold_s: float):
        self.hold_s = hold_s
        self._text: Optional[str] = None
        self._color = COLOR_NEUTRAL
        self._t = -1e9

    def push(self, result: RepResult, config: Config, now: float):
        self._text, self._color = feedback_text(result, config)
        self._t = now

    def current(self, now: float) -> Optional[Tuple[str, Tuple[int, int, int]]]:
        if self._text is not None and (now - self._t) < self.hold_s:
            return self._text, self._color
        return None


# --------------------------------------------------------------------------
# 7) CSV-Export (Frame-CSV und Rep-CSV)
# --------------------------------------------------------------------------
def csv_header() -> List[str]:
    """Spaltennamen der Frame-CSV, Reihenfolge fest an csv_row() gekoppelt.

    - timestamp_s, frame_index: zeitlicher Verlauf
    - good_reps, state: Zuordnung der Frames zu Wiederholungen
    - *_deg: alle sechs Gelenkwinkel (leer, wenn unter min_visibility)
    - rep_status, rep_reasons, rep_vl_pct: nur auf der Zeile gesetzt, in der ein Rep bewertet wurde
    - fatigue_active: Ermuedungs-Status (Trend ueber Reps) pro Frame
    - <landmark>_x_world/_y_world/_z_world/_visibility: alle 33 Punkte,
      3D-Weltkoordinaten (kamerawinkel-robust) + Visibility."""
    cols = ["timestamp_s", "frame_index", "good_reps", "state"]
    cols += [f"{name}_deg" for name in ANGLE_COLUMNS]
    cols += ["rep_status", "rep_reasons", "rep_vl_pct", "fatigue_active"]
    for name in LANDMARK_NAMES:
        cols += [f"{name}_x_world", f"{name}_y_world", f"{name}_z_world", f"{name}_visibility"]
    return cols


def csv_row(
    timestamp_s: float,
    frame_index: int,
    good_count: int,
    state: str,
    angles: Dict[str, Optional[float]],
    result: Optional[RepResult],
    fatigue_active: bool,
    world_landmarks,
) -> List:
    """Eine Zeile passend zu csv_header(). result nur auf der Zeile, in der
    genau in diesem Frame ein Rep bewertet wurde."""
    row: List = [timestamp_s, frame_index, good_count, state]
    row += [angles.get(name) for name in ANGLE_COLUMNS]
    row += [
        result.status if result else "",
        ",".join(result.reasons) if result else "",
        result.vl_pct if (result and result.vl_pct is not None) else "",
        fatigue_active,
    ]

    if world_landmarks is None:
        row += [""] * (len(LANDMARK_NAMES) * 4)
    else:
        # Die MediaPipe-Weltkoordinaten-Objekte heissen intern .x/.y/.z (nicht
        # .x_world/.y_world/.z_world) - "world" beschreibt nur, aus welcher
        # Liste sie stammen (pose_world_landmarks), nicht den Attributnamen.
        for lm in world_landmarks:
            row += [lm.x, lm.y, lm.z, lm.visibility]
    return row


def rep_csv_header() -> List[str]:
    return [
        "rep_index", "set_index", "status", "reasons", "timestamp_end_s",
        "min_knee_deg", "depth_ratio", "t_down_s", "t_up_s",
        "omega_deg_s", "vl_pct", "asymmetry_deg", "min_hip_deg", "quality",
    ]


def rep_csv_row(r: RepResult) -> List:
    def opt(v):
        return "" if v is None else v
    return [
        r.index, r.set_index, r.status, ",".join(r.reasons), r.timestamp_end,
        r.min_knee, r.depth_ratio, r.t_down, r.t_up,
        opt(r.omega_deg_s), opt(r.vl_pct), opt(r.asymmetry_deg), opt(r.min_hip), r.quality,
    ]


def open_csv_writer(path: str, header: List[str]):
    """Legt den Zielordner an (falls noetig) und oeffnet die CSV-Datei mit
    bereits geschriebenem Header. Rueckgabe: (file_handle, writer) - beide
    werden in run() gehalten und am Ende geschlossen."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    file_handle = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(file_handle)
    writer.writerow(header)
    return file_handle, writer


# --------------------------------------------------------------------------
# 8) Overlay (reine OpenCV-Zeichenfunktionen, keine MediaPipe-Abhaengigkeit)
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


def draw_depth_bar(frame, counter: RepCounter, config: Config):
    """Vertikaler Tiefen-Balken am rechten Rand. Oben = Ausgangswinkel,
    unten = Ziel. Dunkelgruene Zone = Zielbereich (Ziel + Toleranz und tiefer).
    Der Balken fuellt sich beim Absenken (gelb), in der Zielzone wird er gruen."""
    import cv2

    h, w = frame.shape[:2]
    x0, x1 = w - 70, w - 40
    top, bottom = int(h * 0.2), int(h * 0.8)
    span_max = 1.25  # Balken reicht bis 125% der Zieltiefe

    def y_of(p: float) -> int:
        p = min(max(p, 0.0), span_max)
        return int(top + (p / span_max) * (bottom - top))

    p_reach = (config.attempt_start_deg - (config.target_knee_deg + config.target_tolerance_deg)) / (
        config.attempt_start_deg - config.target_knee_deg)

    cv2.rectangle(frame, (x0, top), (x1, bottom), (40, 40, 40), -1)
    cv2.rectangle(frame, (x0, y_of(p_reach)), (x1, bottom), (0, 110, 0), -1)

    if counter.knee is not None:
        reached = counter.knee <= config.target_knee_deg + config.target_tolerance_deg
        y_fill = y_of(counter.progress)
        cv2.rectangle(frame, (x0, top), (x1, y_fill), COLOR_GOOD if reached else COLOR_WARN, -1)
        cv2.line(frame, (x0 - 8, y_fill), (x1 + 8, y_fill), (255, 255, 255), 3)
        cv2.putText(frame, f"{counter.knee:.0f}", (x0 - 8, bottom + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.rectangle(frame, (x0, top), (x1, bottom), (255, 255, 255), 1)
    cv2.putText(frame, "Tiefe", (x0 - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def draw_banner(frame, text: str, color: Tuple[int, int, int]):
    """Rep-Ergebnis als farbiges Banner oben mittig."""
    import cv2

    h, w = frame.shape[:2]
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y, pad = (w - tw) // 2, int(h * 0.12), 14
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + pad), color, -1)
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thickness)
    return frame


def draw_tracking_notice(frame):
    """Neutrales graues Band bei Trackingverlust (kein Rot, keine Bewertung)."""
    import cv2

    h, w = frame.shape[:2]
    text = "Bitte ganz ins Bild treten"
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y, pad = (w - tw) // 2, h // 2, 16
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + pad), COLOR_NEUTRAL, -1)
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thickness)
    return frame


def draw_hud(frame, counter: RepCounter, config: Config):
    """Statuszeile oben links mit dunklem Hintergrund-Rechteck. Enthaelt
    gute Reps im Satz, Zustand, aktuellen Velocity Loss und die persistente
    Ermuedungsanzeige (bleibt, bis ein Rep wieder unter der VL-Schwelle liegt)."""
    import cv2

    lines: List[Tuple[str, Tuple[int, int, int]]] = [
        (f"Satz {counter.set_index}  Gut: {counter.good_count}/{config.reps_per_set}", (255, 255, 255)),
        (f"Status: {counter.state}", (255, 255, 255)),
    ]
    if counter.last_vl is not None:
        lines.append((f"Tempo-Verlust: {counter.last_vl:.0f}%", (255, 255, 255)))
    if counter.fatigue_active:
        lines.append(("ERMUEDUNG: Tempo sinkt", COLOR_FATIGUE))

    x, y0 = 10, 15
    line_height = 30
    box_w = 340
    box_h = line_height * len(lines) + 10

    # Halbtransparentes Rechteck: Kopie des Frames nehmen, Box drauf zeichnen,
    # dann mit dem Original ueberblenden (addWeighted) statt hart zu ueberschreiben.
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, y0 - 5), (x - 5 + box_w, y0 - 5 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, (line, color) in enumerate(lines):
        ty = y0 + line_height * (i + 1) - 8
        cv2.putText(frame, line, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame


def draw_overlay(frame, image_landmarks, angles: Dict[str, Optional[float]], counter: RepCounter,
                 config: Config, banner: Optional[Tuple[str, Tuple[int, int, int]]]):
    """Verdrahtet alle Overlay-Teilfunktionen: Skelett -> Gelenkwinkel ->
    Tiefen-Balken -> HUD -> Banner bzw. Tracking-Hinweis."""
    frame = draw_skeleton(frame, image_landmarks, config.min_visibility)
    frame = draw_joint_angles(frame, angles, image_landmarks)
    frame = draw_depth_bar(frame, counter, config)
    frame = draw_hud(frame, counter, config)
    if banner is not None:
        frame = draw_banner(frame, banner[0], banner[1])
    elif not counter.tracking_ok:
        frame = draw_tracking_notice(frame)
    return frame


# --------------------------------------------------------------------------
# 9) Hauptschleife
# --------------------------------------------------------------------------
def run(config: Config):
    import cv2
    import mediapipe as mp

    camera_index = config.camera_index if config.camera_index is not None else find_working_camera()

    landmarker = create_pose_landmarker(config.model_path)
    rep_counter = RepCounter(config)
    smoother = AngleSmoother(config.smoothing_alpha)
    banner = FeedbackBanner(config.feedback_hold_s)

    csv_file, csv_writer_obj = (None, None)
    if config.csv_path:
        csv_file, csv_writer_obj = open_csv_writer(config.csv_path, csv_header())
        print(f"Frame-CSV aktiv: {config.csv_path}")
    rep_file, rep_writer = (None, None)
    if config.rep_csv_path:
        rep_file, rep_writer = open_csv_writer(config.rep_csv_path, rep_csv_header())
        print(f"Rep-CSV aktiv: {config.rep_csv_path}")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera-Index {camera_index} liess sich nicht oeffnen.")

    window = "Pose Tracking"
    cv2.namedWindow(window)
    print(
        "Live-Betrieb gestartet. Videofenster fokussieren: 'q' = beenden, "
        "'n' = neuer Satz. Alternativ im Terminal Ctrl+C."
    )
    frame_index = 0
    t0 = time.monotonic()
    last_ms = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Monotone Zeit ab Sessionstart (unabhaengig von Systemuhr-Spruengen);
            # detect_for_video verlangt streng steigende Zeitstempel.
            timestamp_ms = int((time.monotonic() - t0) * 1000)
            if timestamp_ms <= last_ms:
                timestamp_ms = last_ms + 1
            last_ms = timestamp_ms
            now = timestamp_ms / 1000.0

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            world_landmarks = extract_world_landmarks(result)
            image_landmarks = extract_image_landmarks(result)

            # Winkel aus Weltkoordinaten (kamerawinkel-robust), Overlay mit
            # Bildraum-Landmarks. Ohne Person wird trotzdem update() mit leeren
            # Winkeln aufgerufen, damit Tracking-Luecken erkannt werden.
            if world_landmarks is not None:
                raw_angles = compute_joint_angles(world_landmarks, config.min_visibility)
                angles = smoother.smooth(raw_angles)
            else:
                smoother.reset()
                angles = {}

            rep_result = rep_counter.update(angles, now)
            if rep_result is not None:
                banner.push(rep_result, config, now)
                if rep_writer is not None:
                    rep_writer.writerow(rep_csv_row(rep_result))
                    rep_file.flush()

            if csv_writer_obj is not None:
                csv_writer_obj.writerow(
                    csv_row(now, frame_index, rep_counter.good_count, rep_counter.state,
                            angles, rep_result, rep_counter.fatigue_active, world_landmarks)
                )

            frame = draw_overlay(frame, image_landmarks, angles, rep_counter, config, banner.current(now))
            cv2.putText(frame, "'q' beenden, 'n' neuer Satz (Fenster fokussieren)",
                        (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow(window, frame)
            # cv2.waitKey() liest Tastendruecke nur, wenn DIESES Fenster den
            # OS-Fokus hat - Klick ins Videofenster, nicht ins Terminal.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("n"):
                rep_counter.reset_set()
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
            print(f"Frame-CSV geschrieben: {config.csv_path} ({frame_index} Frames)")
        if rep_file is not None:
            rep_file.close()
            print(f"Rep-CSV geschrieben: {config.rep_csv_path} ({rep_counter.attempt_count} Versuche)")


# --------------------------------------------------------------------------
# Selbsttests ohne pytest, ohne Kamera/Modell (reine Funktionen pruefbar)
# --------------------------------------------------------------------------
def _run_selftests():
    from types import SimpleNamespace

    def lm(x, y, z=0.0, vis=1.0):
        return SimpleNamespace(x=x, y=y, z=z, visibility=vis)

    def feed(counter, knees, t0, dt, hip=170.0):
        """Fuettert eine Kniewinkel-Folge (None = kein Tracking) ein."""
        results, t = [], t0
        for k in knees:
            angles = {} if k is None else {"knee_left": k, "knee_right": k, "hip_left": hip, "hip_right": hip}
            r = counter.update(angles, t)
            if r is not None:
                results.append(r)
            t += dt
        return results, t

    REP = [170, 150, 120, 100, 85, 100, 130, 165]  # ein tiefer Rep, Ziel 90 (+10 Toleranz) erreicht

    # 1) 90-Grad-Winkel-Test
    assert abs(calculate_angle((0, 1, 0), (0, 0, 0), (1, 0, 0)) - 90.0) < 1e-6

    # 2) compute_joint_angles liefert None bei zu geringer visibility
    landmarks = [lm(0, 0)] * 33
    landmarks[LEFT_HIP] = lm(0, 1, vis=0.1)  # unter Schwelle
    landmarks[LEFT_KNEE] = lm(0, 0.5)
    landmarks[LEFT_ANKLE] = lm(0, 0)
    angles = compute_joint_angles(landmarks, min_visibility=0.5)
    assert angles["knee_left"] is None

    # 3) Guter Rep: genau 1 Ergebnis, Status GOOD, Zaehler 1
    counter = RepCounter(Config(max_gap_s=0.5))
    res, _ = feed(counter, REP, 0.0, 0.3)
    assert len(res) == 1 and res[0].status == "GOOD", res
    assert counter.good_count == 1 and counter.attempt_count == 1
    assert abs(res[0].t_down - 0.6) < 1e-6 and abs(res[0].t_up - 0.9) < 1e-6

    # 4) Flacher Rep (min 110 Grad): PARTIAL, zaehlt nicht als gut
    counter = RepCounter(Config(max_gap_s=0.5))
    res, _ = feed(counter, [170, 120, 110, 125, 165], 0.0, 0.3)
    assert len(res) == 1 and res[0].status == "PARTIAL" and counter.good_count == 0, res

    # 5) Minimale Bewegung (min 140 Grad): verworfen, kein Ergebnis
    counter = RepCounter(Config(max_gap_s=0.5))
    res, _ = feed(counter, [170, 140, 150, 165], 0.0, 0.3)
    assert res == [] and counter.attempt_count == 0

    # 6) Tracking-Luecke im Rep: INVALID, kein Fatigue-Einfluss
    counter = RepCounter(Config(max_gap_s=0.5))
    res, _ = feed(counter, [170, 120, 100, 85, None, None, 100, 130, 165], 0.0, 0.3)
    assert len(res) == 1 and res[0].status == "INVALID" and counter.good_count == 0, res
    assert res[0].omega_deg_s is None and counter.fatigue_active is False

    # 7) Abbruch bei langer Luecke: INVALID-Ergebnis, Zustand zurueck auf STANDING
    counter = RepCounter(Config(max_gap_s=0.5))
    res, _ = feed(counter, [170, 120, 100, 85] + [None] * 8, 0.0, 0.3)
    assert len(res) == 1 and res[0].status == "INVALID" and counter.state == "STANDING", res
    assert counter.tracking_ok is False

    # 8) Tempo-Formfehler: sehr schnelle Senkphase -> FORM
    counter = RepCounter(Config(max_gap_s=0.5))
    res, _ = feed(counter, REP, 0.0, 0.15)
    assert len(res) == 1 and res[0].status == "FORM" and "tempo" in res[0].reasons, res

    # 9) Velocity Loss: 3 schnelle Reps, dann langsamer werden
    counter = RepCounter(Config(max_gap_s=0.5))
    t = 0.0
    for _ in range(3):
        res, t = feed(counter, REP, t, 0.3)
    assert counter.fatigue_active is False and abs(counter.last_vl) < 1e-6
    res, t = feed(counter, REP, t, 0.5)  # 4. Rep langsam: VL ca. 40%, erst 1 Rep in Folge
    assert res[0].vl_pct > 30 and counter.fatigue_active is False, res[0]
    res, t = feed(counter, REP, t, 0.5)  # 5. Rep langsam: 2 in Folge -> Ermuedung aktiv
    assert counter.fatigue_active is True
    res, t = feed(counter, REP, t, 0.3)  # wieder schnell: VL 0 -> Ermuedung weg
    assert counter.fatigue_active is False
    # Formel-Kontrolle: omega = (120-85)/0.6 = 58.33, langsam (120-85)/1.0 = 35 -> VL = 40.0
    assert abs(counter.results[3].vl_pct - 40.0) < 1e-6, counter.results[3].vl_pct

    # 10) Satzwechsel setzt Zaehler und Ermuedung zurueck
    counter.reset_set()
    assert counter.good_count == 0 and counter.set_index == 2 and counter.last_vl is None

    # 11) Glaettung
    sm = AngleSmoother(0.5)
    assert sm.smooth({"knee_left": 100.0})["knee_left"] == 100.0
    assert abs(sm.smooth({"knee_left": 200.0})["knee_left"] - 150.0) < 1e-9
    assert sm.smooth({"knee_left": None})["knee_left"] is None
    assert sm.smooth({"knee_left": 200.0})["knee_left"] == 200.0  # Neustart nach Luecke

    # 12) Banner bleibt hold_s sichtbar
    cfg = Config()
    banner = FeedbackBanner(cfg.feedback_hold_s)
    good = RepResult(1, 1, "GOOD", [], 10.0, 85.0, 1.08, 0.6, 0.9, 58.0, 0.0, 2.0, 170.0, 1.0)
    banner.push(good, cfg, 10.0)
    assert banner.current(11.0) is not None and banner.current(11.6) is None

    # 13) CSV: Header- und Zeilenlaenge muessen uebereinstimmen, Landmark-
    # Werte muessen an der richtigen Spalte landen. lm_world() bildet die
    # echten MediaPipe-Weltkoordinaten-Objekte nach: Attribute heissen
    # .x/.y/.z/.visibility, NICHT .x_world/.y_world/.z_world - nur die
    # CSV-Spalten heissen so, zur Klarheit beim Auswerten.
    def lm_world(x, y, z, vis=1.0):
        return SimpleNamespace(x=x, y=y, z=z, visibility=vis)

    header = csv_header()
    world_landmarks = [lm_world(0, 0, 0, vis=0.9) for _ in range(33)]
    world_landmarks[LEFT_KNEE] = lm_world(1.5, 2.5, 3.5, vis=0.9)
    row = csv_row(12.34, 5, 2, "DESCENDING", {"knee_left": 95.0}, good, False, world_landmarks)
    assert len(header) == len(row), (len(header), len(row))
    knee_x_idx = header.index("left_knee_x_world")
    assert row[knee_x_idx] == 1.5 and row[knee_x_idx + 1] == 2.5 and row[knee_x_idx + 2] == 3.5
    assert row[header.index("rep_status")] == "GOOD"

    # CSV ohne erkannte Person (world_landmarks=None) darf nicht crashen
    row_none = csv_row(12.34, 5, 2, "STANDING", {}, None, False, None)
    assert len(row_none) == len(header)

    # Rep-CSV: Header und Zeile gleich lang
    assert len(rep_csv_header()) == len(rep_csv_row(good))

    print("Selbsttests bestanden.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/pose_landmarker_full.task")
    parser.add_argument("--camera", type=int, default=None, help="Kamera-Index direkt setzen (Default: automatisch)")
    parser.add_argument("--csv", default="data/output/pose_data.csv", help="Zielpfad fuer Frame-CSV")
    parser.add_argument("--rep-csv", default="data/output/rep_results.csv", help="Zielpfad fuer Rep-CSV")
    parser.add_argument("--no-csv", action="store_true", help="Beide CSV-Exporte deaktivieren")
    parser.add_argument("--target", type=float, default=90.0, help="Ziel-Kniewinkel in Grad")
    parser.add_argument("--vl-warn", type=float, default=20.0, help="Velocity-Loss-Schwelle in Prozent")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        _run_selftests()
    else:
        run(Config(
            model_path=args.model,
            camera_index=args.camera,
            csv_path=None if args.no_csv else args.csv,
            rep_csv_path=None if args.no_csv else args.rep_csv,
            target_knee_deg=args.target,
            vl_warn_pct=args.vl_warn,
        ))
