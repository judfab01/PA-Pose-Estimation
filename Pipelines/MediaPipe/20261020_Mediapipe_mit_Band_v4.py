"""
run_mediapipe.py
MediaPipe Pose Landmarker (Heavy, CPU) + rotes Theraband (ohne Marker) -> CSV, Meta-JSON, HUD.
Gleicher Aufbau und gleiches Ausgabeformat wie run_instanthmr.py. Eigenständig, keine weiteren Dateien nötig.

Start ohne Argumente (Run-Button der IDE): interaktive Abfrage von Quelle, Pfad,
Übung und Band. Letzte Eingaben werden gemerkt.

Beispiele (Kommandozeile):
  python run_mediapipe.py --video "/Users/fabrice/Desktop/PA Data/Squat frontal.MOV" --exercise squat
  python run_mediapipe.py --camera 0 --exercise seitzug
"""
import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
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


@dataclass
class Pose:
    detected: bool
    w: np.ndarray    # (14, 3) Weltkoordinaten in m
    img: np.ndarray  # (14, 2) Bild, normiert 0..1
    vis: np.ndarray  # (14,)  Sichtbarkeit 0..1


def empty_pose():
    return Pose(False, np.full((len(JOINTS), 3), NAN),
                np.full((len(JOINTS), 2), NAN), np.zeros(len(JOINTS)))


# ---------------------------------------------------------------- Winkel
def joint_angle(a, b, c):
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if not np.isfinite(n) or n < 1e-9:
        return NAN
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def compute_angles(pose):
    out = []
    for _, a, b, c in ANGLES:
        ia, ib, ic = JI[a], JI[b], JI[c]
        if pose.detected and min(pose.vis[ia], pose.vis[ib], pose.vis[ic]) >= VIS_MIN:
            out.append(joint_angle(pose.w[ia], pose.w[ib], pose.w[ic]))
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


# ---------------------------------------------------------------- Band (ohne Marker)
BAND_MIN_FRAC = 0.0003   # Mindestfläche der grössten roten Region (Anteil der Bildfläche)
BAND_MERGE_FRAC = 0.15   # weitere Regionen >= 15 % der grössten gehören zum Band (Band durch Hand geteilt)
BAND_WORK_W = 640        # Band-Erkennung auf Bild dieser Breite (Tempo)
LEN_PCT = (1.0, 99.0)    # Perzentile der Projektion auf die Hauptachse (robust gegen Ausreisser)


class BandTracker:
    """
    Rotes Band ohne Marker.

    Länge L = Ausdehnung aller roten Pixel entlang ihrer Hauptachse (PCA), zwischen
    dem 1. und 99. Perzentil. Mehrere rote Regionen (Band durch Hand/Körper
    unterbrochen) werden zusammen ausgewertet: gemessen wird die Ausdehnung vom einen
    zum anderen Ende, nicht die Summe der sichtbaren Teile.

    Dehnung % = (L / L0 - 1) * 100. L0 = Median von L in den ersten calib_frames
    Frames (Band entspannt) oder per Taste c.
    """

    def __init__(self, calib_frames=30, s_min=120, v_min=70):
        self.calib_frames = calib_frames
        self.s_min, self.v_min = s_min, v_min
        self.n = 0
        self.buf = []
        self.l0 = NAN
        self.ema = Ema(SMOOTH_ALPHA, 1)
        self.k5 = np.ones((5, 5), np.uint8)
        self.k9 = np.ones((9, 9), np.uint8)

    @staticmethod
    def none():
        """Detektionsergebnis 'kein Band'."""
        return dict(present=0, cx=NAN, cy=NAN, area=NAN, len_px=NAN, width_px=NAN,
                    angle=NAN, p0=None, p1=None, box=None)

    def detect(self, frame):
        h, w = frame.shape[:2]
        res = self.none()
        s = min(1.0, BAND_WORK_W / w)
        small = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame
        sh, sw = small.shape[:2]
        hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV)
        m = (cv2.inRange(hsv, (0, self.s_min, self.v_min), (10, 255, 255))
             | cv2.inRange(hsv, (170, self.s_min, self.v_min), (180, 255, 255)))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.k5)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.k9)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            return res
        areas = stats[1:, cv2.CC_STAT_AREA]
        amax = areas.max()
        if amax < BAND_MIN_FRAC * sw * sh:
            return res
        keep = 1 + np.where(areas >= BAND_MERGE_FRAC * amax)[0]
        ys, xs = np.nonzero(np.isin(labels, keep))
        pts = np.column_stack([xs, ys]).astype(np.float64)
        mean = pts.mean(axis=0)
        d = pts - mean
        evals, evecs = np.linalg.eigh(np.cov(d.T))
        axis = evecs[:, 1]                        # Hauptachse (grösster Eigenwert)
        t = d @ axis
        t0, t1 = np.percentile(t, LEN_PCT)
        length = float(t1 - t0)
        if length < 2:
            return res
        area = float(len(pts))
        p0, p1 = mean + axis * t0, mean + axis * t1
        box = cv2.boxPoints(cv2.minAreaRect(pts.astype(np.float32)))
        inv = 1.0 / s
        res.update(present=1, cx=mean[0] * inv, cy=mean[1] * inv, area=area * inv * inv,
                   len_px=length * inv, width_px=area / length * inv,
                   angle=float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0),
                   p0=tuple(p0 * inv), p1=tuple(p1 * inv), box=(box * inv).astype(np.int32))
        return res

    def stretch(self, det, calibrate_now=False):
        """Rückgabe: (len_smooth_px, stretch_raw_pct, stretch_pct)"""
        self.n += 1
        L = det["len_px"] if det["present"] else NAN
        Ls = self.ema([L])[0]
        if np.isfinite(Ls) and self.n <= self.calib_frames:
            self.buf.append(Ls)
        if self.calib_frames and self.n == self.calib_frames and len(self.buf) >= 5:
            self.l0 = float(np.median(self.buf))
        if calibrate_now and np.isfinite(Ls):
            self.l0 = float(Ls)
        ok = np.isfinite(self.l0) and self.l0 > 0
        raw = (L / self.l0 - 1.0) * 100.0 if ok and np.isfinite(L) else NAN
        sm = (Ls / self.l0 - 1.0) * 100.0 if ok and np.isfinite(Ls) else NAN
        return Ls, raw, sm


# ---------------------------------------------------------------- Ausgabe
def csv_header():
    h = ["frame", "timestamp_ms", "pose_detected", "exercise", "subject_id",
         "rep_id", "model", "infer_ms"]
    for j in JOINTS:
        h += [f"{j}_{s}" for s in ("wx", "wy", "wz", "ix", "iy", "vis")]
    h += [f"angle_{a[0]}_raw" for a in ANGLES]
    h += [f"angle_{a[0]}" for a in ANGLES]
    h += ["band_present", "band_cx", "band_cy", "band_area", "band_len_px",
          "band_len_smooth_px", "band_width_px", "band_angle_deg", "band_ref_len_px",
          "stretch_raw_pct", "stretch_pct"]
    return h


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{v:.5f}"
    return str(v)


class Recorder:
    def __init__(self, outdir, stem):
        outdir.mkdir(parents=True, exist_ok=True)
        self.csv_path = outdir / f"{stem}.csv"
        self.meta_path = outdir / f"{stem}_meta.json"
        self.f = open(self.csv_path, "w", newline="")
        self.header = csv_header()
        self.f.write(",".join(self.header) + "\n")
        self.n = 0

    def write(self, row):
        assert len(row) == len(self.header), (len(row), len(self.header))
        self.f.write(",".join(_fmt(v) for v in row) + "\n")
        self.n += 1

    def close(self, meta):
        self.f.close()
        meta["frames_written"] = self.n
        meta["csv_columns"] = len(self.header)
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)


# ---------------------------------------------------------------- HUD
GREEN, GREY, YELLOW, CYAN, MAGENTA = (0, 220, 0), (140, 140, 140), (0, 255, 255), (255, 255, 0), (255, 0, 255)


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

    if band["present"]:
        cv2.polylines(frame, [band["box"]], True, CYAN, 2, cv2.LINE_AA)
        p0 = tuple(int(round(v)) for v in band["p0"])
        p1 = tuple(int(round(v)) for v in band["p1"])
        cv2.line(frame, p0, p1, MAGENTA, 2, cv2.LINE_AA)
        cv2.circle(frame, p0, 7, MAGENTA, 2)
        cv2.circle(frame, p1, 7, MAGENTA, 2)

    if np.isfinite(pct):
        _txt(frame, f"Dehnung {pct:+.1f} %", (12, 44), 1.1, YELLOW, 2)
        _txt(frame, f"L {len_s:.0f} px  L0 {l0:.0f} px", (12, 68), 0.5)
    elif band["present"]:
        _txt(frame, "Dehnung: kalibrieren (c)", (12, 44), 0.8, YELLOW, 2)
    else:
        _txt(frame, "Dehnung: kein Band", (12, 44), 0.8, GREY, 2)
    _txt(frame, f"{model_name}  frame {info['frame']}  rep {info['rep']}  "
                f"{info['infer_ms']:.0f} ms  {info['fps']:.0f} fps", (12, h - 14), 0.55)
    _txt(frame, "Space=rep  c=kalibrieren  q=Ende", (12, h - 36), 0.5)


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
    args.exercise = ask("Übung (Label für die CSV, z.B. squat, seitzug)", cfg.get("exercise", ""))
    args.subject = ask("Probanden-ID (Enter = leer)", cfg.get("subject", ""))
    band = ask_yes_no("Theraband im Video", cfg.get("band", False))
    args.no_band = not band
    if args.camera is None:
        args.no_show = not ask_yes_no("Fenster mit HUD anzeigen (Beenden mit q)", True)
        args.max_frames = ask_int("Nur die ersten N Frames verarbeiten (0 = alle)", 0)
    else:
        args.no_show, args.max_frames = False, 0
        print("Live-Kamera: Beenden mit q im Fenster.")
    if band:
        print("Band: erste 30 Frames entspannt zeigen oder im Fenster c drücken.")
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
                   help="Erste N Frames (Band entspannt) als Referenzlänge L0, 0 = nur Taste c")
    p.add_argument("--red-s-min", type=int, default=120, help="HSV Sättigung min für Rot")
    p.add_argument("--red-v-min", type=int, default=70, help="HSV Helligkeit min für Rot")
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

    rep, frame_idx, last_ts, calib = 0, 0, -1, False
    fps_s, infer_sum = 30.0, 0.0
    t0 = time.perf_counter()
    try:
        while True:
            t_loop = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            ts = int(round(frame_idx * 1000 / src_fps)) if is_video else int((t_loop - t0) * 1000)
            ts = max(ts, last_ts + 1)
            last_ts = ts

            t1 = time.perf_counter()
            pose = infer(frame, ts)
            infer_ms = (time.perf_counter() - t1) * 1000
            infer_sum += infer_ms

            raw = compute_angles(pose)
            sm = ema(raw)
            b = BandTracker.none() if args.no_band else band.detect(frame)
            len_s, pct_raw, pct = band.stretch(b, calib)
            calib = False

            row = [frame_idx, ts, int(pose.detected), args.exercise, args.subject,
                   rep, model_name, infer_ms]
            for i in range(len(JOINTS)):
                row += [*pose.w[i], *pose.img[i], pose.vis[i]]
            row += raw + sm
            row += [b["present"], b["cx"], b["cy"], b["area"], b["len_px"], len_s,
                    b["width_px"], b["angle"], band.l0, pct_raw, pct]
            rec.write(row)

            if not args.no_show:
                draw_hud(frame, model_name, pose, sm, b, len_s, band.l0, pct,
                         dict(frame=frame_idx, rep=rep, infer_ms=infer_ms, fps=fps_s))
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
        meta = dict(
            model=model_name, source=args.video or f"camera {args.camera}",
            source_fps=src_fps, width=W, height=H, started=t_start.isoformat(),
            finished=datetime.now().isoformat(), mean_infer_ms=infer_sum / max(frame_idx, 1),
            vis_min=VIS_MIN, smooth_alpha=SMOOTH_ALPHA, calib_frames=args.calib_frames,
            no_band=args.no_band, band_ref_len_px=band.l0, red_s_min=args.red_s_min, red_v_min=args.red_v_min,
            band_len_percentiles=LEN_PCT, band_work_width=BAND_WORK_W,
            joints=JOINTS, angles=[a[0] for a in ANGLES],
            python=platform.python_version(), opencv=cv2.__version__,
        )
        meta.update(meta_extra or {})
        rec.close(meta)
        print(f"CSV : {rec.csv_path}\nMeta: {rec.meta_path}")
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
        return Pose(True, w, img, vis)

    meta = dict(model_file=path.name, delegate="CPU", mediapipe=mp.__version__, mp_idx=MP_IDX)
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
