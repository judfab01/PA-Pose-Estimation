"""
run_instanthmr.py
InstantHMR (RF-DETR + ONNX) + rotes Theraband (ohne Marker) -> CSV, Meta-JSON, HUD.
Gleicher Aufbau und gleiches Ausgabeformat wie run_mediapipe.py. Eigenständig, keine weiteren Dateien nötig.
Muss im Repo-Ordner `instanthmr/` liegen (neben demo.py), conda env `instanthmr` aktiv.

Beispiele:
  python run_instanthmr.py --video "/Users/fabrice/Desktop/PA Data/Squat frontal.MOV" --device cpu --exercise squat
  python run_instanthmr.py --camera 0 --device coreml --exercise seitzug

VERIFIZIERT (gegen Repo-Quellcode gelesen):
  - PosePipeline-Konstruktor und alle Argumente: instanthmr/pipeline.py
  - predict() -> FrameResult.persons: list[HMRPrediction]: instanthmr/pipeline.py
  - 3D: HMRPrediction.joints_3d_local (70, 3), m, Y-down, rig-lokal;
    2D: HMRPrediction.joints_2d (70, 2) in Vollbild-Pixeln: instanthmr/inference.py
  - Joint-Indizes MHR_IDX: instanthmr/skeleton.py (JOINT_NAMES, MHR70)
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

    def detect(self, frame):
        h, w = frame.shape[:2]
        res = dict(present=0, cx=NAN, cy=NAN, area=NAN, len_px=NAN, width_px=NAN,
                   angle=NAN, p0=None, p1=None, box=None)
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


# ---------------------------------------------------------------- Hauptschleife
def add_common_args(p):
    s = p.add_mutually_exclusive_group(required=True)
    s.add_argument("--video", help="Pfad zu .mov/.mp4")
    s.add_argument("--camera", type=int, help="Kamera-Index (0, 1, ...)")
    p.add_argument("--out", default="output", help="Ausgabeordner")
    p.add_argument("--exercise", default="", help="Label für CSV-Spalte exercise")
    p.add_argument("--subject", default="", help="Label für CSV-Spalte subject_id")
    p.add_argument("--calib-frames", type=int, default=30,
                   help="Erste N Frames (Band entspannt) als Referenzlänge L0, 0 = nur Taste c")
    p.add_argument("--red-s-min", type=int, default=120, help="HSV Sättigung min für Rot")
    p.add_argument("--red-v-min", type=int, default=70, help="HSV Helligkeit min für Rot")
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
    rec = Recorder(Path(args.out), stem)
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
            b = band.detect(frame)
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
            band_ref_len_px=band.l0, red_s_min=args.red_s_min, red_v_min=args.red_v_min,
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
import importlib
import inspect

MODEL_NAME = "instanthmr"

# Gemeinsames Gelenk -> Index im 70-Joint-Array.
# Verifiziert gegen instanthmr/skeleton.py, JOINT_NAMES (MHR70).
MHR_IDX = {
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_hip": 9, "r_hip": 10, "l_knee": 11, "r_knee": 12,
    "l_ankle": 13, "r_ankle": 14,
    "l_foot": 15, "r_foot": 18,     # grosse Zehe links / rechts
    "l_wrist": 62, "r_wrist": 41,
}
IDX = [MHR_IDX[j] for j in JOINTS]


def add_model_args(ap):
    ap.add_argument("--model", default="models/instanthmr.onnx")
    ap.add_argument("--device", default="cpu", help="cpu | coreml")
    ap.add_argument("--detector-variant", default="nano")
    ap.add_argument("--detector-stride", type=int, default=3)
    ap.add_argument("--focal", type=float, default=None, help="Brennweite in px")


def _get(o, key):
    return o.get(key) if isinstance(o, dict) else getattr(o, key, None)


def load_model(args):
    """Rückgabe: (infer(frame_bgr, ts_ms) -> Pose, meta_dict, close())"""
    cls = None
    for mod in ("instanthmr", "instanthmr.pipeline", "instanthmr.inference",
                "instanthmr.pose_pipeline"):
        try:
            cls = getattr(importlib.import_module(mod), "PosePipeline", None)
        except ImportError:
            continue
        if cls:
            break
    if cls is None:
        raise SystemExit("PosePipeline nicht gefunden. grep -n PosePipeline demo.py")
    wanted = dict(onnx_path=args.model, device=args.device,
                  detector_variant=args.detector_variant,
                  detector_stride=args.detector_stride, max_persons=1, focal=args.focal)
    params = inspect.signature(cls.__init__).parameters
    has_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
    used = {k: v for k, v in wanted.items() if v is not None and (has_kw or k in params)}
    dropped = [k for k, v in wanted.items() if v is not None and k not in used]
    print("PosePipeline Argumente:", used, "| nicht unterstützt:", dropped)
    pipe = cls(**used)
    state = {"shape_logged": False}

    def infer(frame_bgr, ts_ms):
        h, w = frame_bgr.shape[:2]
        res = pipe.predict(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not isinstance(res, (list, tuple)):
            persons = _get(res, "persons")
            res = persons if persons is not None else [res]
        if len(res) == 0:
            return empty_pose()
        p = res[0]                        # HMRPrediction
        j3, j2 = _get(p, "joints_3d_local"), _get(p, "joints_2d")
        if j3 is None or j2 is None:
            raise RuntimeError("joints_3d_local/joints_2d fehlen in HMRPrediction.")
        j3, j2 = np.asarray(j3, float), np.asarray(j2, float)
        if not state["shape_logged"]:
            print("joints_3d_local", j3.shape, "joints_2d", j2.shape,
                  "| joints_2d max", float(np.nanmax(j2)))
            state["shape_logged"] = True
        img = j2[IDX] / np.array([w, h])  # Vollbild-Pixel -> normiert 0..1
        return Pose(True, j3[IDX].copy(), img, np.ones(len(JOINTS)))  # keine Sichtbarkeit vom Modell

    meta = dict(device=args.device, detector_variant=args.detector_variant,
                detector_stride=args.detector_stride, focal=args.focal,
                pipeline_kwargs_used=used, pipeline_kwargs_dropped=dropped,
                mhr_idx=MHR_IDX, onnx=Path(args.model).name,
                joints_3d_source="joints_3d_local", joints_2d_source="joints_2d (Vollbild-Pixel)",
                vis_note="InstantHMR liefert keine Sichtbarkeit, vis = 1.0 fix")
    return infer, meta, (lambda: None)
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
    args = ap.parse_args()
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
