"""
20261027_Analyse_Vergleich_v2.py
Vergleicht die CSV-Ausgaben von run_mediapipe und run_instanthmr (Pipelines v6; v4/v5 eingeschraenkt) und erzeugt:
  01_verlauf.png           Zeitverlaeufe beider Modelle (Winkel, Spreizung, Band) mit Wiederholungen
  02_uebereinstimmung.png  Streu- und Bland-Altman-Diagramme je Winkel
  03_band.png              Banddehnung je Modell, Bandlaenge gegen PCA-Referenz und Handabstand, Versatz (nur mit Band)
  04_qualitaet.png         Segmentabweichung, Jitter, Latenz
  05_korrektur.png         Anteil korrigierter Frames je Segment, Winkelwirkung je Winkel      (NEU, braucht v6)
  06_korrektur_zeit.png    Wann welches Segment korrigiert wurde, Winkeldifferenz im Zeitverlauf (NEU, braucht v6)
  report.md                Kennzahlen, Wiederholungen, Segmentkorrektur, Hinweise
  metrics.csv              Kennzahlen dieses Laufs
  korrekturen.csv          Alle Korrektur-Episoden (Modell, Segment, Start, Dauer, Abweichung, Winkelwirkung)  (NEU)
  winkelwirkung.csv        Wirkung der Tiefenkorrektur je Winkel und Modell                                   (NEU)
  vergleich.mp4            Original mit beiden Skeletten und Band (optional)
und haengt eine Zeile pro Modell an data/analysis/all_runs_v2.csv an (Vergleich ueber alle Videos).

Aenderungen gegenueber v1:
  - Liest das v6-Format (angle_*_cor_raw, grip_*, band_len_pca_px, band_red_frac, <stem>_segdev.csv).
  - Neue Analyse: wann welche Segmente tiefenkorrigiert wurden (aus <stem>_segdev.csv) und wie stark sich
    dadurch die Gelenkwinkel gegenueber den Modell-Rohwinkeln aendern (Delta = angle_*_cor_raw - angle_*_raw).
  - --winkel raw|cor: Winkelquelle fuer Verlauf, Uebereinstimmung, Jitter, Wiederholungen
    (raw = Modell pur, cor = nach Tiefenkorrektur). Die Korrekturanalyse nutzt immer beide.
  - Band ist ab v6 modellabhaengig (Laenge = Griffpunktabstand): Banddarstellung und Validierung je Modell,
    Versatz gegen die PCA-Referenz (Entscheidungsgrundlage: konstanter oder haltungsabhaengiger Versatz).
  - Neues Profil reverse_butterfly (Reverse Butterfly mit Resistenzband, Hand zu Hand).
  - Segment-CSVs (*_segdev.csv) werden bei der Auswahl uebersprungen. Sammeldatei heisst all_runs_v2.csv.

Start ohne Argumente (Run-Button der IDE): Abfrage der beiden CSVs, der Uebung und des Videos.
Kommandozeile:
  python 20261027_Analyse_Vergleich_v2.py --mp <mediapipe.csv> --ih <instanthmr.csv> --exercise reverse_butterfly

Uebungen: squat, seitzug, reverse_butterfly, reverse_fly_jacks, generic
Abhaengigkeiten: numpy, pandas, matplotlib (Vergleichsvideo zusaetzlich opencv-python)

Hinweis: Ohne Referenzmessung (Goniometer, gemessene Segmentlaengen) zeigt der Vergleich
Uebereinstimmung und Stabilitaet der Modelle, nicht deren Genauigkeit. Die Winkeldifferenz der Korrektur
ist die Abweichung zur Modell-Rohschaetzung, kein Mass fuer die Naehe zur Wahrheit. Einzige Referenz im
Code: die gemessenen Segmentlaengen (SEGMENT_LENGTHS).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- Konstanten
JOINTS = ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
          "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle", "l_foot", "r_foot"]
BONES = [("l_shoulder", "r_shoulder"), ("l_hip", "r_hip"), ("l_shoulder", "l_hip"),
         ("r_shoulder", "r_hip"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
         ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"), ("l_hip", "l_knee"),
         ("l_knee", "l_ankle"), ("l_ankle", "l_foot"), ("r_hip", "r_knee"),
         ("r_knee", "r_ankle"), ("r_ankle", "r_foot")]
ANGLE_NAMES = ["elbow_l", "elbow_r", "shoulder_l", "shoulder_r", "hip_l", "hip_r",
               "knee_l", "knee_r", "ankle_l", "ankle_r"]
# Winkel bei B zwischen B->A und B->C, identisch zu ANGLES in den Pipelines
ANGLE_DEFS = [
    ("elbow_l", "l_shoulder", "l_elbow", "l_wrist"), ("elbow_r", "r_shoulder", "r_elbow", "r_wrist"),
    ("shoulder_l", "l_hip", "l_shoulder", "l_elbow"), ("shoulder_r", "r_hip", "r_shoulder", "r_elbow"),
    ("hip_l", "l_shoulder", "l_hip", "l_knee"), ("hip_r", "r_shoulder", "r_hip", "r_knee"),
    ("knee_l", "l_hip", "l_knee", "l_ankle"), ("knee_r", "r_hip", "r_knee", "r_ankle"),
    ("ankle_l", "l_knee", "l_ankle", "l_foot"), ("ankle_r", "r_knee", "r_ankle", "r_foot"),
]

# Gemessene Segmentlaengen (m) von Fabrice, identisch zu den Pipelines v5/v6
SEGMENTS = [  # (Anzeigename, Gelenk A, Gelenk B, Laenge); die ersten 12 werden von den Pipelines korrigiert
    ("Rumpf l", "l_hip", "l_shoulder", 0.480), ("Rumpf r", "r_hip", "r_shoulder", 0.480),
    ("Oberarm l", "l_shoulder", "l_elbow", 0.315), ("Oberarm r", "r_shoulder", "r_elbow", 0.315),
    ("Unterarm l", "l_elbow", "l_wrist", 0.270), ("Unterarm r", "r_elbow", "r_wrist", 0.270),
    ("Oberschenkel l", "l_hip", "l_knee", 0.475), ("Oberschenkel r", "r_hip", "r_knee", 0.475),
    ("Unterschenkel l", "l_knee", "l_ankle", 0.415), ("Unterschenkel r", "r_knee", "r_ankle", 0.415),
    ("Fuss l", "l_ankle", "l_foot", 0.200), ("Fuss r", "r_ankle", "r_foot", 0.200),
    ("Hueftbreite", "l_hip", "r_hip", 0.310), ("Schulterbreite", "l_shoulder", "r_shoulder", 0.400),
]
CORR_SEGS = SEGMENTS[:12]
SEG_COL = {n: (a, b) for n, a, b, _ in SEGMENTS}


def _seg_for(j1, j2):
    for name, a, b, _ in SEGMENTS:
        if {a, b} == {j1, j2}:
            return name


# Winkel -> die zwei angrenzenden Segmente (nur deren Richtung bestimmt den Winkel)
ANGLE_SEGS = {n: (_seg_for(b, a), _seg_for(b, c)) for n, a, b, c in ANGLE_DEFS}

VIS_MIN = 0.5
SPIKE_DEG = 20.0          # Winkelsprung pro Frame, ab dem ein Frame als Ausreisser zaehlt
COLORS = {"mediapipe": "#1b9e77", "instanthmr": "#d95f02"}
BGR = {"mediapipe": (0, 200, 0), "instanthmr": (0, 140, 255)}
NICE = {"mediapipe": "MediaPipe", "instanthmr": "InstantHMR"}
ANGLE_MODE = "raw"        # raw = Modell pur (angle_*_raw), cor = nach Tiefenkorrektur (angle_*_cor_raw)
ANGLE_LABEL = {"raw": "rohe Winkel (Modell)", "cor": "tiefenkorrigierte Winkel"}

# Uebungsprofile: Zeitverlauf-Panels (Titel, Spalte, Einheit) und Signal fuer die Wiederholungserkennung
PROFILES = {
    "squat": dict(
        titel="Kniebeuge",
        panels=[("Knie links", "angle_knee_l_raw", "°"), ("Knie rechts", "angle_knee_r_raw", "°"),
                ("Hüfte links", "angle_hip_l_raw", "°"), ("Hüfte rechts", "angle_hip_r_raw", "°"),
                ("Knie links minus rechts", "knee_asym", "°")],
        angles=["knee_l", "knee_r", "hip_l", "hip_r"], rep=("knee_mean", "min", 15.0, "Kniewinkel Mittel (°)")),
    "seitzug": dict(
        titel="Seitzug mit Band",
        panels=[("Ellbogen links", "angle_elbow_l_raw", "°"), ("Ellbogen rechts", "angle_elbow_r_raw", "°"),
                ("Schulter links", "angle_shoulder_l_raw", "°"), ("Schulter rechts", "angle_shoulder_r_raw", "°"),
                ("Banddehnung", "stretch_pct", "%")],
        angles=["elbow_l", "elbow_r", "shoulder_l", "shoulder_r"], rep=("elbow_mean", "min", 15.0, "Ellbogenwinkel Mittel (°)")),
    "reverse_butterfly": dict(
        titel="Reverse Butterfly mit Resistenzband",
        panels=[("Handabstand / Schulterbreite", "wrist_spread", ""),
                ("Ellbogen links", "angle_elbow_l_raw", "°"), ("Ellbogen rechts", "angle_elbow_r_raw", "°"),
                ("Schulter links", "angle_shoulder_l_raw", "°"), ("Schulter rechts", "angle_shoulder_r_raw", "°"),
                ("Banddehnung", "stretch_pct", "%")],
        angles=["elbow_l", "elbow_r", "shoulder_l", "shoulder_r"], rep=("wrist_spread", "max", 0.3, "Handabstand / Schulterbreite")),
    "reverse_fly_jacks": dict(
        titel="Band Reverse Fly Jacks",
        panels=[("Handabstand / Schulterbreite", "wrist_spread", ""), ("Fussabstand / Schulterbreite", "ankle_spread", ""),
                ("Ellbogen links", "angle_elbow_l_raw", "°"), ("Ellbogen rechts", "angle_elbow_r_raw", "°"),
                ("Banddehnung", "stretch_pct", "%")],
        angles=["elbow_l", "elbow_r", "shoulder_l", "shoulder_r"], rep=("wrist_spread", "max", 0.3, "Handabstand / Schulterbreite")),
    "generic": dict(titel="Allgemein", panels=None, angles=None, rep=None),
}

# ---------------------------------------------------------------- Hilfsfunktionen (Abfrage)
CFG_PATH = Path.home() / ".pose_pipeline.json"


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
    for p in Path(__file__).resolve().parents:
        if (p / ".git").exists() or (p / "data").is_dir():
            return p
    return Path.cwd()


def clean_path(s):
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


# ---------------------------------------------------------------- Daten laden
class Run:
    def __init__(self, path):
        self.path = Path(path)
        self.df = pd.read_csv(self.path)
        mpath = self.path.with_name(self.path.stem + "_meta.json")
        self.meta = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
        self.model = (str(self.df["model"].iloc[0]) if "model" in self.df.columns and len(self.df)
                      else self.path.stem.split("_")[0]).lower()
        self.W = int(self.meta.get("width") or 0) or None
        self.H = int(self.meta.get("height") or 0) or None
        if not self.W or not self.H:
            print(f"Warnung: keine Aufloesung in der Meta-Datei von {self.path.name}, nehme normierte Einheiten.")
            self.W = self.H = 1
        ex = self.df["exercise"].dropna() if "exercise" in self.df.columns else pd.Series([], dtype=str)
        self.exercise = str(ex.iloc[0]) if len(ex) else ""
        self.source = self.meta.get("source", "")
        self.has_raw = "l_hip_wx_raw" in self.df.columns          # v5/v6-Format (korrigiert + roh)
        self.has_cor = all(f"angle_{a}_cor_raw" in self.df.columns for a in ANGLE_NAMES)   # v6-Format
        self.seg = None
        spath = self.path.with_name(self.path.stem + "_segdev.csv")
        if spath.exists():
            seg = pd.read_csv(spath)
            if len(seg) == len(self.df):
                self.seg = seg
            else:
                print(f"Warnung: {spath.name} hat {len(seg)} Zeilen, CSV {len(self.df)}; Korrekturanalyse entfaellt.")
        self.D = self._derive()

    def xyz(self, j):
        """Rohe 3D-Koordinaten (v4: _wx ist roh, v5/v6: _wx_raw ist roh)."""
        s = "_raw" if self.has_raw else ""
        return self.df[[f"{j}_wx{s}", f"{j}_wy{s}", f"{j}_wz{s}"]].to_numpy(float)

    def px(self, j):
        return np.column_stack([self.df[f"{j}_ix"] * self.W, self.df[f"{j}_iy"] * self.H])

    def seg_on(self, segname):
        """Pro Frame: 1 = Segment tiefenkorrigiert, 0 = nicht, NaN = keine Pose. None ohne segdev-Datei."""
        if self.seg is None:
            return None
        a, b = SEG_COL[segname]
        col = f"depth_on_{a}-{b}"
        return self.seg[col].to_numpy(float) if col in self.seg.columns else None

    def seg_dev(self, segname):
        if self.seg is None:
            return None
        a, b = SEG_COL[segname]
        col = f"dev_pct_{a}-{b}"
        return self.seg[col].to_numpy(float) if col in self.seg.columns else None

    def _derive(self):
        df = self.df
        D = pd.DataFrame({"frame": df["frame"].to_numpy(), "t": df["timestamp_ms"].to_numpy() / 1000.0})
        mode = ANGLE_MODE
        if mode == "cor" and not self.has_cor:
            print(f"Warnung: {self.path.name} hat keine angle_*_cor_raw (kein v6-Format), nehme rohe Winkel.")
            mode = "raw"
        self.angle_mode = mode
        for a in ANGLE_NAMES:
            orig = df[f"angle_{a}_raw"].to_numpy(float)
            cor = df[f"angle_{a}_cor_raw"].to_numpy(float) if self.has_cor else np.full(len(df), np.nan)
            D[f"angle_{a}_orig"], D[f"angle_{a}_cor"] = orig, cor
            D[f"angle_{a}_raw"] = cor if mode == "cor" else orig        # Auswahl, von allen Plots genutzt
            D[f"d_{a}"] = cor - orig                                    # Wirkung der Tiefenkorrektur
        sw = np.nanmedian(np.linalg.norm(self.px("l_shoulder") - self.px("r_shoulder"), axis=1))
        self.shoulder_px = float(sw)
        D["wrist_dist_px"] = np.linalg.norm(self.px("l_wrist") - self.px("r_wrist"), axis=1)
        D["wrist_spread"] = D["wrist_dist_px"] / sw
        D["ankle_spread"] = np.linalg.norm(self.px("l_ankle") - self.px("r_ankle"), axis=1) / sw
        D["knee_mean"] = D[["angle_knee_l_raw", "angle_knee_r_raw"]].mean(axis=1)
        D["elbow_mean"] = D[["angle_elbow_l_raw", "angle_elbow_r_raw"]].mean(axis=1)
        D["knee_asym"] = D["angle_knee_l_raw"] - D["angle_knee_r_raw"]
        for c in ("band_present", "band_len_px", "band_cx", "band_cy", "band_angle_deg", "stretch_pct",
                  "stretch_raw_pct", "band_len_pca_px", "band_red_frac", "grip_l_ix", "grip_l_iy",
                  "grip_r_ix", "grip_r_iy"):
            D[c] = df[c].to_numpy(float) if c in df.columns else np.nan
        return D

    @property
    def label(self):
        return NICE.get(self.model, self.model)

    @property
    def color(self):
        return COLORS.get(self.model, "#555555")

    @property
    def has_band(self):
        return bool(np.nanmean(self.D["band_present"].to_numpy(float)) > 0.2) and not self.meta.get("no_band", False)

    @property
    def has_grip(self):
        return bool(np.isfinite(self.D["grip_l_ix"].to_numpy(float)).any())


def smooth(s, k=5):
    """Median-Filter gegen Einzelframe-Ausreisser."""
    return pd.Series(s).rolling(k, center=True, min_periods=1).median().to_numpy()


# ---------------------------------------------------------------- Kennzahlen
def detect_reps(sig, mode, min_range, min_gap=10):
    """Hysterese-Erkennung: Wiederholung = Auslenkung ueber 65 % der Spanne, Ruecksetzen unter 35 %."""
    x = pd.Series(sig, dtype=float).interpolate(limit_direction="both").to_numpy()
    if not np.isfinite(x).any():
        return []
    lo, hi = np.nanpercentile(x, [5, 95])
    if hi - lo < min_range:
        return []
    z = (x - lo) / (hi - lo)
    if mode == "min":
        z = 1.0 - z
    reps, inside, start = [], False, 0
    for i, v in enumerate(z):
        if not inside and v > 0.65:
            inside, start = True, i
        elif inside and v < 0.35:
            reps.append(start + int(np.argmax(z[start:i])))
            inside = False
    out = []
    for p in reps:
        if not out or p - out[-1] >= min_gap:
            out.append(p)
    return out


def match_reps(a, b, tol=20):
    pairs, used = [], set()
    for i, pa in enumerate(a):
        best = min(((abs(pa - pb), j) for j, pb in enumerate(b) if j not in used), default=None)
        if best and best[0] <= tol:
            pairs.append((i, best[1]))
            used.add(best[1])
    return pairs


def agreement(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return None
    d = (b - a)[ok]                       # b minus a
    return dict(n=int(ok.sum()), bias=float(d.mean()), mae=float(np.abs(d).mean()),
                rmse=float(np.sqrt((d ** 2).mean())), corr=float(np.corrcoef(a[ok], b[ok])[0, 1]),
                sd=float(d.std(ddof=1)), mean_ab=((a + b) / 2)[ok], diff=d, a=a[ok], b=b[ok])


def jitter(angle_series):
    x = np.asarray(angle_series, float)
    d2 = np.diff(x, n=2)
    d1 = np.diff(x)
    return (float(np.sqrt(np.nanmean(d2 ** 2))) if np.isfinite(d2).any() else np.nan,
            int(np.nansum(np.abs(d1) > SPIKE_DEG)))


def segment_deviation(run):
    out = {}
    for name, a, b, L in SEGMENTS:
        d = np.linalg.norm(run.xyz(a) - run.xyz(b), axis=1)
        ok = (run.df[f"{a}_vis"].to_numpy(float) >= VIS_MIN) & (run.df[f"{b}_vis"].to_numpy(float) >= VIS_MIN) & np.isfinite(d)
        if ok.sum() == 0:
            out[name] = (np.nan, np.nan, L)
        else:
            m = float(np.median(d[ok]))
            out[name] = (m, (m - L) / L * 100.0, L)
    return out


def sync_lag(a, b, max_lag=15):
    """Verschiebung (Frames), bei der b am besten zu a passt. Positiv = b folgt a."""
    a = pd.Series(a).interpolate(limit_direction="both").to_numpy()
    b = pd.Series(b).interpolate(limit_direction="both").to_numpy()
    if np.nanstd(a) < 1e-9 or np.nanstd(b) < 1e-9:
        return None
    a = (a - np.nanmean(a)) / np.nanstd(a)
    b = (b - np.nanmean(b)) / np.nanstd(b)
    best = None
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[:len(a) - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[:len(b) + lag])
        if len(x) < 10:
            continue
        c = float(np.mean(x * y))
        if best is None or c > best[1]:
            best = (lag, c)
    return best


def run_stats(run, prof, ref_frames):
    D = run.D
    st = {"model": run.model, "frames": int(len(D)), "dauer_s": float(D["t"].iloc[-1]) if len(D) else 0.0}
    st["erkannt_pct"] = float(run.df["pose_detected"].mean() * 100.0)
    ims = run.df["infer_ms"].to_numpy(float)
    st["latenz_mean_ms"], st["latenz_median_ms"], st["latenz_p95_ms"] = float(np.nanmean(ims)), float(np.nanmedian(ims)), float(np.nanpercentile(ims, 95))
    vis = run.df[[f"{j}_vis" for j in JOINTS]].to_numpy(float)
    st["hat_sichtbarkeit"] = bool(np.nanstd(vis) > 1e-9)
    st["vis_unter_schwelle_pct"] = float(np.nanmean(vis < VIS_MIN) * 100.0) if st["hat_sichtbarkeit"] else np.nan
    st["detector_stride"] = run.meta.get("detector_stride")
    st["fresh_pct"] = float(run.df["fresh"].mean() * 100.0) if "fresh" in run.df.columns else np.nan
    rel = prof["angles"] or ANGLE_NAMES
    jit, spk = [], 0
    jit_by = {}
    for a in ANGLE_NAMES:
        j, s = jitter(D[f"angle_{a}_raw"].to_numpy())
        jit_by[a] = (j, s)
        if a in rel:
            jit.append(j)
            spk += s
    st["jitter_by_angle"] = jit_by
    st["jitter_mean"] = float(np.nanmean(jit)) if jit else np.nan
    st["spikes"] = spk
    if run.has_cor:
        st["jitter_orig_mean"] = float(np.nanmean([jitter(D[f"angle_{a}_orig"].to_numpy())[0] for a in rel]))
        st["jitter_cor_mean"] = float(np.nanmean([jitter(D[f"angle_{a}_cor"].to_numpy())[0] for a in rel]))
    else:
        st["jitter_orig_mean"] = st["jitter_cor_mean"] = np.nan
    seg = segment_deviation(run)
    st["segmente"] = seg
    st["seg_abw_mean_pct"] = float(np.nanmean([abs(v[1]) for v in seg.values()]))
    rs = prof["rep"]
    st["rep_idx"], st["rep_signal"] = [], None
    if rs:
        col, mode, rng, _ = rs
        sig = smooth(D[col].to_numpy())
        st["rep_signal"] = sig
        st["rep_idx"] = detect_reps(sig, mode, rng)
    return st


# ---------------------------------------------------------------- Segmentkorrektur und Winkelwirkung
def _runs(mask):
    """Zusammenhaengende True-Bereiche als (Start, Ende inklusive)."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def correction_analysis(run, prof):
    """Wann welche Segmente tiefenkorrigiert wurden und wie sich dadurch die Winkel aendern."""
    if run.seg is None or not run.has_cor:
        return None
    D = run.D
    t, frames = D["t"].to_numpy(), D["frame"].to_numpy()
    on = {}
    for name, a, b, L in CORR_SEGS:
        x = run.seg_on(name)
        if x is None:
            return None
        on[name] = x
    per_seg_meta = (run.meta.get("depth_correction") or {}).get("per_segment", {})
    seg_rows, episodes = {}, []
    affected = {n: [a for a, (s1, s2) in ANGLE_SEGS.items() if n in (s1, s2)] for n, *_ in CORR_SEGS}
    for name, a, b, L in CORR_SEGS:
        x = on[name]
        valid = np.isfinite(x)
        dev = run.seg_dev(name)
        ison = valid & (np.nan_to_num(x) > 0.5)
        eps_ = _runs(ison)
        lens = [e - s + 1 for s, e in eps_]
        pm = per_seg_meta.get(f"{a}-{b}", {})
        seg_rows[name] = dict(
            share=float(np.mean(isn := ison[valid]) * 100.0) if valid.any() else np.nan,
            episodes=len(eps_), mean_len_s=float(np.mean(lens) * np.nanmedian(np.diff(t))) if lens else np.nan,
            dev_on=float(np.nanmean(np.abs(dev[isn_mask])) ) if (dev is not None and (isn_mask := valid & ison).any()) else np.nan,
            dev_off=float(np.nanmean(np.abs(dev[off_mask]))) if (dev is not None and (off_mask := valid & ~ison & np.isfinite(dev)).any()) else np.nan,
            fallback_pct=pm.get("fallback_pct", np.nan), gated_pct=pm.get("gated_pct", np.nan))
        for s, e in eps_:
            dseg = dev[s:e + 1] if dev is not None else np.array([np.nan])
            best_a, best_d = "", np.nan
            for ang in affected[name]:
                m = np.nanmax(np.abs(D[f"d_{ang}"].to_numpy()[s:e + 1])) if np.isfinite(D[f"d_{ang}"].to_numpy()[s:e + 1]).any() else np.nan
                if np.isfinite(m) and (not np.isfinite(best_d) or m > best_d):
                    best_a, best_d = ang, float(m)
            episodes.append(dict(model=run.model, segment=name, start_frame=int(frames[s]), end_frame=int(frames[e]),
                                 start_s=float(t[s]), dauer_s=float(t[e] - t[s]) + float(np.nanmedian(np.diff(t))),
                                 dev_mean_pct=float(np.nanmean(dseg)), dev_absmax_pct=float(np.nanmax(np.abs(dseg))) if np.isfinite(dseg).any() else np.nan,
                                 max_winkelwirkung_deg=best_d, winkel=best_a))
    ang_rows, cls_rows = {}, {}
    consistency_bad = 0
    for a in ANGLE_NAMES:
        d = D[f"d_{a}"].to_numpy(float)
        ad = np.abs(d)
        ok = np.isfinite(d)
        s1, s2 = ANGLE_SEGS[a]
        o1, o2 = np.nan_to_num(on[s1]) > 0.5, np.nan_to_num(on[s2]) > 0.5
        if ok.any():
            ang_rows[a] = dict(segs=(s1, s2), n=int(ok.sum()), mean=float(ad[ok].mean()), median=float(np.median(ad[ok])),
                               p95=float(np.percentile(ad[ok], 95)), max=float(ad[ok].max()), bias=float(d[ok].mean()),
                               gt1=float(np.mean(ad[ok] > 1.0) * 100.0), gt5=float(np.mean(ad[ok] > 5.0) * 100.0))
        else:
            ang_rows[a] = dict(segs=(s1, s2), n=0, mean=np.nan, median=np.nan, p95=np.nan, max=np.nan, bias=np.nan, gt1=np.nan, gt5=np.nan)
        classes = {"keins": ~o1 & ~o2, f"nur {s1}": o1 & ~o2, f"nur {s2}": ~o1 & o2, "beide": o1 & o2}
        cls_rows[a] = {k: (int((m & ok).sum()), float(ad[m & ok].mean()) if (m & ok).any() else np.nan) for k, m in classes.items()}
        consistency_bad += int(((~o1 & ~o2) & ok & (ad > 0.5)).sum())
    return dict(segments=seg_rows, episodes=episodes, angles=ang_rows, classes=cls_rows,
                consistency_bad=consistency_bad, on=on, t=t)


def model_mae(ra, rb, common, angles, kind):
    """Mittlere absolute Differenz zwischen den Modellen (kind: orig oder cor)."""
    A, B = ra.D.set_index("frame").loc[common], rb.D.set_index("frame").loc[common]
    vals = []
    for a in angles:
        ag = agreement(A[f"angle_{a}_{kind}"].to_numpy(), B[f"angle_{a}_{kind}"].to_numpy())
        if ag:
            vals.append(ag["mae"])
    return float(np.mean(vals)) if vals else np.nan


# ---------------------------------------------------------------- Plots
def plot_verlauf(path, runs, stats, prof, titel):
    panels = prof["panels"]
    fig, axs = plt.subplots(len(panels), 1, figsize=(12, 2.3 * len(panels)), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, (name, col, unit) in zip(axs, panels):
        for r in runs:
            y = r.D[col].to_numpy(float)
            ax.plot(r.D["t"], y, color=r.color, alpha=0.25, lw=0.8)
            ax.plot(r.D["t"], smooth(y), color=r.color, lw=1.6, label=r.label)
        ax.set_ylabel(f"{name}\n[{unit}]" if unit else name, fontsize=8)
        ax.grid(alpha=0.25)
    for r, st in zip(runs, stats):
        for p in st["rep_idx"]:
            for ax in axs:
                ax.axvline(r.D["t"].iloc[p], color=r.color, ls=":", lw=0.9, alpha=0.7)
    axs[0].legend(loc="upper right", ncol=len(runs))
    axs[0].set_title(f"{titel}: Verlauf, {ANGLE_LABEL[runs[0].angle_mode]} (dünn = roh, dick = Median-Filter 5; gepunktet = erkannte Wiederholungen)", fontsize=10)
    axs[-1].set_xlabel("Zeit [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_uebereinstimmung(path, ra, rb, angles, merged):
    n = len(angles)
    fig, axs = plt.subplots(2, n, figsize=(3.6 * n, 7), squeeze=False)
    for k, a in enumerate(angles):
        ag = agreement(merged[f"{a}_a"], merged[f"{a}_b"])
        ax, ax2 = axs[0, k], axs[1, k]
        ax.set_title(a, fontsize=10)
        if ag is None:
            ax.text(0.5, 0.5, "zu wenig Daten", ha="center"); ax2.axis("off"); continue
        lo, hi = min(ag["a"].min(), ag["b"].min()), max(ag["a"].max(), ag["b"].max())
        ax.scatter(ag["a"], ag["b"], s=6, alpha=0.5, color="#555")
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel(f"{ra.label} [°]"); ax.set_ylabel(f"{rb.label} [°]")
        ax.text(0.04, 0.96, f"r = {ag['corr']:.2f}\nMAE = {ag['mae']:.1f}°", transform=ax.transAxes, va="top", fontsize=9)
        ax.grid(alpha=0.25)
        ax2.scatter(ag["mean_ab"], ag["diff"], s=6, alpha=0.5, color="#555")
        ax2.axhline(ag["bias"], color="r", lw=1)
        ax2.axhline(ag["bias"] + 1.96 * ag["sd"], color="r", ls="--", lw=0.8)
        ax2.axhline(ag["bias"] - 1.96 * ag["sd"], color="r", ls="--", lw=0.8)
        ax2.set_xlabel("Mittelwert beider Modelle [°]"); ax2.set_ylabel(f"{rb.label} minus {ra.label} [°]")
        ax2.set_title(f"Bias {ag['bias']:+.1f}°, Grenzen ±{1.96 * ag['sd']:.1f}°", fontsize=9)
        ax2.grid(alpha=0.25)
    fig.suptitle(f"Übereinstimmung (oben) und Bland-Altman (unten), {ANGLE_LABEL[ra.angle_mode]}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def band_numbers(r):
    """Kennzahlen der Bandmessung eines Laufs (Griffpunktabstand gegen PCA-Referenz und Handabstand)."""
    D = r.D
    L, P, Wd = D["band_len_px"].to_numpy(float), D["band_len_pca_px"].to_numpy(float), D["wrist_dist_px"].to_numpy(float)
    out = dict(present_pct=float(np.nanmean(D["band_present"].to_numpy(float)) * 100.0),
               red_frac=float(np.nanmean(D["band_red_frac"].to_numpy(float))) if np.isfinite(D["band_red_frac"]).any() else np.nan)
    ok = np.isfinite(L) & np.isfinite(Wd)
    out["corr_wrist"] = float(np.corrcoef(L[ok], Wd[ok])[0, 1]) if ok.sum() > 10 else np.nan
    ok = np.isfinite(L) & np.isfinite(P)
    if ok.sum() > 10:
        off = (L - P)[ok]
        out.update(corr_pca=float(np.corrcoef(L[ok], P[ok])[0, 1]), off_mean=float(off.mean()), off_sd=float(off.std(ddof=1)),
                   off_sd_pct=float(off.std(ddof=1) / np.nanmean(L[ok]) * 100.0), off_mean_pct=float(off.mean() / np.nanmean(L[ok]) * 100.0))
    else:
        out.update(corr_pca=np.nan, off_mean=np.nan, off_sd=np.nan, off_sd_pct=np.nan, off_mean_pct=np.nan)
    return out


def plot_band(path, runs, titel):
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for r in runs:
        axs[0].plot(r.D["t"], r.D["stretch_raw_pct"], color=r.color, alpha=0.25, lw=0.8)
        axs[0].plot(r.D["t"], r.D["stretch_pct"], color=r.color, lw=1.6, label=f"{r.label}")
        axs[1].plot(r.D["t"], r.D["band_len_px"], color=r.color, lw=1.6, label=f"Griffpunktabstand {r.label}")
        axs[1].plot(r.D["t"], r.D["band_len_pca_px"], color=r.color, lw=1.0, ls="--", label=f"PCA-Referenz {r.label}")
        axs[1].plot(r.D["t"], r.D["wrist_dist_px"], color=r.color, lw=0.9, ls=":", label=f"Handgelenkabstand {r.label}")
        axs[2].plot(r.D["t"], r.D["band_len_px"] - r.D["band_len_pca_px"], color=r.color, lw=1.4, label=f"Versatz {r.label}")
    axs[0].axhline(0, color="k", lw=0.6)
    axs[0].set_ylabel("Banddehnung [%]\n(dünn = roh, dick = EMA)"); axs[0].legend(); axs[0].grid(alpha=0.25)
    axs[0].set_title(f"{titel}: Band je Modell (Länge = Griffpunktabstand, ab v6 modellabhängig)", fontsize=10)
    axs[1].set_ylabel("Pixel"); axs[1].legend(ncol=2, fontsize=8); axs[1].grid(alpha=0.25)
    axs[1].set_title("Länge: Griffpunkte gegen PCA-Referenz (alle roten Pixel) und Handgelenkabstand", fontsize=10)
    axs[2].axhline(0, color="k", lw=0.6)
    axs[2].set_ylabel("Griffpunktabstand minus PCA [px]"); axs[2].set_xlabel("Zeit [s]"); axs[2].grid(alpha=0.25)
    ax2 = axs[2].twinx()
    for r in runs:
        ax2.plot(r.D["t"], r.D["band_red_frac"], color=r.color, lw=0.8, ls=":", alpha=0.7)
    ax2.set_ylabel("band_red_frac (gepunktet)"); ax2.set_ylim(0, 1.05)
    axs[2].set_title("Versatz: konstant = Kalibrierung möglich, schwankend = haltungsabhängig", fontsize=10)
    axs[2].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_qualitaet(path, runs, stats):
    fig, axs = plt.subplots(1, 3, figsize=(18, 8), gridspec_kw=dict(width_ratios=[1.6, 1.2, 0.8]))
    names = [s[0] for s in SEGMENTS]
    y = np.arange(len(names)); h = 0.38
    for k, (r, st) in enumerate(zip(runs, stats)):
        dev = [st["segmente"][n][1] for n in names]
        axs[0].barh(y + (k - 0.5) * h, dev, h, color=r.color, label=f"{r.label} (Ø |Abw.| {st['seg_abw_mean_pct']:.1f} %)")
    axs[0].set_yticks(y); axs[0].set_yticklabels(names, fontsize=8); axs[0].invert_yaxis()
    axs[0].axvline(0, color="k", lw=0.8); axs[0].set_xlabel("Abweichung zu gemessen [%] (Modell roh)")
    axs[0].set_title("Segmentlängen gegen deine Messung", fontsize=10); axs[0].legend(fontsize=8); axs[0].grid(alpha=0.25)
    x = np.arange(len(ANGLE_NAMES)); w = 0.38
    for k, (r, st) in enumerate(zip(runs, stats)):
        axs[1].bar(x + (k - 0.5) * w, [st["jitter_by_angle"][a][0] for a in ANGLE_NAMES], w, color=r.color, label=r.label)
    axs[1].set_xticks(x); axs[1].set_xticklabels(ANGLE_NAMES, rotation=60, fontsize=8)
    axs[1].set_ylabel("Jitter (RMS 2. Differenz) [°/Frame²]"); axs[1].set_title(f"Unruhe der Winkel, {ANGLE_LABEL[runs[0].angle_mode]}", fontsize=10)
    axs[1].legend(fontsize=8); axs[1].grid(alpha=0.25, axis="y")
    for k, (r, st) in enumerate(zip(runs, stats)):
        axs[2].bar(k, st["latenz_mean_ms"], color=r.color)
        axs[2].plot([k, k], [st["latenz_median_ms"], st["latenz_p95_ms"]], color="k", lw=1.5)
        axs[2].text(k, st["latenz_mean_ms"] + 1, f"{st['latenz_mean_ms']:.0f} ms", ha="center", fontsize=9)
    axs[2].set_xticks(range(len(runs))); axs[2].set_xticklabels([r.label for r in runs])
    axs[2].set_ylabel("Inferenzzeit [ms]"); axs[2].set_title("Latenz (Balken Mittel, Linie Median bis P95)", fontsize=10)
    axs[2].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_korrektur(path, runs, cas):
    """Anteil korrigierter Frames je Segment und Wirkung der Korrektur je Winkel."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.5), gridspec_kw=dict(width_ratios=[1.2, 1.1, 1.1]))
    names = [s[0] for s in CORR_SEGS]
    y = np.arange(len(names)); h = 0.38
    for k, (r, ca) in enumerate(zip(runs, cas)):
        if ca is None:
            continue
        axs[0].barh(y + (k - 0.5) * h, [ca["segments"][n]["share"] for n in names], h, color=r.color, label=r.label)
    axs[0].set_yticks(y); axs[0].set_yticklabels(names, fontsize=8); axs[0].invert_yaxis()
    axs[0].set_xlabel("Frames mit Tiefenkorrektur [%]"); axs[0].set_title("Welche Segmente werden korrigiert", fontsize=10)
    axs[0].legend(fontsize=8); axs[0].grid(alpha=0.25)
    x = np.arange(len(ANGLE_NAMES)); w = 0.38
    for k, (r, ca) in enumerate(zip(runs, cas)):
        if ca is None:
            continue
        axs[1].bar(x + (k - 0.5) * w, [ca["angles"][a]["mean"] for a in ANGLE_NAMES], w, color=r.color, label=f"{r.label} Mittel")
        axs[1].plot(x + (k - 0.5) * w, [ca["angles"][a]["p95"] for a in ANGLE_NAMES], "k_", ms=9, mew=1.6)
    axs[1].set_xticks(x); axs[1].set_xticklabels(ANGLE_NAMES, rotation=60, fontsize=8)
    axs[1].set_ylabel("|Δ Winkel| [°] (Balken Mittel, Strich 95. Perz.)")
    axs[1].set_title("Wirkung auf die Winkel (korrigiert minus roh)", fontsize=10); axs[1].legend(fontsize=8); axs[1].grid(alpha=0.25, axis="y")
    for k, (r, ca) in enumerate(zip(runs, cas)):
        if ca is None:
            continue
        axs[2].bar(x + (k - 0.5) * w, [ca["angles"][a]["bias"] for a in ANGLE_NAMES], w, color=r.color, label=r.label)
    axs[2].axhline(0, color="k", lw=0.8)
    axs[2].set_xticks(x); axs[2].set_xticklabels(ANGLE_NAMES, rotation=60, fontsize=8)
    axs[2].set_ylabel("Mittlere Änderung Δ Winkel [°] (mit Vorzeichen)")
    axs[2].set_title("Richtung der Änderung (positiv = korrigiert grösser)", fontsize=10); axs[2].legend(fontsize=8); axs[2].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_korrektur_zeit(path, runs, cas, stats, prof):
    """Je Modell: Raster 'Segment x Zeit' (schwarz = korrigiert) und die Winkeländerung darunter."""
    use = [(r, ca, st) for r, ca, st in zip(runs, cas, stats) if ca is not None]
    fig, axs = plt.subplots(2 * len(use), 1, figsize=(13, 5.2 * len(use)), sharex=True,
                            gridspec_kw=dict(height_ratios=[1.5, 1] * len(use)))
    axs = np.atleast_1d(axs)
    names = [s[0] for s in CORR_SEGS]
    rel = prof["angles"] or ANGLE_NAMES[:4]
    cmap = plt.get_cmap("tab10")
    for k, (r, ca, st) in enumerate(use):
        ax, ax2 = axs[2 * k], axs[2 * k + 1]
        M = np.vstack([np.nan_to_num(ca["on"][n]) for n in names])
        t = ca["t"]
        ax.imshow(M, aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest",
                  extent=[t[0], t[-1], len(names), 0])
        ax.set_yticks(np.arange(len(names)) + 0.5); ax.set_yticklabels(names, fontsize=7)
        ax.set_title(f"{r.label}: Segmente mit Tiefenkorrektur (schwarz = aktiv)", fontsize=10)
        for a in ANGLE_NAMES:
            ax2.plot(t, r.D[f"d_{a}"], color="#bbbbbb", lw=0.6)
        for i, a in enumerate(rel):
            ax2.plot(t, r.D[f"d_{a}"], color=cmap(i), lw=1.2, label=f"{a}")
        ax2.axhline(0, color="k", lw=0.6)
        ax2.set_ylabel("Δ Winkel [°]\n(korrigiert minus roh)"); ax2.grid(alpha=0.25)
        ax2.legend(ncol=len(rel), fontsize=8, loc="upper right")
        ax2.set_title("Winkeländerung durch die Korrektur (bunt = relevante Winkel, grau = übrige)", fontsize=9)
        for p in st["rep_idx"]:
            ax.axvline(t[p], color="#d62728", ls=":", lw=0.9)
            ax2.axvline(t[p], color="#d62728", ls=":", lw=0.9)
    axs[-1].set_xlabel("Zeit [s] (rot gepunktet = erkannte Wiederholungen)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- Vergleichsvideo
def render_video(path, runs, video_path, band_runs, max_frames=0):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Video nicht lesbar: {video_path}")
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    idx = {r.model: {int(f): i for i, f in enumerate(r.D["frame"])} for r in runs}
    last = min(int(r.D["frame"].iloc[-1]) for r in runs)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or n > last or (max_frames and n >= max_frames):
            break
        for r in runs:
            i = idx[r.model].get(n)
            if i is None or not r.df["pose_detected"].iloc[i]:
                continue
            col = BGR.get(r.model, (200, 200, 200))
            pts = {j: (int(r.df[f"{j}_ix"].iloc[i] * W), int(r.df[f"{j}_iy"].iloc[i] * H)) for j in JOINTS}
            for a, b in BONES:
                cv2.line(frame, pts[a], pts[b], col, 2, cv2.LINE_AA)
            for j in JOINTS:
                cv2.circle(frame, pts[j], 5, col, -1)
        for k, r in enumerate(band_runs):
            i = idx[r.model].get(n)
            if i is None or r.D["band_present"].iloc[i] != 1:
                continue
            col = BGR.get(r.model, (255, 0, 255))
            if r.has_grip and np.isfinite(r.D["grip_l_ix"].iloc[i]) and np.isfinite(r.D["grip_r_ix"].iloc[i]):
                p0 = (int(r.D["grip_l_ix"].iloc[i] * W), int(r.D["grip_l_iy"].iloc[i] * H))
                p1 = (int(r.D["grip_r_ix"].iloc[i] * W), int(r.D["grip_r_iy"].iloc[i] * H))
            else:                                        # v4/v5: Mittelpunkt, Winkel und Laenge der PCA-Achse
                cx, cy = r.D["band_cx"].iloc[i], r.D["band_cy"].iloc[i]
                ang, ln = np.radians(r.D["band_angle_deg"].iloc[i]), r.D["band_len_px"].iloc[i]
                p0 = (int(cx - np.cos(ang) * ln / 2), int(cy - np.sin(ang) * ln / 2))
                p1 = (int(cx + np.cos(ang) * ln / 2), int(cy + np.sin(ang) * ln / 2))
            cv2.line(frame, p0, p1, col, 3, cv2.LINE_AA)
            s = r.D["stretch_pct"].iloc[i]
            if np.isfinite(s):
                cv2.putText(frame, f"{r.label} Dehnung {s:+.0f} %", (12, 78 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
        for k, r in enumerate(runs):
            cv2.putText(frame, r.label, (12, 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BGR.get(r.model, (200, 200, 200)), 2, cv2.LINE_AA)
        cv2.putText(frame, f"frame {n}  t {n / fps:.2f} s", (12, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(frame)
        n += 1
    cap.release(); vw.release()
    return True


# ---------------------------------------------------------------- Bericht
def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt(v, nd=1, unit=""):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}{unit}"


def correction_report(runs, cas):
    """Markdown-Abschnitt zur Segmentkorrektur."""
    L = ["## Segmentkorrektur und Wirkung auf die Winkel", "",
         "Δ Winkel = `angle_*_cor_raw` minus `angle_*_raw` (beide ungeglättet). Ein Segment gilt als korrigiert, wenn die Tiefenkorrektur "
         "in diesem Frame aktiv war (Abweichung zur Solllänge ab 5 %, Rückfall unter 3 %). Sind beide an einem Winkel beteiligten "
         "Segmente nicht korrigiert, bleibt die Richtung erhalten und Δ ist 0.", ""]
    for r, ca in zip(runs, cas):
        L.append(f"### {r.label}")
        if ca is None:
            L += ["Keine Korrekturdaten (kein v6-Format oder `_segdev.csv` fehlt).", ""]
            continue
        rows = []
        for name, *_ in CORR_SEGS:
            s = ca["segments"][name]
            rows.append([name, fmt(s["share"], 1), s["episodes"], fmt(s["mean_len_s"], 2), fmt(s["dev_on"], 1), fmt(s["dev_off"], 1),
                         fmt(s["fallback_pct"], 1), fmt(s["gated_pct"], 1)])
        L += ["**Wann welche Segmente korrigiert wurden**",
              md_table(["Segment", "korrigiert [% Frames]", "Episoden", "Ø Dauer [s]", "Ø Betrag Abw. bei Korrektur [%]",
                        "Ø Betrag Abw. ohne Korrektur [%]", "Fallback [% Frames]", "gesperrt vis [% Frames]"], rows), ""]
        rows = []
        for a in ANGLE_NAMES:
            v = ca["angles"][a]
            rows.append([a, f"{v['segs'][0]} + {v['segs'][1]}", fmt(v["mean"], 2), fmt(v["median"], 2), fmt(v["p95"], 1), fmt(v["max"], 1),
                         fmt(v["bias"], 2), fmt(v["gt1"], 0), fmt(v["gt5"], 0)])
        L += ["**Wirkung auf die Winkel (Δ = korrigiert minus roh)**",
              md_table(["Winkel", "angrenzende Segmente", "Ø Betrag Δ [°]", "Median Betrag Δ [°]", "P95 Betrag Δ [°]", "Max Betrag Δ [°]",
                        "Ø Δ [°]", "Betrag Δ > 1° [% Frames]", "Betrag Δ > 5° [% Frames]"], rows), ""]
        rows = []
        for a in ANGLE_NAMES:
            cl = ca["classes"][a]
            keys = list(cl)
            rows.append([a] + [f"{k.replace('nur ', 'nur ')}: n={cl[k][0]}, Ø Betrag Δ {fmt(cl[k][1], 1)}°" for k in keys])
        L += ["**Zuordnung: welche Segmentkorrektur erzeugt welche Winkeländerung**",
              md_table(["Winkel", "kein Segment", "nur Segment 1", "nur Segment 2", "beide"], rows), ""]
        L.append(f"Konsistenzprüfung: Frames mit beiden Segmenten unkorrigiert und Betrag Δ > 0.5°: {ca['consistency_bad']} (Soll 0).")
        eps = sorted([e for e in ca["episodes"] if np.isfinite(e["max_winkelwirkung_deg"])], key=lambda e: -e["max_winkelwirkung_deg"])[:8]
        if eps:
            L += ["", "**Die 8 Korrektur-Episoden mit der grössten Winkelwirkung** (alle Episoden: `korrekturen.csv`)",
                  md_table(["Segment", "Frames", "Start [s]", "Dauer [s]", "Ø Abw. roh [%]", "max Betrag Δ [°]", "Winkel"],
                           [[e["segment"], f"{e['start_frame']}–{e['end_frame']}", fmt(e["start_s"], 2), fmt(e["dauer_s"], 2),
                             fmt(e["dev_mean_pct"], 1), fmt(e["max_winkelwirkung_deg"], 1), e["winkel"]] for e in eps])]
        L.append("")
    return L


def analyse(ra, rb, exercise, out_dir, video, make_video):
    prof = PROFILES.get(exercise, PROFILES["generic"])
    if prof["panels"] is None:   # generic: die vier Winkel mit der groessten Streuung
        var = {a: np.nanstd(smooth(ra.D[f"angle_{a}_raw"].to_numpy())) for a in ANGLE_NAMES}
        top = sorted(var, key=lambda k: -(var[k] if np.isfinite(var[k]) else 0))[:4]
        prof = dict(prof, panels=[(a, f"angle_{a}_raw", "°") for a in top], angles=top,
                    rep=(f"angle_{top[0]}_raw", "max", 15.0, top[0]))
    titel = prof["titel"]
    common = sorted(set(ra.D["frame"]) & set(rb.D["frame"]))
    if len(common) < 10:
        raise SystemExit("Zu wenige gemeinsame Frames.")
    runs = [ra, rb]
    stats = [run_stats(r, prof, common) for r in runs]
    cas = [correction_analysis(r, prof) for r in runs]
    A, B = ra.D.set_index("frame").loc[common], rb.D.set_index("frame").loc[common]
    merged = pd.DataFrame(index=A.index)
    for a in ANGLE_NAMES:
        merged[f"{a}_a"], merged[f"{a}_b"] = A[f"angle_{a}_raw"].to_numpy(), B[f"angle_{a}_raw"].to_numpy()
    out_dir.mkdir(parents=True, exist_ok=True)
    angles = prof["angles"] or ANGLE_NAMES[:4]

    plot_verlauf(out_dir / "01_verlauf.png", runs, stats, prof, titel)
    plot_uebereinstimmung(out_dir / "02_uebereinstimmung.png", ra, rb, angles, merged)
    band_runs = [r for r in runs if r.has_band]
    if band_runs:
        plot_band(out_dir / "03_band.png", band_runs, titel)
    plot_qualitaet(out_dir / "04_qualitaet.png", runs, stats)
    have_corr = any(ca is not None for ca in cas)
    if have_corr:
        plot_korrektur(out_dir / "05_korrektur.png", runs, cas)
        plot_korrektur_zeit(out_dir / "06_korrektur_zeit.png", runs, cas, stats, prof)
        pd.DataFrame([e for ca in cas if ca for e in ca["episodes"]]).to_csv(out_dir / "korrekturen.csv", index=False)
        wrows = []
        for r, ca in zip(runs, cas):
            if ca:
                for a in ANGLE_NAMES:
                    v = ca["angles"][a]
                    wrows.append(dict(model=r.model, angle=a, segment_1=v["segs"][0], segment_2=v["segs"][1], n=v["n"],
                                      mean_abs_delta=v["mean"], median_abs_delta=v["median"], p95_abs_delta=v["p95"],
                                      max_abs_delta=v["max"], mean_delta=v["bias"], pct_gt_1deg=v["gt1"], pct_gt_5deg=v["gt5"]))
        pd.DataFrame(wrows).to_csv(out_dir / "winkelwirkung.csv", index=False)
    if make_video and video:
        render_video(out_dir / "vergleich.mp4", runs, video, band_runs)

    # ---- Kennzahlen
    ag_rows, ag_all = [], {}
    for a in ANGLE_NAMES:
        ag = agreement(merged[f"{a}_a"], merged[f"{a}_b"])
        ag_all[a] = ag
        if ag:
            ag_rows.append([a, ag["n"], f"{ag['bias']:+.1f}", f"{ag['mae']:.1f}", f"{ag['rmse']:.1f}", f"{ag['corr']:.2f}"])
    mae_primary = float(np.nanmean([ag_all[a]["mae"] for a in angles if ag_all.get(a)])) if any(ag_all.get(a) for a in angles) else np.nan
    mae_orig = model_mae(ra, rb, common, angles, "orig")
    mae_cor = model_mae(ra, rb, common, angles, "cor") if (ra.has_cor and rb.has_cor) else np.nan

    rep_tab, rep_col = [], prof["rep"]
    reps_by = [st["rep_idx"] for st in stats]
    extra_lines = []
    if rep_col:
        col, mode, _, label = rep_col
        for r, st in zip(runs, stats):
            for k, p in enumerate(st["rep_idx"], 1):
                nxt = st["rep_idx"][k] if k < len(st["rep_idx"]) else None
                row = [r.label, k, int(r.D["frame"].iloc[p]), f"{r.D['t'].iloc[p]:.2f}",
                       fmt(float(st["rep_signal"][p]), 2 if col in ("wrist_spread",) else 0),
                       fmt(float(r.D["t"].iloc[nxt] - r.D["t"].iloc[p]) if nxt is not None else None, 2)]
                if r.has_band:
                    row.append(fmt(float(r.D["stretch_pct"].iloc[p]), 0, " %"))
                if exercise in ("reverse_fly_jacks", "reverse_butterfly"):
                    row.append(fmt(float(np.nanmean([r.D["angle_elbow_l_raw"].iloc[p], r.D["angle_elbow_r_raw"].iloc[p]])), 0, "°"))
                rep_tab.append(row)
        pairs = match_reps(reps_by[0], reps_by[1])
        extra_lines.append(f"Wiederholungen erkannt: {ra.label} {len(reps_by[0])}, {rb.label} {len(reps_by[1])}; "
                           f"davon zeitlich zuordenbar (±20 Frames): {len(pairs)}.")
        if pairs:
            dts = [abs(ra.D['t'].iloc[reps_by[0][i]] - rb.D['t'].iloc[reps_by[1][j]]) for i, j in pairs]
            extra_lines.append(f"Mittlerer Zeitversatz der Wiederholungsspitzen zwischen den Modellen: {np.mean(dts) * 1000:.0f} ms.")
    sync = {}
    fps = float(ra.meta.get("source_fps") or 30.0)
    if exercise == "reverse_fly_jacks":
        for r in runs:
            s = sync_lag(smooth(r.D["wrist_spread"].to_numpy()), smooth(r.D["ankle_spread"].to_numpy()))
            if s:
                sync[r.model] = s
                extra_lines.append(f"Synchronität Arme/Beine ({r.label}): Beinspreizung folgt der Armspreizung mit {s[0]:+d} Frames "
                                   f"({s[0] * 1000 / fps:+.0f} ms bei {fps:.0f} fps), Korrelation {s[1]:.2f}.")
    bnum = {r.model: band_numbers(r) for r in band_runs}

    # ---- metrics.csv + all_runs_v2.csv
    created = datetime.now().isoformat(timespec="seconds")
    video_name = Path(video).stem if video else Path(ra.source).stem
    rows = []
    for r, st, ca in zip(runs, stats, cas):
        shares = [v["share"] for v in ca["segments"].values()] if ca else []
        dmean = [ca["angles"][a]["mean"] for a in (prof["angles"] or ANGLE_NAMES)] if ca else []
        bn = bnum.get(r.model, {})
        rows.append(dict(run_id=out_dir.name, created=created, video=video_name, exercise=exercise, model=r.model,
                         angle_mode=r.angle_mode, detector_stride=st["detector_stride"] if st["detector_stride"] is not None else "",
                         frames=st["frames"], common_frames=len(common), duration_s=round(st["dauer_s"], 2),
                         detect_pct=round(st["erkannt_pct"], 1), infer_ms_mean=round(st["latenz_mean_ms"], 1),
                         infer_ms_p95=round(st["latenz_p95_ms"], 1), jitter_mean=round(st["jitter_mean"], 2),
                         jitter_orig_mean=round(st["jitter_orig_mean"], 2) if np.isfinite(st["jitter_orig_mean"]) else "",
                         jitter_cor_mean=round(st["jitter_cor_mean"], 2) if np.isfinite(st["jitter_cor_mean"]) else "",
                         spikes=st["spikes"], seg_abw_mean_pct=round(st["seg_abw_mean_pct"], 1),
                         corr_share_mean_pct=round(float(np.nanmean(shares)), 1) if shares else "",
                         delta_abs_mean=round(float(np.nanmean(dmean)), 2) if dmean else "",
                         reps=len(st["rep_idx"]), agree_mae_mean=round(mae_primary, 2) if np.isfinite(mae_primary) else "",
                         band_corr_wrist=round(bn["corr_wrist"], 3) if bn and np.isfinite(bn["corr_wrist"]) else "",
                         band_offset_sd_pct=round(bn["off_sd_pct"], 1) if bn and np.isfinite(bn["off_sd_pct"]) else ""))
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
    all_path = out_dir.parent / "all_runs_v2.csv"
    pd.DataFrame(rows).to_csv(all_path, mode="a", header=not all_path.exists(), index=False)

    # ---- report.md
    def jc(st):
        return f"{fmt(st['jitter_orig_mean'], 1)} / {fmt(st['jitter_cor_mean'], 1)}"
    L = [f"# Vergleich {ra.label} und {rb.label}: {titel}", "",
         f"Video: {video_name} | Übung: {exercise} | Winkelquelle: {ANGLE_LABEL[ra.angle_mode]} | gemeinsame Frames: {len(common)} | erstellt: {created}", "",
         "## Überblick", md_table(["", ra.label, rb.label], [
             ["Frames (gesamt)", stats[0]["frames"], stats[1]["frames"]],
             ["Dauer [s]", fmt(stats[0]["dauer_s"], 1), fmt(stats[1]["dauer_s"], 1)],
             ["Pose erkannt [%]", fmt(stats[0]["erkannt_pct"], 1), fmt(stats[1]["erkannt_pct"], 1)],
             ["Detektor-Stride (Meta)", stats[0]["detector_stride"] if stats[0]["detector_stride"] is not None else "–",
              stats[1]["detector_stride"] if stats[1]["detector_stride"] is not None else "–"],
             ["Frisch ausgewertete Frames `fresh` [%]", fmt(stats[0]["fresh_pct"], 1), fmt(stats[1]["fresh_pct"], 1)],
             ["Latenz Mittel / 95. Perz. [ms]", f"{stats[0]['latenz_mean_ms']:.0f} / {stats[0]['latenz_p95_ms']:.0f}", f"{stats[1]['latenz_mean_ms']:.0f} / {stats[1]['latenz_p95_ms']:.0f}"],
             ["Gelenke unter Sichtbarkeit 0.5 [%]", fmt(stats[0]["vis_unter_schwelle_pct"], 1), fmt(stats[1]["vis_unter_schwelle_pct"], 1)],
             ["Jitter Ø (RMS 2. Diff., relevante Winkel, gewählte Quelle) [°/Frame²]", fmt(stats[0]["jitter_mean"], 1), fmt(stats[1]["jitter_mean"], 1)],
             ["Jitter Ø roh / korrigiert [°/Frame²]", jc(stats[0]), jc(stats[1])],
             [f"Winkelsprünge > {SPIKE_DEG:.0f}°/Frame (relevante Winkel)", stats[0]["spikes"], stats[1]["spikes"]],
             ["Ø Betrag Segmentabweichung roh zu gemessen [%]", fmt(stats[0]["seg_abw_mean_pct"], 1), fmt(stats[1]["seg_abw_mean_pct"], 1)],
             ["Wiederholungen erkannt", len(stats[0]["rep_idx"]), len(stats[1]["rep_idx"])]]), "",
         f"## Übereinstimmung ({rb.label} minus {ra.label}, {ANGLE_LABEL[ra.angle_mode]})", md_table(["Winkel", "n", "Bias [°]", "MAE [°]", "RMSE [°]", "Korrelation"], ag_rows), ""]
    if np.isfinite(mae_orig) and np.isfinite(mae_cor):
        L += [f"Modell-zu-Modell-MAE über die relevanten Winkel: roh {mae_orig:.2f}°, nach Tiefenkorrektur {mae_cor:.2f}° "
              f"({'näher beieinander' if mae_cor < mae_orig else 'weiter auseinander'}). Das ist ein Mass der Übereinstimmung, nicht der Genauigkeit.", ""]
    if rep_tab:
        hdr = ["Modell", "Nr.", "Frame", "Zeit [s]", f"Spitzenwert ({rep_col[3]})", "bis nächste [s]"]
        if any(r.has_band for r in runs):
            hdr.append("Dehnung an Spitze")
        if exercise in ("reverse_fly_jacks", "reverse_butterfly"):
            hdr.append("Ellbogen Mittel an Spitze")
        L += ["## Wiederholungen", md_table(hdr, rep_tab), ""] + [f"- {x}" for x in extra_lines] + [""]
    if bnum:
        brow = [[NICE.get(k, k), fmt(v["present_pct"], 1), fmt(v["red_frac"], 2), fmt(v["corr_wrist"], 2), fmt(v["corr_pca"], 2),
                 f"{fmt(v['off_mean'], 1)} ± {fmt(v['off_sd'], 1)}", fmt(v["off_mean_pct"], 1), fmt(v["off_sd_pct"], 1)] for k, v in bnum.items()]
        L += ["## Band (Länge = Griffpunktabstand, ab v6 modellabhängig)",
              md_table(["Modell", "present [%]", "Ø band_red_frac", "r Handgelenkabstand", "r PCA-Referenz", "Versatz Griffpunkt minus PCA Ø ± SD [px]",
                        "Versatz Ø [% der Länge]", "Versatz SD [% der Länge]"], brow),
              "Der Versatz Griffpunkt minus PCA-Referenz zeigt, ob die Fingerspitzen-Griffpunkte konstant gegenüber dem Band verschoben sind "
              "(kleine SD: Kalibrierung des Versatzes möglich) oder haltungsabhängig (grosse SD: Bandenden aus den Rotpixeln bestimmen). "
              "Die PCA-Referenz nimmt alle roten Pixel im Bild und ist bei rotem Hintergrund oder roter Kleidung verfälscht.", ""]
    if have_corr:
        L += correction_report(runs, cas)
    seg_rows = [[n, f"{L_:.3f}"] + [f"{st['segmente'][n][0]:.3f} ({st['segmente'][n][1]:+.1f} %)" if np.isfinite(st['segmente'][n][0]) else "–" for st in stats]
                for n, _, _, L_ in SEGMENTS]
    hints = ["- Ohne Referenzmessung (Goniometer, Marker) beschreibt dieser Bericht Übereinstimmung und Stabilität, nicht Genauigkeit.",
             "- Bias, MAE und Korrelation gelten nur für diese Aufnahme (eine Person, eine Kameraposition).",
             "- Die Tiefenkorrektur übernimmt das Vorzeichen von dz vom Modell. Ein falsches Vorzeichen wird nicht korrigiert.",
             "- Hüftbreite weicht bei beiden Modellen stark ab, vermutlich Definitionsunterschied (Gelenkzentrum gegen gemessene Breite). Hüft- und Schulterbreite werden nicht korrigiert."]
    for r, st in zip(runs, stats):
        if st["detector_stride"] not in (None, 1):
            hints.append(f"- {r.label}: Detektor-Stride {st['detector_stride']}. Bbox-Wechsel zwischen Detektor-Frames erzeugen Winkelsprünge; Lauf mit `--detector-stride 1` wiederholen.")
    L += ["## Segmentlängen gegen Messung (Modell roh)", md_table(["Segment", "gemessen [m]", ra.label, rb.label], seg_rows), "",
          "## Hinweise"] + hints + ["",
          "## Dateien", "01_verlauf.png, 02_uebereinstimmung.png, " + ("03_band.png, " if band_runs else "") + "04_qualitaet.png, "
          + ("05_korrektur.png, 06_korrektur_zeit.png, korrekturen.csv, winkelwirkung.csv, " if have_corr else "") + "metrics.csv"
          + (", vergleich.mp4" if (make_video and video) else "")]
    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    return out_dir, stats, ag_all, bnum, sync, cas


# ---------------------------------------------------------------- Auswahl und Start
def guess_exercise(label):
    s = label.lower()
    if "squat" in s or "kniebeuge" in s:
        return "squat"
    if "butterfly" in s:
        return "reverse_butterfly"
    if "fly" in s or "jack" in s:
        return "reverse_fly_jacks"
    if "zug" in s or "row" in s or "pull" in s:
        return "seitzug"
    return ""


def scan_csvs(folder):
    items = []
    for p in sorted(Path(folder).glob("*.csv"), key=lambda q: q.stat().st_mtime, reverse=True):
        stem = p.stem.lower()
        if stem.endswith("_segdev"):
            continue
        if stem.startswith("mediapipe_") or stem.startswith("instanthmr_"):
            model = stem.split("_")[0]
            mp = p.with_name(p.stem + "_meta.json")
            frames = ""
            try:
                frames = json.loads(mp.read_text(encoding="utf-8")).get("frames_written", "")
            except Exception:
                pass
            items.append((p, model, frames, datetime.fromtimestamp(p.stat().st_mtime)))
    return items


def interactive(args):
    cfg = load_cfg()
    folder = Path(args.folder) if args.folder else find_repo_root() / "data" / "output"
    folder = Path(clean_path(ask("Ordner mit den CSV-Dateien", str(folder))))
    items = scan_csvs(folder)
    if not items:
        raise SystemExit(f"Keine mediapipe_*.csv / instanthmr_*.csv in {folder}")
    print("\nGefundene Läufe (neueste zuerst):")
    for i, (p, m, fr, ts) in enumerate(items, 1):
        print(f"  {i:2d}) {m:10s} {fr!s:>5} Frames  {ts:%d.%m. %H:%M}  {p.name}")
    first = {m: i for i, (_, m, _, _) in reversed(list(enumerate(items, 1)))}

    def pick(model):
        while True:
            v = ask(f"Nummer der {NICE[model]}-Datei", first.get(model, ""))
            if v.isdigit() and 1 <= int(v) <= len(items) and items[int(v) - 1][1] == model:
                return items[int(v) - 1][0]
            print(f"Bitte eine Zeile mit Modell {model} wählen.")
    args.mp, args.ih = pick("mediapipe"), pick("instanthmr")
    probe = Run(args.mp)
    ex = guess_exercise(probe.exercise) or cfg.get("analyse_exercise", "reverse_butterfly")
    opts = list(PROFILES)
    print("\nÜbung: " + ", ".join(f"{i + 1} = {o}" for i, o in enumerate(opts)))
    v = ask("Übung (Nummer oder Name)", ex)
    args.exercise = opts[int(v) - 1] if v.isdigit() and 1 <= int(v) <= len(opts) else (v if v in PROFILES else ex)
    w = ask("Winkelquelle: 1 = roh (Modell pur), 2 = tiefenkorrigiert", "2" if cfg.get("analyse_winkel") == "cor" else "1")
    args.winkel = "cor" if w in ("2", "cor") else "raw"
    meta_src = Run(args.ih).source or probe.source
    vid = ask("Originalvideo für das Vergleichsvideo (Enter = aus der Meta-Datei, '-' = kein Video)", meta_src)
    args.no_video = vid.strip() == "-"
    args.video = None if args.no_video else clean_path(vid)
    save_cfg(analyse_exercise=args.exercise, analyse_winkel=args.winkel)


def main():
    global ANGLE_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp", help="CSV von run_mediapipe")
    ap.add_argument("--ih", help="CSV von run_instanthmr")
    ap.add_argument("--exercise", default=None, choices=list(PROFILES))
    ap.add_argument("--winkel", default="raw", choices=["raw", "cor"],
                    help="Winkelquelle: raw = Modell pur (angle_*_raw), cor = tiefenkorrigiert (angle_*_cor_raw)")
    ap.add_argument("--video", default=None, help="Originalvideo (Standard: aus Meta-Datei)")
    ap.add_argument("--no-video", action="store_true", help="kein Vergleichsvideo")
    ap.add_argument("--folder", default=None, help="Ordner der CSVs (nur für die Abfrage)")
    ap.add_argument("--out", default=None, help="Ausgabeordner (Standard: <Repo>/data/analysis)")
    if len(sys.argv) == 1:
        args = ap.parse_args([])
        interactive(args)
    else:
        args = ap.parse_args()
        if not (args.mp and args.ih):
            ap.error("--mp und --ih angeben (oder ohne Argumente starten für die Abfrage)")
        args.exercise = args.exercise or "generic"
    ANGLE_MODE = args.winkel
    ra, rb = Run(args.mp), Run(args.ih)
    if ra.model == rb.model:
        raise SystemExit("Beide CSVs stammen vom selben Modell.")
    video = args.video or ra.source or rb.source
    if video and not Path(video).exists():
        print(f"Originalvideo nicht gefunden ({video}), kein Vergleichsvideo.")
        video = None
    if ra.source and rb.source and Path(ra.source).stem != Path(rb.source).stem:
        print("Warnung: die beiden CSVs stammen von verschiedenen Videos.")
    base = Path(args.out) if args.out else find_repo_root() / "data" / "analysis"
    stem = Path(video).stem if video else Path(ra.source).stem or "lauf"
    out_dir = base / f"{stem}_{datetime.now():%Y%m%d_%H%M%S}".replace(" ", "_")
    print("\nAuswertung läuft ...")
    out_dir, stats, ag, bn, sync, cas = analyse(ra, rb, args.exercise, out_dir, video, not args.no_video)
    print(f"\nFertig. Ordner: {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")
    print(f"Sammeldatei: {out_dir.parent / 'all_runs_v2.csv'}")


if __name__ == "__main__":
    main()
