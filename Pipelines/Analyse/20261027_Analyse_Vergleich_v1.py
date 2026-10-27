"""
20261027_Analyse_Vergleich_v1.py
Vergleicht die CSV-Ausgaben von run_mediapipe und run_instanthmr (Pipelines v4/v5) und erzeugt:
  01_verlauf.png      Zeitverläufe beider Modelle (Winkel, Spreizung, Band) mit Wiederholungen
  02_uebereinstimmung.png  Streu- und Bland-Altman-Diagramme je Winkel
  03_band.png         Banddehnung und Validierung gegen den Handabstand (nur mit Band)
  04_qualitaet.png    Segmentabweichung, Jitter, Latenz
  report.md           Kennzahlen, Wiederholungen, Hinweise
  metrics.csv         Kennzahlen dieses Laufs
  vergleich.mp4       Original mit beiden Skeletten (optional)
und hängt eine Zeile pro Modell an data/analysis/all_runs.csv an (Vergleich über alle Videos).

Start ohne Argumente (Run-Button der IDE): Abfrage der beiden CSVs, der Übung und des Videos.
Kommandozeile:
  python 20261027_Analyse_Vergleich_v1.py --mp <mediapipe.csv> --ih <instanthmr.csv> --exercise reverse_fly_jacks

Übungen: squat, seitzug, reverse_fly_jacks, generic
Abhängigkeiten: numpy, pandas, matplotlib (Vergleichsvideo zusätzlich opencv-python)

Hinweis: Ohne Referenzmessung (Goniometer, gemessene Segmentlängen) zeigt der Vergleich
Übereinstimmung und Stabilität der Modelle, nicht deren Genauigkeit. Einzige Referenz im
Code: die gemessenen Segmentlängen (SEGMENT_LENGTHS).
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

# Gemessene Segmentlängen (m) von Fabrice, identisch zu den Pipelines v5
SEGMENTS = [  # (Anzeigename, Gelenk A, Gelenk B, Länge)
    ("Rumpf l", "l_hip", "l_shoulder", 0.480), ("Rumpf r", "r_hip", "r_shoulder", 0.480),
    ("Oberarm l", "l_shoulder", "l_elbow", 0.315), ("Oberarm r", "r_shoulder", "r_elbow", 0.315),
    ("Unterarm l", "l_elbow", "l_wrist", 0.270), ("Unterarm r", "r_elbow", "r_wrist", 0.270),
    ("Oberschenkel l", "l_hip", "l_knee", 0.475), ("Oberschenkel r", "r_hip", "r_knee", 0.475),
    ("Unterschenkel l", "l_knee", "l_ankle", 0.415), ("Unterschenkel r", "r_knee", "r_ankle", 0.415),
    ("Fuss l", "l_ankle", "l_foot", 0.200), ("Fuss r", "r_ankle", "r_foot", 0.200),
    ("Hüftbreite", "l_hip", "r_hip", 0.310), ("Schulterbreite", "l_shoulder", "r_shoulder", 0.400),
]
VIS_MIN = 0.5
SPIKE_DEG = 20.0          # Winkelsprung pro Frame, ab dem ein Frame als Ausreisser zählt
COLORS = {"mediapipe": "#1b9e77", "instanthmr": "#d95f02"}
BGR = {"mediapipe": (0, 200, 0), "instanthmr": (0, 140, 255)}
NICE = {"mediapipe": "MediaPipe", "instanthmr": "InstantHMR"}

# Übungsprofile: Zeitverlauf-Panels (Titel, Spalte, Einheit) und Signal für die Wiederholungserkennung
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
            print(f"Warnung: keine Auflösung in der Meta-Datei von {self.path.name}, nehme normierte Einheiten.")
            self.W = self.H = 1
        ex = self.df["exercise"].dropna() if "exercise" in self.df.columns else pd.Series([], dtype=str)
        self.exercise = str(ex.iloc[0]) if len(ex) else ""
        self.source = self.meta.get("source", "")
        self.has_raw = "l_hip_wx_raw" in self.df.columns          # v5-Format (korrigiert + roh)
        self.D = self._derive()

    def xyz(self, j):
        """Rohe 3D-Koordinaten (v4: _wx ist roh, v5: _wx_raw ist roh)."""
        s = "_raw" if self.has_raw else ""
        return self.df[[f"{j}_wx{s}", f"{j}_wy{s}", f"{j}_wz{s}"]].to_numpy(float)

    def px(self, j):
        return np.column_stack([self.df[f"{j}_ix"] * self.W, self.df[f"{j}_iy"] * self.H])

    def _derive(self):
        df = self.df
        D = pd.DataFrame({"frame": df["frame"].to_numpy(), "t": df["timestamp_ms"].to_numpy() / 1000.0})
        for a in ANGLE_NAMES:
            D[f"angle_{a}_raw"] = df[f"angle_{a}_raw"].to_numpy(float)
        sw = np.nanmedian(np.linalg.norm(self.px("l_shoulder") - self.px("r_shoulder"), axis=1))
        self.shoulder_px = float(sw)
        D["wrist_dist_px"] = np.linalg.norm(self.px("l_wrist") - self.px("r_wrist"), axis=1)
        D["wrist_spread"] = D["wrist_dist_px"] / sw
        D["ankle_spread"] = np.linalg.norm(self.px("l_ankle") - self.px("r_ankle"), axis=1) / sw
        D["knee_mean"] = D[["angle_knee_l_raw", "angle_knee_r_raw"]].mean(axis=1)
        D["elbow_mean"] = D[["angle_elbow_l_raw", "angle_elbow_r_raw"]].mean(axis=1)
        D["knee_asym"] = D["angle_knee_l_raw"] - D["angle_knee_r_raw"]
        for c in ("band_present", "band_len_px", "band_cx", "band_cy", "band_angle_deg", "stretch_pct", "stretch_raw_pct"):
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


def smooth(s, k=5):
    """Median-Filter gegen Einzelframe-Ausreisser."""
    return pd.Series(s).rolling(k, center=True, min_periods=1).median().to_numpy()


# ---------------------------------------------------------------- Kennzahlen
def detect_reps(sig, mode, min_range, min_gap=10):
    """Hysterese-Erkennung: Wiederholung = Auslenkung über 65 % der Spanne, Rücksetzen unter 35 %."""
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
    mask = D["frame"].isin(ref_frames)
    st = {"model": run.model, "frames": int(len(D)), "dauer_s": float(D["t"].iloc[-1]) if len(D) else 0.0}
    st["erkannt_pct"] = float(run.df["pose_detected"].mean() * 100.0)
    ims = run.df["infer_ms"].to_numpy(float)
    st["latenz_mean_ms"], st["latenz_median_ms"], st["latenz_p95_ms"] = float(np.nanmean(ims)), float(np.nanmedian(ims)), float(np.nanpercentile(ims, 95))
    vis = run.df[[f"{j}_vis" for j in JOINTS]].to_numpy(float)
    st["hat_sichtbarkeit"] = bool(np.nanstd(vis) > 1e-9)
    st["vis_unter_schwelle_pct"] = float(np.nanmean(vis < VIS_MIN) * 100.0) if st["hat_sichtbarkeit"] else np.nan
    jit, spk = [], 0
    jit_by = {}
    for a in ANGLE_NAMES:
        j, s = jitter(D[f"angle_{a}_raw"].to_numpy())
        jit_by[a] = (j, s)
        if a in (prof["angles"] or ANGLE_NAMES):
            jit.append(j)
            spk += s
    st["jitter_by_angle"] = jit_by
    st["jitter_mean"] = float(np.nanmean(jit)) if jit else np.nan
    st["spikes"] = spk
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
    rep_col = prof["rep"][0] if prof["rep"] else None
    for r, st in zip(runs, stats):
        for p in st["rep_idx"]:
            for ax in axs:
                ax.axvline(r.D["t"].iloc[p], color=r.color, ls=":", lw=0.9, alpha=0.7)
    axs[0].legend(loc="upper right", ncol=len(runs))
    axs[0].set_title(f"{titel}: Verlauf (dünn = roh, dick = Median-Filter 5; gepunktet = erkannte Wiederholungen)", fontsize=10)
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
    fig.suptitle("Übereinstimmung (oben) und Bland-Altman (unten), rohe Winkel", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_band(path, runs, titel):
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    r0 = runs[0]
    axs[0].plot(r0.D["t"], r0.D["stretch_raw_pct"], color="#888", alpha=0.5, lw=0.8, label="roh")
    axs[0].plot(r0.D["t"], r0.D["stretch_pct"], color="#c1121f", lw=1.6, label="EMA-geglättet")
    axs[0].axhline(0, color="k", lw=0.6)
    axs[0].set_ylabel("Banddehnung [%]"); axs[0].legend(); axs[0].grid(alpha=0.25)
    axs[0].set_title(f"{titel}: Band (die Bandmessung ist modellunabhängig)", fontsize=10)
    ax = axs[1]
    ax.plot(r0.D["t"], r0.D["band_len_px"], color="#c1121f", lw=1.6, label="Bandlänge [px]")
    txt = []
    for r in runs:
        ax.plot(r.D["t"], r.D["wrist_dist_px"], color=r.color, lw=1.2, label=f"Handabstand {r.label} [px]")
        ok = np.isfinite(r.D["band_len_px"]) & np.isfinite(r.D["wrist_dist_px"])
        if ok.sum() > 10:
            txt.append(f"{r.label}: r = {np.corrcoef(r.D['band_len_px'][ok], r.D['wrist_dist_px'][ok])[0, 1]:.2f}")
    ax.set_ylabel("Pixel"); ax.set_xlabel("Zeit [s]"); ax.grid(alpha=0.25); ax.legend(loc="upper right")
    ax.set_title("Validierung: Bandlänge gegen Handabstand aus der Pose   " + " | ".join(txt), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_qualitaet(path, runs, stats):
    fig, axs = plt.subplots(1, 3, figsize=(17, 6), gridspec_kw=dict(width_ratios=[1.6, 1.2, 0.8]))
    names = [s[0] for s in SEGMENTS]
    y = np.arange(len(names)); h = 0.38
    for k, (r, st) in enumerate(zip(runs, stats)):
        dev = [st["segmente"][n][1] for n in names]
        axs[0].barh(y + (k - 0.5) * h, dev, h, color=r.color, label=f"{r.label} (Ø |Abw.| {st['seg_abw_mean_pct']:.1f} %)")
    axs[0].set_yticks(y); axs[0].set_yticklabels(names, fontsize=8); axs[0].invert_yaxis()
    axs[0].axvline(0, color="k", lw=0.8); axs[0].set_xlabel("Abweichung zu gemessen [%]")
    axs[0].set_title("Segmentlängen gegen deine Messung", fontsize=10); axs[0].legend(fontsize=8); axs[0].grid(alpha=0.25)
    x = np.arange(len(ANGLE_NAMES)); w = 0.38
    for k, (r, st) in enumerate(zip(runs, stats)):
        axs[1].bar(x + (k - 0.5) * w, [st["jitter_by_angle"][a][0] for a in ANGLE_NAMES], w, color=r.color, label=r.label)
    axs[1].set_xticks(x); axs[1].set_xticklabels(ANGLE_NAMES, rotation=60, fontsize=8)
    axs[1].set_ylabel("Jitter (RMS 2. Differenz) [°/Frame²]"); axs[1].set_title("Unruhe der Winkel (kleiner = ruhiger)", fontsize=10)
    axs[1].legend(fontsize=8); axs[1].grid(alpha=0.25, axis="y")
    for k, (r, st) in enumerate(zip(runs, stats)):
        axs[2].bar(k, st["latenz_mean_ms"], color=r.color)
        axs[2].plot([k, k], [st["latenz_median_ms"], st["latenz_p95_ms"]], color="k", lw=1.5)
        axs[2].text(k, st["latenz_mean_ms"] + 1, f"{st['latenz_mean_ms']:.0f} ms", ha="center", fontsize=9)
    axs[2].set_xticks(range(len(runs))); axs[2].set_xticklabels([r.label for r in runs])
    axs[2].set_ylabel("Inferenzzeit [ms] (Balken Mittel, Linie Median bis 95. Perz.)"); axs[2].set_title("Latenz", fontsize=10)
    axs[2].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- Vergleichsvideo
def render_video(path, runs, video_path, band_run, max_frames=0):
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
        if band_run is not None:
            i = idx[band_run.model].get(n)
            if i is not None and band_run.D["band_present"].iloc[i] == 1:
                cx, cy = band_run.D["band_cx"].iloc[i], band_run.D["band_cy"].iloc[i]
                ang, ln = np.radians(band_run.D["band_angle_deg"].iloc[i]), band_run.D["band_len_px"].iloc[i]
                p0 = (int(cx - np.cos(ang) * ln / 2), int(cy - np.sin(ang) * ln / 2))
                p1 = (int(cx + np.cos(ang) * ln / 2), int(cy + np.sin(ang) * ln / 2))
                cv2.line(frame, p0, p1, (255, 0, 255), 2, cv2.LINE_AA)
                s = band_run.D["stretch_pct"].iloc[i]
                if np.isfinite(s):
                    cv2.putText(frame, f"Dehnung {s:+.0f} %", (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
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


def analyse(ra, rb, exercise, out_dir, video, make_video):
    prof = PROFILES.get(exercise, PROFILES["generic"])
    if prof["panels"] is None:   # generic: die vier Winkel mit der grössten Streuung
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
        plot_band(out_dir / "03_band.png", runs, titel)
    plot_qualitaet(out_dir / "04_qualitaet.png", runs, stats)
    if make_video and video:
        render_video(out_dir / "vergleich.mp4", runs, video, band_runs[0] if band_runs else None)

    # ---- Kennzahlen
    ag_rows, ag_all = [], {}
    for a in ANGLE_NAMES:
        ag = agreement(merged[f"{a}_a"], merged[f"{a}_b"])
        ag_all[a] = ag
        if ag:
            ag_rows.append([a, ag["n"], f"{ag['bias']:+.1f}", f"{ag['mae']:.1f}", f"{ag['rmse']:.1f}", f"{ag['corr']:.2f}"])
    mae_primary = float(np.nanmean([ag_all[a]["mae"] for a in angles if ag_all.get(a)])) if any(ag_all.get(a) for a in angles) else np.nan

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
                if exercise == "reverse_fly_jacks":
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
    band_corr = {}
    for r in band_runs:
        ok = np.isfinite(r.D["band_len_px"]) & np.isfinite(r.D["wrist_dist_px"])
        if ok.sum() > 10:
            band_corr[r.model] = float(np.corrcoef(r.D["band_len_px"][ok], r.D["wrist_dist_px"][ok])[0, 1])

    # ---- metrics.csv + all_runs.csv
    created = datetime.now().isoformat(timespec="seconds")
    video_name = Path(video).stem if video else Path(ra.source).stem
    rows = []
    for r, st in zip(runs, stats):
        rows.append(dict(run_id=out_dir.name, created=created, video=video_name, exercise=exercise, model=r.model,
                         frames=st["frames"], common_frames=len(common), duration_s=round(st["dauer_s"], 2),
                         detect_pct=round(st["erkannt_pct"], 1), infer_ms_mean=round(st["latenz_mean_ms"], 1),
                         infer_ms_p95=round(st["latenz_p95_ms"], 1), jitter_mean=round(st["jitter_mean"], 2),
                         spikes=st["spikes"], seg_abw_mean_pct=round(st["seg_abw_mean_pct"], 1),
                         reps=len(st["rep_idx"]), agree_mae_mean=round(mae_primary, 2) if np.isfinite(mae_primary) else "",
                         band_corr=round(band_corr.get(r.model, np.nan), 3) if r.model in band_corr else ""))
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
    all_path = out_dir.parent / "all_runs.csv"
    pd.DataFrame(rows).to_csv(all_path, mode="a", header=not all_path.exists(), index=False)

    # ---- report.md
    L = [f"# Vergleich {ra.label} und {rb.label}: {titel}", "",
         f"Video: {video_name} | Übung: {exercise} | gemeinsame Frames: {len(common)} | erstellt: {created}", "",
         "## Überblick", md_table(["", ra.label, rb.label], [
             ["Frames (gesamt)", stats[0]["frames"], stats[1]["frames"]],
             ["Dauer [s]", fmt(stats[0]["dauer_s"], 1), fmt(stats[1]["dauer_s"], 1)],
             ["Pose erkannt [%]", fmt(stats[0]["erkannt_pct"], 1), fmt(stats[1]["erkannt_pct"], 1)],
             ["Latenz Mittel / 95. Perz. [ms]", f"{stats[0]['latenz_mean_ms']:.0f} / {stats[0]['latenz_p95_ms']:.0f}", f"{stats[1]['latenz_mean_ms']:.0f} / {stats[1]['latenz_p95_ms']:.0f}"],
             ["Gelenke unter Sichtbarkeit 0.5 [%]", fmt(stats[0]["vis_unter_schwelle_pct"], 1), fmt(stats[1]["vis_unter_schwelle_pct"], 1)],
             ["Jitter Ø (RMS 2. Diff., relevante Winkel) [°/Frame²]", fmt(stats[0]["jitter_mean"], 1), fmt(stats[1]["jitter_mean"], 1)],
             [f"Winkelsprünge > {SPIKE_DEG:.0f}°/Frame (relevante Winkel)", stats[0]["spikes"], stats[1]["spikes"]],
             ["Ø |Segmentabweichung| zu gemessen [%]", fmt(stats[0]["seg_abw_mean_pct"], 1), fmt(stats[1]["seg_abw_mean_pct"], 1)],
             ["Wiederholungen erkannt", len(stats[0]["rep_idx"]), len(stats[1]["rep_idx"])]]), "",
         f"## Übereinstimmung ({rb.label} minus {ra.label}, rohe Winkel)", md_table(["Winkel", "n", "Bias [°]", "MAE [°]", "RMSE [°]", "Korrelation"], ag_rows), ""]
    if rep_tab:
        hdr = ["Modell", "Nr.", "Frame", "Zeit [s]", f"Spitzenwert ({rep_col[3]})", "bis nächste [s]"]
        if any(r.has_band for r in runs):
            hdr.append("Dehnung an Spitze")
        if exercise == "reverse_fly_jacks":
            hdr.append("Ellbogen Mittel an Spitze")
        L += ["## Wiederholungen", md_table(hdr, rep_tab), ""] + [f"- {x}" for x in extra_lines] + [""]
    if band_corr:
        L += ["## Band", f"Bandlänge gegen Handabstand aus der Pose (Korrelation): " + ", ".join(f"{NICE.get(k, k)} {v:.2f}" for k, v in band_corr.items()),
              "Die Bandmessung ist modellunabhängig; eine niedrige Korrelation deutet auf Verdeckung, Fehlerkennung oder ein Band, das nicht zwischen den Händen liegt.", ""]
    seg_rows = [[n, f"{L_:.3f}"] + [f"{st['segmente'][n][0]:.3f} ({st['segmente'][n][1]:+.1f} %)" if np.isfinite(st['segmente'][n][0]) else "–" for st in stats]
                for n, _, _, L_ in SEGMENTS]
    L += ["## Segmentlängen gegen Messung", md_table(["Segment", "gemessen [m]", ra.label, rb.label], seg_rows), "",
          "## Hinweise",
          "- Ohne Referenzmessung (Goniometer, Marker) beschreibt dieser Bericht Übereinstimmung und Stabilität, nicht Genauigkeit.",
          "- Bias, MAE und Korrelation gelten nur für diese Aufnahme (eine Person, eine Kameraposition).",
          "- Winkelsprünge bei InstantHMR hängen mit `detector_stride` zusammen (siehe Auswertung vom 26.10.); Vergleich mit `--detector-stride 1` prüfen.",
          "- Hüftbreite weicht bei beiden Modellen stark ab, vermutlich Definitionsunterschied (Gelenkzentrum gegen gemessene Breite).", "",
          "## Dateien", "01_verlauf.png, 02_uebereinstimmung.png, " + ("03_band.png, " if band_runs else "") + "04_qualitaet.png, metrics.csv" + (", vergleich.mp4" if (make_video and video) else "")]
    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    return out_dir, stats, ag_all, band_corr, sync


# ---------------------------------------------------------------- Auswahl und Start
def guess_exercise(label):
    s = label.lower()
    if "squat" in s or "kniebeuge" in s:
        return "squat"
    if "fly" in s or "jack" in s:
        return "reverse_fly_jacks"
    if "zug" in s or "row" in s or "pull" in s:
        return "seitzug"
    return ""


def scan_csvs(folder):
    items = []
    for p in sorted(Path(folder).glob("*.csv"), key=lambda q: q.stat().st_mtime, reverse=True):
        stem = p.stem.lower()
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
    ex = guess_exercise(Run(args.mp).exercise) or cfg.get("analyse_exercise", "reverse_fly_jacks")
    opts = list(PROFILES)
    print("\nÜbung: " + ", ".join(f"{i + 1} = {o}" for i, o in enumerate(opts)))
    v = ask("Übung (Nummer oder Name)", ex)
    args.exercise = opts[int(v) - 1] if v.isdigit() and 1 <= int(v) <= len(opts) else (v if v in PROFILES else ex)
    meta_src = Run(args.ih).source or Run(args.mp).source
    vid = ask("Originalvideo für das Vergleichsvideo (Enter = aus der Meta-Datei, '-' = kein Video)", meta_src)
    args.no_video = vid.strip() == "-"
    args.video = None if args.no_video else clean_path(vid)
    save_cfg(analyse_exercise=args.exercise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp", help="CSV von run_mediapipe")
    ap.add_argument("--ih", help="CSV von run_instanthmr")
    ap.add_argument("--exercise", default=None, choices=list(PROFILES))
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
    out_dir, stats, ag, bc, sync = analyse(ra, rb, args.exercise, out_dir, video, not args.no_video)
    print(f"\nFertig. Ordner: {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")
    print(f"Sammeldatei: {out_dir.parent / 'all_runs.csv'}")


if __name__ == "__main__":
    main()
