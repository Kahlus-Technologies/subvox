"""Render the README hero from a real Subvox session (AirPods Pro head motion).

Dark field, wordmark, and three real traces from one recording: user acceleration,
rotation rate and pitch. The session and window are printed in the corner so the
figure can be checked. Nothing is synthetic.

Usage: python scripts/make_readme_hero.py ~/Documents/Subvox/Sessions/<session> --out docs/banner.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BG, INK, DIM, ACCENT = "#0B0B0C", "#F5F5F5", "#5C5C60", "#7CFFD4"
WIN_S = 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/banner.png"))
    a = ap.parse_args()
    m = pd.read_csv(a.session / "motion.csv")
    t = m.t.to_numpy()
    acc = np.linalg.norm(m[["ax", "ay", "az"]].to_numpy(), axis=1)
    rot = np.linalg.norm(m[["rx", "ry", "rz"]].to_numpy(), axis=1)
    pitch = m.pitch.to_numpy()
    # Window with the most rotation energy, so the trace shows real jaw/head movement.
    fs = 1 / np.median(np.diff(t))
    n = int(WIN_S * fs)
    e = np.convolve(rot**2, np.ones(n), "valid")
    skip = int(5 * fs)  # the first seconds hold the insertion transient
    lo = skip + int(e[skip:].argmax())
    sl = slice(lo, lo + n)
    tt = t[sl] - t[lo]

    norm = lambda x: (x - np.median(x)) / (np.percentile(np.abs(x - np.median(x)), 99) + 1e-9)
    fig = plt.figure(figsize=(16, 6), dpi=200, facecolor=BG)
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.58], facecolor=BG)
    ax.plot(tt, np.clip(norm(acc[sl]), -1.4, 1.4) * 0.5 + 2.2, color=INK, lw=0.7, alpha=0.95)
    ax.plot(tt, np.clip(norm(rot[sl]), -1.4, 1.4) * 0.5 + 0.95, color=INK, lw=0.7, alpha=0.6)
    ax.plot(tt, np.clip(norm(pitch[sl]), -1.4, 1.4) * 0.3 - 0.25, color=ACCENT, lw=1.4)
    for y, label in [(2.95, "USER ACCELERATION"), (1.7, "ROTATION RATE"), (0.4, "PITCH")]:
        ax.text(0.3, y, label, color=DIM, fontsize=9, family="monospace")
    ax.text(WIN_S - 0.3, -0.95, f"AirPods Pro  ·  real recording  ·  {a.session.name}  ·  {fs:.0f} Hz",
            color=DIM, fontsize=8, family="monospace", ha="right")
    ax.set_xlim(0, WIN_S)
    ax.set_ylim(-1.05, 3.15)
    ax.axis("off")
    fig.text(0.5, 0.80, "SUBVOX", color=INK, ha="center", va="center", fontsize=78, fontweight="bold",
             family="sans-serif")
    fig.text(0.5, 0.655, "LISTEN  ·  MOUTH  ·  DECODE", color=DIM, ha="center", fontsize=11, family="monospace")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor=BG)
    print(a.out, f"window {t[lo]:.1f}s..{t[lo]+WIN_S:.1f}s")


main()
