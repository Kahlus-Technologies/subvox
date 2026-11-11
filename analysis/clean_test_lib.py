"""Shared helpers for the analysis scripts: dataset path and --since args, trimming, smoothing, DTW, scoring.
Needs numpy and numba: pip install -r analysis/requirements.txt
"""
import argparse, json, pathlib
import numpy as np
from numba import njit

DEFAULT_DATASET = pathlib.Path.home() / "Library/Application Support/Subvox/dataset.json"
DRILL_SINCE = pathlib.Path(__file__).resolve().parent.parent / ".drill_since"
ROT = ["rx", "ry", "rz"]; ACC = ["ax", "ay", "az"]

def parse_args(description, drill_since_file=False):
    """--dataset PATH (default: the app's dataset.json) and --since CREATED_AT (where drill takes start).
    With drill_since_file, a missing --since is read from .drill_since at the repo root if it exists."""
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=pathlib.Path, default=DEFAULT_DATASET, help="Subvox dataset.json (default: %(default)s)")
    p.add_argument("--since", type=float, help="createdAt of the first drill take (seconds since 2001, as in dataset.json)")
    args = p.parse_args()
    if args.since is None and drill_since_file and DRILL_SINCE.exists():
        args.since = float(DRILL_SINCE.read_text())
    return args

def load_takes(path):
    """Every take sorted by createdAt, and the sorted word list."""
    takes = sorted(json.load(open(path))["templates"], key=lambda t: t["createdAt"])
    return takes, sorted({t["label"] for t in takes})

def raw(t, keys):
    return np.array([[s[k] for k in keys] for s in t["samples"]], float)

def energy(t):
    r = raw(t, ROT); a = raw(t, ACC)
    return np.linalg.norm(r, axis=1) + 5 * np.linalg.norm(a, axis=1)

def trim_idx(e, frac, pad, floor_q=0.2):
    if len(e) <= 6: return 0, len(e)
    s = np.sort(e); fl = s[int(len(s) * floor_q)]; pk = s[-1]
    if pk <= fl * 1.5: return 0, len(e)
    th = fl + frac * (pk - fl); idx = np.where(e > th)[0]
    if len(idx) == 0: return 0, len(e)
    lo, hi = max(0, idx[0] - pad), min(len(e), idx[-1] + pad + 1)
    return (lo, hi) if hi - lo >= 5 else (0, len(e))

def smooth(x, k):
    if k <= 1: return x
    w = np.ones(k) / k
    return np.stack([np.convolve(x[:, i], w, "same") for i in range(x.shape[1])], 1)

def make(T, keys, frac=0.15, pad=2, sm=1, resample=None, znorm=False):
    out = []
    for t in T:
        x = raw(t, keys)
        if frac is not None:
            lo, hi = trim_idx(smooth(energy(t)[:, None], 3)[:, 0], frac, pad); x = x[lo:hi]
        x = smooth(x, sm)
        x = x - x.mean(0)
        if znorm: x = x / (x.std(0) + 1e-6)  # per-take: removes loud vs soft mouthing differences
        if resample:
            p = np.linspace(0, len(x) - 1, resample); lo_ = np.floor(p).astype(int); hi_ = np.minimum(lo_ + 1, len(x) - 1)
            w = (p - lo_)[:, None]; x = x[lo_] * (1 - w) + x[hi_] * w
        out.append(x)
    sd = np.concatenate(out).std(0) + 1e-9
    return [x / sd for x in out]

@njit(cache=True)
def dtw(a, b, band):
    n, m = len(a), len(b); D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0.0
    w = max(band, abs(n - m))
    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m, i + w) + 1):
            c = np.sqrt(((a[i - 1] - b[j - 1]) ** 2).sum())
            D[i, j] = c + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    return D[n, m] / (n + m)

def dmat(X):
    n = len(X); M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n): M[i, j] = M[j, i] = dtw(X[i], X[j], 10)
    return M

def scores(T, X):
    words = sorted({t["label"] for t in T})
    y = np.array([words.index(t["label"]) for t in T]); M = dmat(X)
    L = M.copy(); np.fill_diagonal(L, np.inf); loo = np.mean(y[L.argmin(1)] == y)
    # within-word half split (first half of each word's takes vs second half), both directions
    half = np.zeros(len(y), bool)
    for w in set(y):
        idx = np.where(y == w)[0]; half[idx[: len(idx) // 2]] = True
    a = np.mean(y[half][M[~half][:, half].argmin(1)] == y[~half]); b = np.mean(y[~half][M[half][:, ~half].argmin(1)] == y[half])
    return loo, (a + b) / 2

