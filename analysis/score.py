"""Score Aayu's Subvox AirPods Pro recordings honestly.
1. Leave-one-out nearest-neighbour (DTW) with several feature sets, like Subvox's Evaluate.
2. Label shuffle test: what accuracy does luck give with the same method.
3. Time split: train on the first 60% of takes (by createdAt), test on the last 40%.
   This is closer to real use, since neighbouring takes recorded back to back look alike.
4. Head-motion control: does mere clip length or total head movement separate the words?
Usage: python3 analysis/score.py [--dataset PATH] [--since CREATED_AT]
"""
import json, random
import numpy as np
from numba import njit

from clean_test_lib import parse_args
args = parse_args(__doc__)
d = json.load(open(args.dataset))
ALL = sorted(d["templates"], key=lambda t: t["createdAt"])
# --since <createdAt> scores only takes recorded after that point (drill sessions).
T = [t for t in ALL if args.since is None or t["createdAt"] > args.since]
words = sorted({t["label"] for t in T})
y = np.array([words.index(t["label"]) for t in T])
ts0 = np.array([s["t"] for s in T[0]["samples"]])
rate = (len(ts0) - 1) / (ts0[-1] - ts0[0])
print(f"{len(T)} takes, words {words}, counts {np.bincount(y).tolist()}, ~{rate:.0f} samples/s, "
      f"clip length {np.median([len(t['samples']) for t in T]):.0f} samples median")

GROUPS = {
    "acceleration": ["ax", "ay", "az"],
    "rotation rate": ["rx", "ry", "rz"],
    "attitude change": ["pitch", "roll", "yaw"],
    "accel + rotation": ["ax", "ay", "az", "rx", "ry", "rz"],
    "all motion": ["ax", "ay", "az", "rx", "ry", "rz", "pitch", "roll", "yaw"],
}

def feats(t, keys):
    x = np.array([[s[k] for k in keys] for s in t["samples"]], float)
    for i, k in enumerate(keys):
        if k in ("pitch", "roll", "yaw"):
            x[:, i] -= x[0, i]  # change since segment start
    x -= x.mean(0)
    return x

@njit(cache=True)
def dtw(a, b, band):
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0.0
    w = max(band, abs(n - m))
    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m, i + w) + 1):
            c = np.sqrt(((a[i - 1] - b[j - 1]) ** 2).sum())
            D[i, j] = c + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    return D[n, m] / (n + m)

def dist_matrix(keys):
    X = [feats(t, keys) for t in T]
    sd = np.concatenate(X).std(0) + 1e-9
    X = [x / sd for x in X]
    n = len(X); M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = dtw(X[i], X[j], 10)
    return M

def loo(M, labels):
    M = M.copy(); np.fill_diagonal(M, np.inf)
    return np.mean(labels[M.argmin(1)] == labels)

def timesplit(M, labels, frac=0.6):
    n = len(labels); cut = int(n * frac)
    sub = M[cut:, :cut]
    return np.mean(labels[:cut][sub.argmin(1)] == labels[cut:])

rng = np.random.default_rng(0)
chance = max(np.bincount(y)) / len(y)
print(f"\nluck baseline: always guessing the most common word = {chance:.0%}, uniform = {1/len(words):.0%}\n")
print(f"{'features':18s} {'leave-one-out':>14s} {'shuffled labels':>16s} {'p':>7s} {'later takes':>12s}")
res = {}
for name, keys in GROUPS.items():
    M = dist_matrix(keys)
    acc = loo(M, y)
    null = np.array([loo(M, rng.permutation(y)) for _ in range(300)])
    p = (np.sum(null >= acc) + 1) / (len(null) + 1)
    ts = timesplit(M, y)
    res[name] = (acc, ts)
    print(f"{name:18s} {acc:>14.0%} {null.mean():>16.0%} {p:>7.3f} {ts:>12.0%}")

# controls: does trivial info separate words?
L = np.array([len(t["samples"]) for t in T], float)
E = np.array([np.abs(feats(t, GROUPS['all motion'])).sum() for t in T])
def loo1d(v):
    D = np.abs(v[:, None] - v[None, :]); np.fill_diagonal(D, np.inf); return np.mean(y[D.argmin(1)] == y)
print(f"\ncontrols (should be near luck if the signal is really jaw shape):")
print(f"  clip length only        {loo1d(L):.0%}")
print(f"  total movement only     {loo1d(E):.0%}")

best = max(res, key=lambda k: res[k][0]); M = dist_matrix(GROUPS[best]); np.fill_diagonal(M, np.inf)
pred = y[M.argmin(1)]
print(f"\nconfusion for best set ({best}), rows = what you said, cols = what it guessed:")
print("        " + " ".join(f"{w:>6s}" for w in words))
for i, w in enumerate(words):
    print(f"{w:>7s} " + " ".join(f"{np.sum((y == i) & (pred == j)):>6d}" for j in range(len(words))))

# Fairer test for block-recorded data: for each word, split its own takes into first half and second half
# (in time). Train on first halves of every word, test on second halves, and the reverse. Neighbouring takes
# of the same word still sit close in time, so this is still optimistic, but less so than leave-one-out.
print("\nfirst-half vs second-half of each word's own takes (still optimistic: same short session):")
for name in ("rotation rate", "accel + rotation", "all motion"):
    M = dist_matrix(GROUPS[name])
    order = np.arange(len(y)); half = np.zeros(len(y), bool)
    for w in range(len(words)):
        idx = order[y == w]; half[idx[: len(idx) // 2]] = True
    a = np.mean(y[half][M[~half][:, half].argmin(1)] == y[~half])
    b = np.mean(y[~half][M[half][:, ~half].argmin(1)] == y[half])
    print(f"  {name:18s} {(a + b) / 2:.0%}  (train first half -> test second {a:.0%}, reverse {b:.0%})")
# drift check: how far does one take sit from the next take of the same word vs from takes 10 later?
M = dist_matrix(GROUPS["rotation rate"])
near = [M[i, i + 1] for i in range(len(y) - 1) if y[i] == y[i + 1]]
far = [M[i, i + 10] for i in range(len(y) - 10) if y[i] == y[i + 10]]
other = [M[i, j] for i in range(len(y)) for j in range(len(y)) if y[i] != y[j]]
print(f"\naverage distance: next take same word {np.mean(near):.2f}, 10 takes later same word {np.mean(far):.2f}, different word {np.mean(other):.2f}")
