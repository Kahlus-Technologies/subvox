"""Real-data check of plan step 3: train on wear 0, test on wear 1 (AirPods re-seated). DTW-1NN like the app."""
import json, sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from clean_test_lib import dtw, trim_idx, smooth, energy, raw
d = json.load(open("/Users/aayu/Library/Application Support/Subvox/dataset.json"))
T = sorted(d["templates"], key=lambda t: t["createdAt"])
c = np.array([t["createdAt"] for t in T]); wear = np.concatenate([[0], np.cumsum(np.diff(c) > 180)])
words = sorted({t["label"] for t in T}); y = np.array([words.index(t["label"]) for t in T])
KEYS = ["ax", "ay", "az", "rx", "ry", "rz", "pitch", "roll", "yaw"]
G = [raw(t, ["gx", "gy", "gz"]).mean(0) for t in T]

def feats(t, gfix):
    x = raw(t, KEYS)
    lo, hi = trim_idx(smooth(energy(t)[:, None], 3)[:, 0], 0.15, 2); x = x[lo:hi].copy()
    x[:, 6:] -= x[0, 6:]
    if gfix is not None:  # rotate accel and rotation into a frame where this wear's mean gravity is fixed
        x[:, 0:3] = x[:, 0:3] @ gfix.T; x[:, 3:6] = x[:, 3:6] @ gfix.T
    return x - x.mean(0)

def rot_to(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b); v = np.cross(a, b); s = np.linalg.norm(v); cth = a @ b
    if s < 1e-9: return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
    th = np.arctan2(s, cth); return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

def run(perwear_z, gravity, aug, seed=0):
    ref = np.mean([G[i] for i in range(len(T)) if wear[i] == 0], 0)
    gw = {w: np.mean([G[i] for i in range(len(T)) if wear[i] == w], 0) for w in set(wear)}
    X = [feats(t, rot_to(gw[wear[i]], ref) if gravity else None) for i, t in enumerate(T)]
    if perwear_z:
        for w in set(wear):
            idx = [i for i in range(len(T)) if wear[i] == w]; sd = np.concatenate([X[i] for i in idx]).std(0) + 1e-9
            for i in idx: X[i] = X[i] / sd
    else:
        sd = np.concatenate(X).std(0) + 1e-9; X = [x / sd for x in X]
    tr = [i for i in range(len(T)) if wear[i] == 0]; te = [i for i in range(len(T)) if wear[i] == 1]
    trX = [X[i] for i in tr]; trY = list(y[tr])
    if aug:
        rng = np.random.default_rng(seed)
        for i in tr:
            for _ in range(2):
                ang = rng.normal(0, np.radians(8), 3); R = rot_to(np.array([0, 0, 1.0]), np.array([0, 0, 1.0]))
                for k, a in enumerate(ang):
                    ax = np.zeros(3); ax[k] = 1; K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
                    R = R @ (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K)
                x = X[i].copy(); x[:, 0:3] = x[:, 0:3] @ R.T; x[:, 3:6] = x[:, 3:6] @ R.T
                n = len(x); m = max(5, int(n * rng.uniform(0.85, 1.15))); p = np.linspace(0, n - 1, m)
                x = np.stack([np.interp(p, np.arange(n), x[:, j]) for j in range(9)], 1) * rng.uniform(0.85, 1.15)
                trX.append(x); trY.append(y[i])
    pred = [trY[int(np.argmin([dtw(X[j], a, 10) for a in trX]))] for j in te]
    return np.mean(np.array(pred) == y[te]), len(te)

def chance(n=200):
    rng = np.random.default_rng(1); te = y[wear == 1]; return np.mean([np.mean(rng.permutation(te) == te) for _ in range(n)])

print(f"wear0 train={np.sum(wear==0)} wear1 test={np.sum(wear==1)} words={words} chance~{chance():.2f}")
for name, cfg in [("baseline (app-style global z)", (False, False, False)), ("+ per-wear z-score", (True, False, False)),
                  ("+ gravity-aligned frame", (False, True, False)), ("+ augmentation", (False, False, True)),
                  ("per-wear z + gravity", (True, True, False)), ("all three", (True, True, True))]:
    acc, n = run(*cfg); print(f"{name:32s} {acc:.1%}  ({int(round(acc*n))}/{n})")

# Fair reference: the SAME 172 training takes (wear 0 minus its last 29), tested on wear 0's last 29 vs on wear 1.
# Comparing the 201-take baseline above to a 172-take same-wear split would credit re-seating for extra training data.
X = [feats(t, None) for t in T]; sd = np.concatenate(X).std(0) + 1e-9; X = [x / sd for x in X]
w0 = [i for i in range(len(T)) if wear[i] == 0]; w1 = [i for i in range(len(T)) if wear[i] == 1]; tr = w0[:-29]
hit = lambda te: np.array([y[tr][int(np.argmin([dtw(X[j], X[i], 10) for i in tr]))] == y[j] for j in te])
same, cross = hit(w0[-29:]), hit(w1); rng = np.random.default_rng(0)
diff = [rng.choice(cross, 29).mean() - rng.choice(same, 29).mean() for _ in range(5000)]
print(f"matched 172-take train: same-wear {same.mean():.1%}, re-seated {cross.mean():.1%}, "
      f"diff {cross.mean() - same.mean():+.1%} (95% bootstrap CI {np.percentile(diff, 2.5):+.0%} to {np.percentile(diff, 97.5):+.0%})")
print("wear sizes by gap threshold:", {s: np.bincount(np.concatenate([[0], np.cumsum(np.diff(c) > s)])).tolist() for s in (120, 180)})
from math import comb
p = sum(comb(29, k) * 0.2**k * 0.8**(29 - k) for k in range(20, 30)); print(f"P(>=20/29 by chance at 20%) = {p:.1e}")
