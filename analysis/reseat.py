"""Re-seat drift (issue #14): do templates from some wears of the AirPods work on another wear?
Takes are grouped into wears by createdAt gaps over --gap seconds; a take's own "wear" field wins if present.
DTW-1NN scores: leave-one-wear-out (train on every other wear), within-wear leave-one-out, and chance by
label shuffle. Each with two cheap fixes on and off: per-wear z-score of each channel, and a gravity-aligned
frame (rotate accel and rotation rate so every wear's mean gx,gy,gz points the same way). Both use the held-out
wear's own takes but never its labels; global z is fit on each fold's training wears only, like the app's
templates. Features are accel + rotation rate, the vectors a re-seat rotates.
Usage: python3 analysis/reseat.py [--dataset PATH] [--since CREATED_AT] [--gap SECONDS] [--results analysis/RESULTS.md]
"""
import pathlib, sys
import numpy as np
from clean_test_lib import parse_args, load_takes, raw, energy, smooth, trim_idx, dtw, dmat, rotvec, turn, ACC, ROT

CONFIGS = [("none (global z, like the app)", False, False), ("per-wear z", True, False),
           ("gravity frame", False, True), ("per-wear z + gravity frame", True, True)]

def wears(T, gap=180.0):
    """Wear number per take (T sorted by createdAt): a new wear after a gap over `gap` s, unless the take has "wear"."""
    c = np.array([t["createdAt"] for t in T]); g = np.concatenate([[0], np.cumsum(np.diff(c) > gap)])
    ids = {}
    return np.array([ids.setdefault(("wear", t["wear"]) if "wear" in t else ("gap", k), len(ids)) for t, k in zip(T, g)])

def feats(t):
    """Accel + rotation rate, trimmed like the app, per-take mean removed."""
    x = raw(t, ACC + ROT); lo, hi = trim_idx(smooth(energy(t)[:, None], 3)[:, 0], 0.15, 2)
    x = x[lo:hi]
    return x - x.mean(0)

def rot_to(a, b):
    """Smallest rotation taking direction a onto direction b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b); v = np.cross(a, b); s = np.linalg.norm(v)
    return rotvec(v / s * np.arctan2(s, a @ b)) if s > 1e-12 else np.eye(3)

DOWN = np.array([0.0, 0.0, -1.0])

def gravity_frame(X, G, wear):
    """Rotate each wear so its mean gravity points straight down (a fixed direction, so no wear's frame
    depends on another's). Undoes a tilt of the AirPods; a turn about the gravity axis leaves gravity
    unchanged, so this cannot see or fix it."""
    R = {w: rot_to(G[wear == w].mean(0), DOWN) for w in set(wear)}
    return [turn(x, R[w]) for x, w in zip(X, wear)]

def features(T, wear, perwear, gravity):
    """Unscaled features per take, and per take the channel std it is divided by: its own wear's (per-wear z),
    or None for global z, which fold_sd fits on the training takes."""
    X = [feats(t) for t in T]
    if gravity: X = gravity_frame(X, np.array([raw(t, ["gx", "gy", "gz"]).mean(0) for t in T]), wear)
    if not perwear: return X, None
    sd = np.zeros((len(X), X[0].shape[1]))
    for w in set(wear):
        sd[wear == w] = np.concatenate([x for x, m in zip(X, wear == w) if m]).std(0) + 1e-9
    return X, sd

def fold_sd(X, sd, train):
    """sd as given (per-wear z), or for global z the std of the `train` takes, applied to every take."""
    if sd is not None: return sd
    return np.tile(np.concatenate([X[i] for i in train]).std(0) + 1e-9, (len(X), 1))

def lowo_nn(X, sd, wear):
    """Leave-one-wear-out: index of each take's nearest neighbour among the other wears' takes, with global z
    fit on those training wears only."""
    nn = np.zeros(len(X), int)
    for w in sorted(set(wear)):
        tr = np.where(wear != w)[0]; s = fold_sd(X, sd, tr); A = [X[i] / s[i] for i in tr]
        for j in np.where(wear == w)[0]:
            nn[j] = tr[np.argmin([dtw(X[j] / s[j], a, 10) for a in A])]
    return nn

def nearest(M, allowed):
    """Index of each take's nearest neighbour among the takes allowed for it (never itself)."""
    D = np.where(allowed, M, np.inf); np.fill_diagonal(D, np.inf)
    return D.argmin(1)

def pct(hits):
    return f"{hits.mean():.0%} ({hits.sum()}/{len(hits)})"

def table(T, y, wear, shuffles=1000):
    same = wear[:, None] == wear[None, :]
    alone = same.sum(1) == 1  # no other take in its wear, so no within-wear score
    rows = ["| fix | leave-one-wear-out | by held-out wear | within-wear leave-one-out | chance (shuffled labels) | p vs chance |",
            "|---|---|---|---|---|---|"]
    for name, perwear, gravity in CONFIGS:
        X, sd = features(T, wear, perwear, gravity)
        nn = lowo_nn(X, sd, wear); hit = y[nn] == y
        M = dmat([x / s for x, s in zip(X, fold_sd(X, sd, range(len(X))))])
        within = (y[nearest(M, same)] == y)[~alone]
        rng = np.random.default_rng(0); null = np.array([np.mean(s[nn] == s) for s in (rng.permutation(y) for _ in range(shuffles))])
        p = (np.sum(null >= hit.mean()) + 1) / (shuffles + 1)
        byw = ", ".join(f"{hit[wear == w].sum()}/{np.sum(wear == w)}" for w in sorted(set(wear)))
        rows.append(f"| {name} | {pct(hit)} | {byw} | {pct(within)} | {null.mean():.0%} | {p:.3f} |")
    return "\n".join(rows)

if __name__ == "__main__":
    def options(p):
        p.add_argument("--gap", type=float, default=180.0, help="seconds between takes that start a new wear (default: %(default)s)")
        p.add_argument("--results", type=pathlib.Path, help="also append the output to this markdown file, e.g. analysis/RESULTS.md")
    args = parse_args(__doc__, extra=options)
    ALL, _ = load_takes(args.dataset)
    T = [t for t in ALL if args.since is None or t["createdAt"] > args.since]
    words = sorted({t["label"] for t in T}); y = np.array([words.index(t["label"]) for t in T]); wear = wears(T, args.gap)
    if len(set(wear)) < 2: sys.exit(f"only one wear (no createdAt gap over {args.gap:g} s and no wear field): nothing to hold out")
    head = (f"{len(T)} takes, words {words}, {len(set(wear))} wears of {np.bincount(wear).tolist()} takes "
            f"(gap > {args.gap:g} s or wear field). DTW-1NN on accel + rotation rate.")
    out = f"{head}\n\n{table(T, y, wear)}"
    print(out)
    if args.results:
        with open(args.results, "a") as f: f.write(f"\n### `python3 {' '.join(sys.argv)}`\n\n{out}\n")
