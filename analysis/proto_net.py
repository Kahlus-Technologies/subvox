"""Prototypical 1D CNN embedding vs DTW-1NN (issue #18, plan step 4 in docs/research/prior-work-and-training-plan.md).
A 1D CNN with squeeze-excitation blocks maps a take's (T, 9) motion at 50 Hz (ax ay az rx ry rz pitch roll yaw) to a
32-number embedding. It trains with a prototypical loss over episodes of training takes, each a fresh augmented copy
(all four augment.py steps, #17). Enrolment: a word's prototype is the mean embedding of 5 of its takes. A take gets
the nearest prototype's word, and is rejected when the margin 1 - best/second-best distance is under --margin (the
app's confidence, default its 0.05). DTW-1NN (clean_test_lib.dtw) uses the same 5 takes per word as templates.
Both are scored on the same splits:
  leave-one-wear-out: hold out one wear of one person. Train on every other take, enrol from that person's
    other wears (first 5 takes per word by createdAt), test on the held-out wear.
  leave-one-person-out: hold out one person. Train on everyone else, enrol from the person's first 5 takes per
    word, test on the rest of their takes.
Features for both: all 9 channels trimmed like the app, attitude relative to the take's first sample, per-take mean
removed, then reseat.py's best fix (gravity frame and per-wear z). Chance shuffles the labels like reseat.py, with
the trained nets and DTW distances fixed.
People: one --dataset file per person, since the app's dataset.json holds one user's takes. Wears inside a file
come from reseat.wears. Needs torch (pip install -r analysis/requirements-ml.txt). CPU only, seeded.
Usage: python3 analysis/proto_net.py --dataset A.json [--dataset B.json ...] [--episodes N] [--margin M]
                                     [--no-augment] [--results analysis/RESULTS.md]
"""
import argparse, pathlib, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clean_test_lib import DEFAULT_DATASET, ACC, ROT, load_takes, raw, energy, smooth, trim_idx, dtw
from augment import augment, STEPS
import reseat

ATT = ["pitch", "roll", "yaw"]; GRAV = ["gx", "gy", "gz"]
LENGTH = 100  # samples in the net's window, 2 s at 50 Hz: shorter takes are centred in zeros, longer ones cropped
SHOT, QUERY, WAY = 5, 3, 10  # enrolment takes per word; per episode, query takes per word and most words

def feats(t):
    """All 9 channels trimmed like the app, attitude relative to the first sample and wrapped like the app, mean removed."""
    x = raw(t, ACC + ROT + ATT); lo, hi = trim_idx(smooth(energy(t)[:, None], 3)[:, 0], 0.15, 2); x = x[lo:hi]
    x[:, 6:] = np.angle(np.exp(1j * (x[:, 6:] - x[0, 6:])))
    return x - x.mean(0)

def features(T, wear):
    """Gravity-framed features in sensor units, and the per-wear channel std each take is divided by."""
    X = reseat.gravity_frame([feats(t) for t in T], np.array([raw(t, GRAV).mean(0) for t in T]), wear)
    sd = np.zeros((len(X), 9))
    for w in set(wear): sd[wear == w] = np.concatenate([x for x, k in zip(X, wear) if k == w]).std(0) + 1e-9
    return X, sd

def window(x, n=LENGTH):
    """Centre x in n zero samples (zero is the take's mean: a still head), or keep its middle n."""
    out = np.zeros((n, x.shape[1]), np.float32); k = min(n, len(x)); a, b = (n - k) // 2, (len(x) - k) // 2
    out[a:a + k] = x[b:b + k]
    return out

class SE(nn.Module):
    """Squeeze-excitation: scale each channel by a gate computed from all channels' means over time."""
    def __init__(self, c, r=4):
        super().__init__(); self.gate = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(), nn.Linear(c // r, c), nn.Sigmoid())
    def forward(self, x): return x * self.gate(x.mean(2))[:, :, None]

class Block(nn.Module):
    """Residual block: two width-5 convolutions and SE, plus a shortcut."""
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv1d(cin, cout, 5, stride, 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(),
                                  nn.Conv1d(cout, cout, 5, 1, 2, bias=False), nn.BatchNorm1d(cout), SE(cout))
        self.skip = (nn.Identity() if cin == cout and stride == 1 else
                     nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout)))
    def forward(self, x): return torch.relu(self.body(x) + self.skip(x))

def make_net(dim=32):
    """(batch, 9, time) to (batch, dim)."""
    return nn.Sequential(nn.Conv1d(9, 32, 5, 1, 2, bias=False), nn.BatchNorm1d(32), nn.ReLU(),
                         Block(32, 32), Block(32, 48, 2), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(48, dim))

def n_params(net):
    return sum(p.numel() for p in net.parameters())

def train(X, sd, y, idx, seed, episodes, steps=tuple(STEPS)):
    """Prototypical training on takes idx. An episode draws up to WAY words and SHOT support + QUERY query takes of
    each (with replacement if a word has fewer), augments each in sensor units and divides by its take's std."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = make_net(); opt = torch.optim.Adam(net.parameters(), 3e-3, weight_decay=1e-4)
    by = {c: idx[y[idx] == c] for c in np.unique(y[idx])}; n = SHOT + QUERY
    for _ in range(episodes):
        words = rng.choice(list(by), min(WAY, len(by)), replace=False)
        take = [i for c in words for i in rng.choice(by[c], n, replace=len(by[c]) < n)]
        z = net(torch.from_numpy(np.stack([window(augment(X[i], rng, steps) / sd[i]) for i in take])).transpose(1, 2))
        z = z.view(len(words), n, -1)
        logits = -torch.cdist(z[:, SHOT:].reshape(-1, z.shape[-1]), z[:, :SHOT].mean(1)) ** 2
        loss = F.cross_entropy(logits, torch.arange(len(words)).repeat_interleave(QUERY))
        opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()

def embed(net, Z):
    with torch.no_grad(): return net(torch.from_numpy(np.stack([window(z) for z in Z])).transpose(1, 2)).numpy()

def decide(D, labels):
    """Distances D (queries x enrolment items with `labels`): each query's word (nearest item) and its margin
    1 - best/second-best over words, as TemplateClassifier.margin in the app."""
    words = np.unique(labels); C = np.stack([D[:, labels == c].min(1) for c in words], 1); o = np.sort(C, 1)
    margin = 1 - o[:, 0] / np.maximum(o[:, 1], 1e-12) if len(words) > 1 else np.zeros(len(D))
    return words[C.argmin(1)], margin

def dtw_rule(Z, en, q):
    """DTW-1NN against the enrolment takes en. Returns labels -> (word, margin) per query, distances computed once."""
    D = np.array([[dtw(Z[i], Z[j], 10) for j in en] for i in q])
    return lambda lab: decide(D, lab[en])

def proto_rule(E, en, q):
    """Nearest prototype: each word's prototype is the mean embedding of its enrolment takes."""
    def rule(lab):
        words = np.unique(lab[en]); P = np.stack([E[en][lab[en] == c].mean(0) for c in words])
        return decide(np.linalg.norm(E[q][:, None] - P[None], axis=2), words)
    return rule

def load(paths, gap):
    """Every file's takes (one file per person, each sorted by createdAt), the person and wear of each take
    (wear numbered across people), the wear within its person, and the sorted word list."""
    T, who, wear, local = [], [], [], []
    for p, path in enumerate(paths):
        t, _ = load_takes(path); w = reseat.wears(t, gap)
        T += t; who += [p] * len(t); local += list(w); wear += list(w + (max(wear) + 1 if wear else 0))
    words = sorted({t["label"] for t in T})
    return T, np.array([words.index(t["label"]) for t in T]), np.array(who), np.array(wear), np.array(local), words

def folds(y, who, wear, local):
    """(split, held-out name, train, enrol, test) index arrays. Takes are in createdAt order within a person."""
    first = lambda idx: np.concatenate([idx[y[idx] == c][:SHOT] for c in np.unique(y[idx])] or [np.zeros(0, int)])
    out = []
    for w in np.unique(wear):
        q = np.where(wear == w)[0]; p = who[q[0]]; en = first(np.where((who == p) & (wear != w))[0])
        if len(en): out.append(("leave-one-wear-out", f"{chr(65 + p)}{local[q[0]]}", np.where(wear != w)[0], en, q))
    if len(set(who)) > 1:
        for p in np.unique(who):
            mine = np.where(who == p)[0]; en = first(mine)
            out.append(("leave-one-person-out", chr(65 + p), np.where(who != p)[0], en, np.setdiff1d(mine, en)))
    return out

def pct(hits):
    return f"{hits.mean():.0%} ({hits.sum()}/{len(hits)})" if len(hits) else "-"

def row(split, method, fr, y, margin, shuffles):
    """One table row from [(held-out name, test idx, rule)]. Chance: the same label shuffles for every method."""
    hit, keep, byf = [], [], []
    for name, q, rule in fr:
        pred, m = rule(y); h = pred == y[q]; hit.append(h); keep.append(m >= margin); byf.append(f"{name} {h.sum()}/{len(h)}")
    hit, keep = np.concatenate(hit), np.concatenate(keep)
    rng = np.random.default_rng(0); null = np.zeros(shuffles)
    for k in range(shuffles):
        s = rng.permutation(y); null[k] = np.mean(np.concatenate([rule(s)[0] == s[q] for _, q, rule in fr]))
    p = (np.sum(null >= hit.mean()) + 1) / (shuffles + 1)
    return (f"| {split} | {method} | {pct(hit)} | {', '.join(byf)} | {1 - keep.mean():.0%} | {pct(hit[keep])} "
            f"| {null.mean():.0%} | {p:.3f} |")

DTW_ROW, PROTO_ROW = f"DTW-1NN, {SHOT} takes per word", f"proto-net, {SHOT}-take prototypes"
SPLITS = ("leave-one-wear-out", "leave-one-person-out")

def methods(T, y, who, wear, local, episodes, steps, seed=0, splits=SPLITS):
    """{split: [(held-out name, train, enrol, test, {method: rule})]} for DTW-1NN and the proto-net. Fold k's net uses
    seed + k, so one split's numbers do not depend on whether the other runs. Training times go to stderr."""
    X, sd = features(T, wear); Z = [x / s for x, s in zip(X, sd)]; got = {}
    for k, (split, name, tr, en, q) in enumerate(folds(y, who, wear, local)):
        if split not in splits: continue
        t0 = time.time(); E = embed(train(X, sd, y, tr, seed + k, episodes, steps), Z)
        print(f"{split} {name}: trained on {len(tr)} takes in {time.time() - t0:.1f} s", file=sys.stderr)
        got.setdefault(split, []).append((name, tr, en, q, {DTW_ROW: dtw_rule(Z, en, q), PROTO_ROW: proto_rule(E, en, q)}))
    return got

def table(got, y, margin, shuffles):
    """One row per split and method."""
    rows = ["| split | method | accuracy | by held-out wear or person | rejected (margin < %g) | accuracy on the rest "
            "| chance (shuffled labels) | p vs chance |" % margin, "|---|---|---|---|---|---|---|---|"]
    for split, fr in got.items():
        rows += [row(split, m, [(n, q, r[m]) for n, _, _, q, r in fr], y, margin, shuffles) for m in fr[0][-1]]
    return "\n".join(rows)

def cli(doc, extra=None):
    """Options shared with ssl_probe.py; extra(parser) adds a script's own. Also makes torch deterministic."""
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=pathlib.Path, action="append",
                    help="one person's dataset.json; repeat once per person (default: the app's, %s)" % DEFAULT_DATASET)
    ap.add_argument("--gap", type=float, default=180.0, help="seconds between takes that start a new wear (default: %(default)s)")
    ap.add_argument("--episodes", type=int, default=300, help="training episodes per fold (default: %(default)s)")
    ap.add_argument("--margin", type=float, default=0.05, help="reject below this margin (default: %(default)s, the app's)")
    ap.add_argument("--shuffles", type=int, default=1000, help="label shuffles for chance (default: %(default)s)")
    ap.add_argument("--no-augment", action="store_true", help="train on the takes as they are")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", type=pathlib.Path, help="also append the output to this markdown file, e.g. analysis/RESULTS.md")
    if extra: extra(ap)
    args = ap.parse_args(); args.dataset = args.dataset or [DEFAULT_DATASET]
    args.steps = () if args.no_augment else tuple(STEPS)
    torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
    return args

def describe(args, T, words, who, wear):
    """The line above the table: data, features and proto-net settings."""
    people = "; ".join(f"{chr(65 + p)} = {path} ({np.sum(who == p)} takes, {len(set(wear[who == p]))} wears)"
                       for p, path in enumerate(args.dataset))
    return (f"{len(T)} takes, words {words}. People: {people}. Features: 9 channels, gravity frame + per-wear z. "
            f"Proto-net: {n_params(make_net()):,} parameters (1D CNN, 2 residual SE blocks, 32-number embedding), "
            f"{args.episodes} episodes per fold of up to {WAY} words x ({SHOT} support + {QUERY} query) takes, "
            f"augmentation: {', '.join(args.steps) or 'none'}, seed {args.seed}.")

def append(path, out):
    with open(path, "a") as f: f.write(f"\n### `python3 {' '.join(sys.argv)}`\n\n{out}\n")

if __name__ == "__main__":
    args = cli(__doc__); start = time.time()
    T, y, who, wear, local, words = load(args.dataset, args.gap)
    if len(set(wear)) < 2: sys.exit(f"one person with one wear (no createdAt gap over {args.gap:g} s, no wear field): nothing to hold out")
    head = describe(args, T, words, who, wear) + ("" if len(args.dataset) > 1 else " One person: no leave-one-person-out split.")
    out = f"{head}\n\n{table(methods(T, y, who, wear, local, args.episodes, args.steps, args.seed), y, args.margin, args.shuffles)}"
    print(out); print(f"total {time.time() - start:.0f} s", file=sys.stderr)
    if args.results: append(args.results, out)
