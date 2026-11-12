"""Linear probe of a pretrained wearable accelerometer encoder (issue #19, plan step 5 in
docs/research/prior-work-and-training-plan.md). Loads the OxWearables ssl-wearables ResNet (harnet5 by default)
from torch.hub with its self-supervised UK Biobank weights, freezes it, and fits logistic regression on its
embeddings. Scored leave-one-wear-out next to DTW-1NN and the proto-net from proto_net.py (#18), on the same folds
and seeds, so those two rows match proto_net.py's.
Encoder input: 3-axis acceleration in g at 30 Hz in 5 s windows (harnet5, 150 samples; harnet10 takes 10 s, 300).
Its training data were wrist accelerometers, which measure gravity too, so each take's input is its user
acceleration (gravity frame, as in proto_net.py) plus 1 g along reseat.DOWN, where the gravity frame turns every
wear's gravity. It is resampled from 50 to 30 Hz by linear interpolation and centred in a window of a still head
(gravity only). No per-wear z: the encoder expects g.
Two rows: logistic regression fitted on every training take of the fold (all people, the other wears), and the
nearest prototype on the frozen embeddings, from the same 5 enrolment takes as DTW and the proto-net. Chance
refits the logistic regression on every label shuffle (all shuffles at once).
Needs torch and, the first time, network access: torch.hub caches the code and weights in torch.hub.get_dir()
(~/.cache/torch/hub), outside the repo. The hub code imports only torch and numpy.
Usage: python3 analysis/ssl_probe.py --dataset A.json [--dataset B.json ...] [--model harnet5|harnet10]
                                     [--results analysis/RESULTS.md]
"""
import contextlib, sys, time
import numpy as np
import torch
import torch.nn.functional as nnf
import proto_net
import reseat

RATE, HZ = 50.0, 30.0  # AirPods motion rate, the encoder's rate
WINDOW = {"harnet5": 150, "harnet10": 300}  # encoder window in samples at 30 Hz

def encoder(name):
    """The pretrained feature extractor, frozen. The hub's "Weights loaded" line goes to stderr."""
    with contextlib.redirect_stdout(sys.stderr):
        m = torch.hub.load("OxWearables/ssl-wearables", name, class_num=2, pretrained=True, trust_repo=True)
    m = m.feature_extractor.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m

def accel(X, n):
    """Encoder input (takes, 3, n) from proto_net.features X: user accel plus 1 g of gravity along the direction the
    gravity frame turns every wear to, at 30 Hz, centred in n samples of a still head."""
    out = []
    for x in X:
        t = np.arange(0, (len(x) - 1) / RATE + 1e-9, 1 / HZ)
        a = np.stack([np.interp(t, np.arange(len(x)) / RATE, c) for c in x[:, :3].T], 1)
        out.append((proto_net.window(a, n) + reseat.DOWN).T)
    return np.array(out, np.float32)

def embed(m, A, batch=256):
    with torch.no_grad():
        return np.concatenate([m(torch.from_numpy(A[i:i + batch])).flatten(1).numpy() for i in range(0, len(A), batch)])

def fit_lr(A, Y, n_words, steps=300, lam=1e-3):
    """Multinomial logistic regression of each row of labels Y (k, n) on features A (n, d), all k at once: Adam from
    zero with an L2 penalty. Returns weights (k, d, n_words) and biases (k, 1, n_words)."""
    X, Yt = torch.from_numpy(A).float(), torch.from_numpy(Y)
    W = torch.zeros(len(Y), A.shape[1], n_words, requires_grad=True); b = torch.zeros(len(Y), 1, n_words, requires_grad=True)
    opt = torch.optim.Adam([W, b], 0.05)
    for _ in range(steps):
        loss = nnf.cross_entropy((X @ W + b).flatten(0, 1), Yt.flatten(), reduction="sum") / len(A) + lam * (W ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return W.detach(), b.detach()

def lr_rule(E, tr, q, n_words):
    """Logistic regression on the training takes' standardised embeddings. Margin: 1 - second/best probability."""
    mu, sd = E[tr].mean(0), E[tr].std(0) + 1e-6; A, B = (E[tr] - mu) / sd, torch.from_numpy((E[q] - mu) / sd).float()
    def probs(Y):
        W, b = fit_lr(A, Y[:, tr], n_words)
        return torch.softmax(B @ W + b, -1).numpy()
    def rule(lab):
        p = probs(lab[None])[0]; o = np.sort(p, 1)
        return p.argmax(1), 1 - o[:, -2] / o[:, -1]
    rule.many = lambda S: probs(S).argmax(2)
    return rule

if __name__ == "__main__":
    args = proto_net.cli(__doc__, lambda ap: ap.add_argument("--model", choices=sorted(WINDOW), default="harnet5",
                                                              help="ssl-wearables encoder (default: %(default)s)"))
    start = time.time()
    T, y, who, wear, local, words = proto_net.load(args.dataset, args.gap)
    if len(set(wear)) < 2: sys.exit(f"one person with one wear (no createdAt gap over {args.gap:g} s, no wear field): nothing to hold out")
    try: m = encoder(args.model)
    except Exception as e: sys.exit(f"could not load {args.model} from torch.hub (needs network the first time): {e!r}")
    split = "leave-one-wear-out"
    got = proto_net.methods(T, y, who, wear, local, args.episodes, args.steps, args.seed, splits=(split,))
    E = embed(m, accel(proto_net.features(T, wear)[0], WINDOW[args.model]))
    for name, tr, en, q, r in got[split]:
        r[f"{args.model} + logistic regression, all training takes"] = lr_rule(E, tr, q, len(words))
        r[f"{args.model}, {proto_net.SHOT}-take prototypes"] = proto_net.proto_rule(E, en, q)
    head = (proto_net.describe(args, T, words, who, wear) + f" Encoder: ssl-wearables {args.model}, "
            f"{proto_net.n_params(m):,} parameters, frozen, {E.shape[1]}-number embedding; input: user accel + 1 g "
            f"gravity at 30 Hz, centred in {WINDOW[args.model]} samples.")
    out = f"{head}\n\n{proto_net.table(got, y, args.margin, args.shuffles)}"
    print(out); print(f"total {time.time() - start:.0f} s", file=sys.stderr)
    if args.results: proto_net.append(args.results, out)
