"""Open-vocabulary baselines on talk-mode data (#11): can AirPods head motion spell words it never saw?

Every word is scored by a model that never saw its session (or its speaker):
1. DTW-1NN: a test word takes the word of its nearest training word by dynamic time warping over the 9 motion
   channels. It can only name words that occur in training, so the "seen" column is its ceiling.
2. CTC: a small 1D-CNN + GRU reads a word's motion and spells its ARPAbet phonemes (trained with CTC loss), so
   it can spell unseen words. For word accuracy its spelling is snapped to the closest pronunciation in the
   word list of the whole dataset, test words included. That word accuracy is closed-set; CER is the
   open-vocabulary measure.
Splits: held-out sessions (each speaker's sessions in time order, cut into --folds blocks, so a sentence
recorded seconds earlier in the same fit never trains its neighbour; --folds 0 is leave-one-session-out) and
held-out speakers (leave-one-speaker-out, when there are 2+ speakers).
Chance: the same predictions scored against the test targets shuffled within each fold, --perms times, the
label-shuffle standard of score.py. p is the share of shuffles that score at least as well.
Results from speakers named "Synthetic ..." (analysis/fixtures/make_sessions.py) are labelled synthetic.

Input: --sessions ROOT [ROOT ...] (folders of session folders, read with tools/build_dataset.py) or --npz
words.npz (written by tools/build_dataset.py). Silent takes (paired Collect) are left out. Phonemes come from each session's phones.json
(tools/align_phones.py), else from --lexicon FILE, a CMUdict-format pronunciation file you supply (lines in
MFA .dict layout, with probability columns before the phones, are read too). words.npz has no folder paths, so
with --npz phonemes come from --lexicon only and word times from words.json.

Usage: python3 analysis/open_vocab.py --sessions ~/Documents/Subvox/Sessions [--lexicon FILE] [--results analysis/RESULTS.md]
Needs numpy and numba (requirements.txt), plus torch for CTC (requirements-ml.txt). CPU only, seeded.
"""
import argparse, datetime, importlib.util, json, pathlib, re, shlex, sys, time
from collections import Counter, defaultdict
import numpy as np
from numba import njit, prange

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
from build_dataset import build, clean
from clean_test_lib import dtw

PHONES = "AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH".split()
PHONE_ID = {p: i + 1 for i, p in enumerate(PHONES)}  # 0 is the CTC blank
HEADING = "## Open vocabulary (#11)"


def arpabet(p):
    return re.sub(r"[012]$", "", p.upper())


def load_lexicon(path):
    """{word: [phone, ...]} from a CMUdict-style file: WORD then ARPAbet phones. Stress is dropped, the first
    pronunciation wins, ;;; and # comments and numeric columns (MFA .dict probabilities) are skipped."""
    lex = {}
    for line in pathlib.Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.split(" #")[0]
        if not line.strip() or line.startswith((";;;", "#")):
            continue
        word, *rest = line.split()
        phones = [arpabet(p) for p in rest if re.fullmatch(r"[A-Za-z]+[012]?", p)]
        word = clean(re.sub(r"\(\d+\)$", "", word))
        if word and phones and word not in lex and all(p in PHONE_ID for p in phones):
            lex[word] = phones
    return lex


def load(args):
    """Word rows (speaker, session, word, start, end, motion, phones) and a count of where phonemes came from.
    Sessions with phones.json are cut on its word times, so the motion window matches the phones it is trained
    on; the rest are cut on words.json's ASR word times."""
    if args.npz:
        z = np.load(args.npz)
        o = z["offsets"]
        rows = [dict(speaker=str(z["speaker"][i]), session=str(z["session"][i]), word=str(z["word"][i]),
                     start=float(z["start"][i]), end=float(z["end"][i]), motion=z["motion"][o[i]:o[i + 1]])
                for i in range(len(o) - 1) if "labels_from" not in z.files or not z["labels_from"][i]]
        aligned = []
    else:
        rows, summary = build(args.sessions)
        print(summary, end="")
        aligned, _ = build(args.sessions, "phones.json")
        # Silent takes borrow their spoken partner's words (labels_from) and have no phones.json of their own.
        # This is talk-mode data, so they are left out.
        silent = {r["session"] for r in rows if r["labels_from"]}
        rows, aligned = ([r for r in rs if not r["labels_from"]] for rs in (rows, aligned))
        if silent:
            print(f"left out {len(silent)} silent takes")
        folders = {d.name: d for root in args.sessions for d in root.expanduser().iterdir() if d.is_dir()}
        timed = {r["session"] for r in aligned}
        rows = aligned + [r for r in rows if r["session"] not in timed]
        print(f"word times from phones.json in {len(timed)} sessions, from words.json in the other "
              f"{len({r['session'] for r in rows}) - len(timed)}")
    lexicon = load_lexicon(args.lexicon) if args.lexicon else {}
    alignments, source = {}, Counter()
    for r in rows[:len(aligned)]:
        if r["session"] not in alignments:
            alignments[r["session"]] = json.loads((folders[r["session"]] / "phones.json").read_text())["phones"]
        phones = [arpabet(p["phone"]) for p in alignments[r["session"]]
                  if p["start"] >= r["start"] - 1e-6 and p["end"] <= r["end"] + 1e-6]
        r["phones"] = phones if phones and all(p in PHONE_ID for p in phones) else None  # "spn": no pronunciation
    for r in rows:
        how = "phones.json" if r.get("phones") else "lexicon" if r["word"] in lexicon else "none"
        r["phones"] = r.get("phones") or lexicon.get(r["word"])
        source[how] += 1
    return rows, source


def features(rows):
    """Per word: the 9 channels with attitude as change since the word's start, z-scored per speaker over all
    of that speaker's words. No labels are used, so a new speaker is normalised the same way. (Per session is
    worse: a session is one sentence, so its mean and spread depend on which phonemes it happens to contain.)"""
    X, by_speaker = [], defaultdict(list)
    for i, r in enumerate(rows):
        x = r["motion"].astype(np.float64)
        X.append(np.column_stack([x[:, :6], x[:, 6:] - x[0, 6:]]))
        by_speaker[r["speaker"]].append(i)
    for idx in by_speaker.values():
        both = np.concatenate([X[i] for i in idx])
        mean, sd = both.mean(0), both.std(0) + 1e-6
        for i in idx:
            X[i] = (X[i] - mean) / sd
    return X


@njit(parallel=True, cache=True)
def dtw_matrix(X, off, band):
    n = len(off) - 1
    M = np.zeros((n, n), np.float32)
    for i in prange(n):
        for j in range(i + 1, n):
            M[i, j] = dtw(X[off[i]:off[i + 1]], X[off[j]:off[j + 1]], band)
    return M + M.T


@njit(cache=True)
def lev(a, b):
    prev = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = np.empty(len(b) + 1, np.int64)
        cur[0] = i
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[len(b)]


@njit(cache=True)
def edits(P, Po, T, To, items, targets):
    """Total edit distance of prediction items[n] against target targets[n]."""
    s = 0
    for n in range(len(items)):
        i, k = items[n], targets[n]
        s += lev(P[Po[i]:Po[i + 1]], T[To[k]:To[k + 1]])
    return s


@njit(cache=True)
def snap(P, Po, V, Vo):
    """For each prediction, the index of the closest vocabulary pronunciation (edit distance, first on ties)."""
    out = np.zeros(len(Po) - 1, np.int64)
    for i in range(len(Po) - 1):
        best = 1 << 30
        for j in range(len(Vo) - 1):
            d = lev(P[Po[i]:Po[i + 1]], V[Vo[j]:Vo[j + 1]])
            if d < best:
                best = d
                out[i] = j
    return out


def flat(seqs):
    """Sequences of phone ids as one array plus offsets, the layout numba wants."""
    seqs = [np.asarray(s, np.int64) for s in seqs]
    return np.concatenate(seqs + [np.zeros(0, np.int64)]), np.concatenate([[0], np.cumsum([len(s) for s in seqs])]).astype(np.int64)


def ctc_predictions(X, targets, folds, args):
    """Greedy CTC phoneme ids for every item, each from a model trained without the item's fold."""
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    torch.use_deterministic_algorithms(True)

    class Net(nn.Module):  # ~80k parameters
        def __init__(self, h=64):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(9, h, 5, padding=2), nn.ReLU(), nn.Conv1d(h, h, 5, padding=2), nn.ReLU())
            self.gru = nn.GRU(h, h, batch_first=True, bidirectional=True)
            self.out = nn.Linear(2 * h, len(PHONES) + 1)

        def forward(self, x, lengths):
            y = self.conv(x.transpose(1, 2)).transpose(1, 2)
            y, _ = self.gru(pack_padded_sequence(y, lengths, batch_first=True, enforce_sorted=False))
            y, _ = pad_packed_sequence(y, batch_first=True, total_length=x.shape[1])
            return self.out(y).log_softmax(-1)

    def batch(idx):
        lengths = torch.tensor([len(X[i]) for i in idx])
        x = torch.zeros(len(idx), int(lengths.max()), 9)
        for k, i in enumerate(idx):
            x[k, :len(X[i])] = torch.from_numpy(X[i]).float()
        return x, lengths

    preds = [None] * len(X)
    for f in np.unique(folds):
        train, test = np.where(folds != f)[0], np.where(folds == f)[0]
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        net, loss_fn = Net(), nn.CTCLoss(zero_infinity=True)
        opt = torch.optim.Adam(net.parameters(), 2e-3)
        for _ in range(args.epochs):
            for idx in np.array_split(rng.permutation(train), max(1, len(train) // 32)):
                x, lengths = batch(idx)
                y = torch.tensor([p for i in idx for p in targets[i]])
                loss = loss_fn(net(x, lengths).transpose(0, 1), y, lengths, torch.tensor([len(targets[i]) for i in idx]))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
        with torch.no_grad():
            for idx in np.array_split(test, max(1, len(test) // 256)):
                x, lengths = batch(idx)
                for i, best, n in zip(idx, net(x, lengths).argmax(-1).numpy(), lengths.tolist()):
                    best = best[:n]
                    keep = (best != 0) & np.concatenate([[True], best[1:] != best[:-1]])  # collapse repeats, drop blanks
                    preds[i] = best[keep]
    return preds


def score(items, pred_w, P, Po, data, folds, args, rng):
    """Word accuracy and phoneme CER of the predictions for items, each with its shuffle chance and p."""
    groups = [np.where(folds[items] == f)[0] for f in np.unique(folds[items])]

    def once(targets):
        acc = np.mean(data["pron"][pred_w[items]] == data["pron"][data["word"][targets]])
        if data["T"] is None:
            return acc, np.nan
        return acc, edits(P, Po, data["T"], data["To"], items, targets) / np.diff(data["To"])[targets].sum()

    acc, cer = once(items)
    null = []
    for _ in range(args.perms):
        targets = items.copy()
        for g in groups:
            targets[g] = rng.permutation(items[g])
        null.append(once(targets))
    null = np.array(null)
    return dict(acc=acc, acc_chance=null[:, 0].mean(), acc_p=(np.sum(null[:, 0] >= acc) + 1) / (args.perms + 1),
                cer=cer, cer_chance=null[:, 1].mean(), cer_p=(np.sum(null[:, 1] <= cer) + 1) / (args.perms + 1))


def session_folds(speakers, sessions, k):
    """Fold per row: each speaker's sessions in time order (folder names are timestamps) cut into k blocks."""
    if k == 0:  # one fold per session
        return np.unique([f"{a}\0{b}" for a, b in zip(speakers, sessions)], return_inverse=True)[1]
    fold = np.zeros(len(speakers), np.int64)
    for s in np.unique(speakers):
        names = sorted(set(sessions[speakers == s]))
        block = {name: i * min(k, len(names)) // len(names) for i, name in enumerate(names)}
        for i in np.where(speakers == s)[0]:
            fold[i] = block[sessions[i]]
    return fold


def cell(r, metric):
    if np.isnan(r[metric]):
        return "-"
    return f"{r[metric]:.0%} ({r[metric + '_chance']:.0%}) p={r[metric + '_p']:.3f}"


def write_results(path, kind, block):
    """Put block between this kind's markers in path's open-vocabulary section, replacing any earlier run."""
    start, end = f"<!-- open_vocab.py {kind} -->", f"<!-- /open_vocab.py {kind} -->"
    text = path.read_text() if path.exists() else "# Results\n"
    if start in text and end in text:
        head, rest = text.split(start, 1)
        text = head + start + "\n" + block + end + rest.split(end, 1)[1]
    else:
        if HEADING not in text:
            text = text.rstrip("\n") + f"\n\n{HEADING}\n"
        head, rest = text.split(HEADING, 1)
        cut = rest.find("\n## ") if "\n## " in rest else len(rest)
        text = head + HEADING + rest[:cut].rstrip("\n") + f"\n\n{start}\n{block}{end}\n" + rest[cut:]
    path.write_text(text)


def main(argv=None):
    began = time.time()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sessions", nargs="+", type=pathlib.Path, metavar="ROOT", help="folders of session folders")
    src.add_argument("--npz", type=pathlib.Path, help="words.npz from tools/build_dataset.py")
    p.add_argument("--lexicon", type=pathlib.Path, help="CMUdict-format pronunciations for words without phones.json")
    p.add_argument("--folds", type=int, default=5, help="time blocks per speaker for held-out sessions; 0 = one per session")
    p.add_argument("--epochs", type=int, default=30, help="CTC training epochs per fold (default %(default)s)")
    p.add_argument("--perms", type=int, default=200, help="target shuffles for chance (default %(default)s)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-ctc", action="store_true", help="DTW only (no torch needed)")
    p.add_argument("--results", type=pathlib.Path, help="write this run into the open-vocabulary section of this file")
    args = p.parse_args(argv)

    rows, source = load(args)
    if not rows:
        sys.exit("no words with motion found")
    has_phones = source["phones.json"] + source["lexicon"] > 0
    if has_phones:
        rows = [r for r in rows if r["phones"]]
        print(f"phonemes from phones.json: {source['phones.json']} words, from the lexicon: {source['lexicon']}, "
              f"neither (left out): {source['none']}")
    speakers, sessions = np.array([r["speaker"] for r in rows]), np.array([r["session"] for r in rows])
    vocab = sorted({r["word"] for r in rows})
    word = np.searchsorted(vocab, [r["word"] for r in rows])
    data = dict(word=word, T=None, To=None, pron=np.arange(len(vocab)))
    if has_phones:
        targets = [[PHONE_ID[p] for p in r["phones"]] for r in rows]
        data["T"], data["To"] = flat(targets)
        # A word's pronunciation is its most common phone string; words that sound the same count as the same word.
        spellings = defaultdict(Counter)
        for t, w in zip(targets, word):
            spellings[w][tuple(t)] += 1
        prons = [spellings[v].most_common(1)[0][0] for v in range(len(vocab))]
        data["pron"] = np.unique([" ".join(map(str, pr)) for pr in prons], return_inverse=True)[1]
        V, Vo = flat(prons)
    synthetic = any(s.lower().startswith("synthetic") for s in speakers)
    print(f"{len(rows)} words, {len(vocab)} unique, {len(set(zip(speakers, sessions)))} sessions, "
          f"{len(set(speakers))} speakers{' (SYNTHETIC fixture, not recordings)' if synthetic else ''}")

    X = features(rows)
    t0 = time.time()
    M = dtw_matrix(np.concatenate(X), np.concatenate([[0], np.cumsum([len(x) for x in X])]).astype(np.int64), 10)
    print(f"DTW distances between all {len(X)} words: {time.time() - t0:.0f} s")

    splits = [(f"held-out sessions ({'one fold per session' if args.folds == 0 else f'{args.folds} time blocks per speaker'})",
               session_folds(speakers, sessions, args.folds)),
              ("held-out speakers", np.unique(speakers, return_inverse=True)[1])]
    for name, folds in splits:
        if len(np.unique(folds)) < 2:
            print(f"{name}: skipped, needs 2+ {name.split()[1]}")
    splits = [(name, folds) for name, folds in splits if len(np.unique(folds)) > 1]
    use_ctc = has_phones and not args.no_ctc
    if not has_phones:
        print("CTC and CER skipped: no phonemes (add phones.json with tools/align_phones.py, or pass --lexicon)")
    elif use_ctc and importlib.util.find_spec("torch") is None:
        print("CTC skipped: torch is not installed (pip install -r analysis/requirements-ml.txt)")
        use_ctc = False

    rng, results = np.random.default_rng(args.seed), []
    everything = np.arange(len(rows))
    for name, folds in splits:
        seen = np.zeros(len(rows), bool)
        for f in np.unique(folds):
            seen[folds == f] = np.isin(word[folds == f], word[folds != f])
        models = []
        pred_w = np.zeros(len(rows), np.int64)  # DTW-1NN: word of the nearest word outside the test fold
        for f in np.unique(folds):
            test, train = np.where(folds == f)[0], np.where(folds != f)[0]
            pred_w[test] = word[train[M[np.ix_(test, train)].argmin(1)]]
        models.append(("DTW-1NN", pred_w) + (flat([prons[w] for w in pred_w]) if has_phones else (None, None)))
        if use_ctc:
            t0 = time.time()
            P, Po = flat(ctc_predictions(X, targets, folds, args))
            print(f"CTC on {name}: {len(np.unique(folds))} models trained in {time.time() - t0:.0f} s")
            models.append(("CTC", snap(P, Po, V, Vo), P, Po))
        for model, pred_w, P, Po in models:
            r = dict(split=name, model=model, words=len(rows), seen=seen.mean())
            r.update(score(everything, pred_w, P, Po, data, folds, args, rng))
            unseen = np.where(~seen)[0]
            r["unseen"] = score(unseen, pred_w, P, Po, data, folds, args, rng) if len(unseen) else None
            results.append(r)

    head = ["split", "model", "words", "seen in training", "word accuracy (chance) p", "phoneme CER (chance) p",
            "unseen words: accuracy", "unseen words: CER"]
    table = [[r["split"], r["model"], str(r["words"]), f"{r['seen']:.0%}", cell(r, "acc"), cell(r, "cer")]
             + [cell(r["unseen"], m) if r["unseen"] else "-" for m in ("acc", "cer")] for r in results]
    widths = [max(len(x) for x in col) for col in zip(head, *table)]
    print()
    for line in [head] + table:
        print("  ".join(x.ljust(w) for x, w in zip(line, widths)).rstrip())
    print(f"\nchance = the same predictions against test targets shuffled within each fold ({args.perms} times); "
          f"p = share of shuffles at least as good.\nseen in training = test words that also occur in the training "
          f"folds (DTW-1NN can only name those); unseen words are the rest. Lower CER is better. Took {time.time() - began:.0f} s.")

    if args.results:
        command = "python3 analysis/open_vocab.py " + shlex.join(sys.argv[1:] if argv is None else argv)
        summary = (f"{'**Synthetic fixture, not recordings.**' if synthetic else '**Real data.**'} Run "
                   f"{datetime.date.today()}: {len(set(speakers))} speakers, {len(set(zip(speakers, sessions)))} sessions, "
                   f"{len(rows)} words ({len(vocab)} unique); phonemes from phones.json for {source['phones.json']} words, "
                   f"from the lexicon for {source['lexicon']}. Took {time.time() - began:.0f} s.")
        md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in table]
        write_results(args.results, "synthetic" if synthetic else "real", f"{summary}\n\n    {command}\n\n" + "\n".join(md) + "\n")
        print(f"wrote {args.results}")
    return results


if __name__ == "__main__":
    main()
