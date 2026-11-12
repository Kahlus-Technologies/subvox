"""Turn Subvox session folders into one training-ready word dataset.

Every session with motion.csv and words.json gives one row per spoken word: speaker, session, prompt,
word, start, end and the motion samples inside [start, end] (the 9 channels ax ay az rx ry rz pitch roll
yaw, float32). Words are lowercased with punctuation stripped except apostrophes ("don't"), the same
vocabulary as tools/align_phones.py. Needs only numpy.

Silent takes (meta.json condition "Silent mouthing", or a folder name ending in -silent) have no speech in
their audio, so their own words.json is never used. Collect with pairing on records each sentence spoken,
then mouthed, as two sessions with the same meta.json pairId. A silent take gets its spoken partner's
words: the two motion streams are DTW-aligned (accel and rotation-rate channels, each z-scored within its
take) and every word start/end is moved through the warping path onto the silent take's clock. Those rows
have labels_from set to the spoken session's folder name. Silent takes without a spoken partner are skipped.

Usage: python3 tools/build_dataset.py ROOT [ROOT ...] [--out DIR]
    ROOT: a folder of session folders, e.g. ~/Documents/Subvox/Sessions or an unzipped export's Sessions/.
    Writes DIR/words.npz and DIR/summary.txt (default DIR: subvox-dataset).

words.npz arrays (load with np.load, no pickle needed):
    speaker, session, prompt, word   str, one per row
    labels_from                      str: "" when the words came from the session's own audio, else the
                                     paired spoken session the timings were moved over from
    start, end                       float64 seconds on the motion.csv clock
    motion                           float32 [total samples, 9], every row's slice back to back
    offsets                          int64 [rows + 1]: row i is motion[offsets[i]:offsets[i + 1]]
"""
import argparse, json, pathlib, re
from collections import Counter
import numpy as np

CHANNELS = ["ax", "ay", "az", "rx", "ry", "rz", "pitch", "roll", "yaw"]
# Channels that align a silent take with its spoken take. Attitude is left out: slow posture drift between
# the two takes would outweigh the articulation.
ALIGN = ["ax", "ay", "az", "rx", "ry", "rz"]
SILENT = "Silent mouthing"


def clean(word):
    return re.sub(r"[^\w']", "", word).lower()


def read_meta(d):
    return json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}


def is_silent(d, meta):
    return meta.get("condition") == SILENT or d.name.endswith("-silent")


def read_motion(d):
    """(t, the 9 CHANNELS as [samples, 9]) from motion.csv."""
    with open(d / "motion.csv") as f:
        header = f.readline().strip().split(",")
    columns = [header.index(c) for c in ["t"] + CHANNELS]
    motion = np.loadtxt(d / "motion.csv", delimiter=",", skiprows=1, usecols=columns, ndmin=2)
    if len(motion) == 0:
        raise ValueError("motion.csv is empty")
    return motion[:, 0], motion[:, 1:]


def align_features(x):
    """ALIGN channels of x ([samples, 9]), each z-scored, so weaker mouthed motion still matches spoken motion."""
    x = x[:, [CHANNELS.index(c) for c in ALIGN]]
    return (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-9)


def dtw_path(a, b):
    """Lowest-cost monotone alignment of a [n, k] and b [m, k] under Euclidean frame distance, as an int
    array of (i, j) pairs from (0, 0) to (n - 1, m - 1). Each step moves i, j, or both by one."""
    n, m = len(a), len(b)
    sq = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T
    cost = np.sqrt(np.maximum(sq, 0))
    acc = np.empty((n, m))
    acc[0] = np.cumsum(cost[0])
    for i in range(1, n):
        # acc[i, j] = cost[i, j] + min(acc[i-1, j-1], acc[i-1, j], acc[i, j-1]). The diagonal and vertical
        # steps come from the row above; the run of horizontal steps is a running minimum over the row.
        from_above = cost[i] + np.minimum(acc[i - 1], np.concatenate([[np.inf], acc[i - 1, :-1]]))
        run = np.cumsum(cost[i])
        acc[i] = run + np.minimum.accumulate(from_above - run)
    i, j, path = n - 1, m - 1, [(n - 1, m - 1)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step = int(np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]]))
            i, j = [(i - 1, j - 1), (i - 1, j), (i, j - 1)][step]
        path.append((i, j))
    return np.array(path[::-1])


def transfer(spans, t_spoken, x_spoken, t_silent, x_silent):
    """(start, end) times on the spoken take's clock, moved onto the silent take's clock, or None for a span
    with no spoken sample inside. The path matches each spoken sample to a run of silent samples. A start
    maps through the first sample of each run and an end through the last, interpolating between spoken
    samples, so a word covers every silent sample matched to its own samples."""
    path = dtw_path(align_features(x_spoken), align_features(x_silent))
    first = np.full(len(t_spoken), len(t_silent) - 1)
    last = np.zeros(len(t_spoken), dtype=int)
    np.minimum.at(first, path[:, 0], path[:, 1])
    np.maximum.at(last, path[:, 0], path[:, 1])
    out = []
    for start, end in spans:
        inside = np.searchsorted(t_spoken, end, "right") > np.searchsorted(t_spoken, start, "left")
        out.append((float(np.interp(start, t_spoken, t_silent[first])),
                    float(np.interp(end, t_spoken, t_silent[last]))) if inside else None)
    return out


def load_session(d, spoken=None, words_file="words.json"):
    """(rows, minutes, dropped words) for one session folder, or raise ValueError saying why it's skipped.
    spoken maps a pairId to its spoken take's folder, where a silent take gets its words.
    words_file="phones.json" slices on the phone alignment's word times instead of the ASR's."""
    if not (d / "motion.csv").exists():
        raise ValueError("no motion.csv")
    meta = read_meta(d)
    source = d
    if is_silent(d, meta):
        source = (spoken or {}).get(meta.get("pairId"))
        if source is None:
            raise ValueError("silent take with no paired spoken take")
    if not (source / words_file).exists():
        raise ValueError(f"no {words_file}" if source == d else f"paired spoken take has no {words_file}")
    words = json.loads((source / words_file).read_text()).get("words") or []
    if not words:
        raise ValueError(f"{words_file} has no words")
    t, motion = read_motion(d)
    spans = [(w["start"], w["end"]) for w in words]
    if source != d:
        spans = transfer(spans, *read_motion(source), t, motion)
    rows, dropped = [], 0
    for w, span in zip(words, spans):
        word = clean(w.get("text") or w.get("word") or "")
        lo, hi = (np.searchsorted(t, span[0], "left"), np.searchsorted(t, span[1], "right")) if span else (0, 0)
        if not word or hi <= lo:
            dropped += 1
            continue
        rows.append(dict(speaker=meta.get("speaker") or "", session=d.name, prompt=meta.get("prompt") or "",
                         word=word, start=float(span[0]), end=float(span[1]), motion=motion[lo:hi].astype(np.float32),
                         labels_from="" if source == d else source.name))
    return rows, (t[-1] - t[0]) / 60, dropped


def build(roots, words_file="words.json"):
    """All rows plus the summary.txt text."""
    folders = [d for root in roots for d in sorted(p for p in pathlib.Path(root).expanduser().iterdir() if p.is_dir())]
    spoken = {}
    for d in folders:
        try:
            meta = read_meta(d)
        except (ValueError, OSError):
            continue  # load_session reports it
        if meta.get("pairId") and not is_silent(d, meta):
            spoken.setdefault(meta["pairId"], d)
    rows, used, skipped, minutes, dropped = [], [], Counter(), 0.0, 0
    for d in folders:
        try:
            session_rows, session_minutes, session_dropped = load_session(d, spoken, words_file)
        except (ValueError, OSError, KeyError) as e:
            skipped[str(e)] += 1
            continue
        rows += session_rows
        used.append(d.name)
        minutes += session_minutes
        dropped += session_dropped
    words = Counter(r["word"] for r in rows)
    lines = [
        f"sessions: {len(used)}",
        f"speakers: {len({r['speaker'] for r in rows})}",
        f"minutes: {minutes:.1f}",
        f"words: {len(rows)}",
        f"unique words: {len(words)}",
        f"words on silent takes, timed from their paired spoken take: {sum(1 for r in rows if r['labels_from'])}",
        f"words dropped (no text or no motion in their window): {dropped}",
        f"sessions skipped: {sum(skipped.values())}",
    ] + [f"  {why}: {n}" for why, n in sorted(skipped.items())]
    return rows, "\n".join(lines) + "\n"


def save(rows, out):
    lengths = [len(r["motion"]) for r in rows]
    np.savez_compressed(
        out,
        **{k: np.array([r[k] for r in rows], dtype=str) for k in ["speaker", "session", "prompt", "word", "labels_from"]},
        start=np.array([r["start"] for r in rows]), end=np.array([r["end"] for r in rows]),
        motion=np.concatenate([r["motion"] for r in rows]) if rows else np.zeros((0, len(CHANNELS)), np.float32),
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", type=pathlib.Path)
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("subvox-dataset"))
    args = p.parse_args()
    rows, summary = build(args.roots)
    args.out.mkdir(parents=True, exist_ok=True)
    save(rows, args.out / "words.npz")
    (args.out / "summary.txt").write_text(summary)
    print(summary, end="")
    print(f"wrote {args.out / 'words.npz'}")


if __name__ == "__main__":
    main()
