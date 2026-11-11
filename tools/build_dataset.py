"""Turn Subvox session folders into one training-ready word dataset.

Every session with motion.csv and words.json gives one row per spoken word: speaker, session, prompt,
word, start, end and the motion samples inside [start, end] (the 9 channels ax ay az rx ry rz pitch roll
yaw, float32). Words are lowercased with punctuation stripped except apostrophes ("don't"), the same
vocabulary as tools/align_phones.py. Needs only numpy.

Usage: python3 tools/build_dataset.py ROOT [ROOT ...] [--out DIR]
    ROOT: a folder of session folders, e.g. ~/Documents/Subvox/Sessions or an unzipped export's Sessions/.
    Writes DIR/words.npz and DIR/summary.txt (default DIR: subvox-dataset).

words.npz arrays (load with np.load, no pickle needed):
    speaker, session, prompt, word   str, one per row
    start, end                       float64 seconds on the motion.csv clock
    motion                           float32 [total samples, 9], every row's slice back to back
    offsets                          int64 [rows + 1]: row i is motion[offsets[i]:offsets[i + 1]]
"""
import argparse, json, pathlib, re
from collections import Counter
import numpy as np

CHANNELS = ["ax", "ay", "az", "rx", "ry", "rz", "pitch", "roll", "yaw"]


def clean(word):
    return re.sub(r"[^\w']", "", word).lower()


def load_session(d):
    """(rows, minutes, dropped words) for one session folder, or raise ValueError saying why it's skipped."""
    if not (d / "motion.csv").exists():
        raise ValueError("no motion.csv")
    if not (d / "words.json").exists():
        raise ValueError("no words.json")
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    words = json.loads((d / "words.json").read_text()).get("words") or []
    if not words:
        raise ValueError("words.json has no words")
    with open(d / "motion.csv") as f:
        header = f.readline().strip().split(",")
    columns = [header.index(c) for c in ["t"] + CHANNELS]
    motion = np.loadtxt(d / "motion.csv", delimiter=",", skiprows=1, usecols=columns, ndmin=2)
    if len(motion) == 0:
        raise ValueError("motion.csv is empty")
    t = motion[:, 0]
    rows, dropped = [], 0
    for w in words:
        word = clean(w.get("text") or w.get("word") or "")
        lo, hi = np.searchsorted(t, w["start"], "left"), np.searchsorted(t, w["end"], "right")
        if not word or hi <= lo:
            dropped += 1
            continue
        rows.append(dict(speaker=meta.get("speaker") or "", session=d.name, prompt=meta.get("prompt") or "",
                         word=word, start=float(w["start"]), end=float(w["end"]),
                         motion=motion[lo:hi, 1:].astype(np.float32)))
    return rows, (t[-1] - t[0]) / 60, dropped


def build(roots):
    """All rows plus the summary.txt text."""
    rows, used, skipped, minutes, dropped = [], [], Counter(), 0.0, 0
    for root in roots:
        for d in sorted(p for p in pathlib.Path(root).expanduser().iterdir() if p.is_dir()):
            try:
                session_rows, session_minutes, session_dropped = load_session(d)
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
        f"words dropped (no text or no motion in their window): {dropped}",
        f"sessions skipped: {sum(skipped.values())}",
    ] + [f"  {why}: {n}" for why, n in sorted(skipped.items())]
    return rows, "\n".join(lines) + "\n"


def save(rows, out):
    lengths = [len(r["motion"]) for r in rows]
    np.savez_compressed(
        out,
        **{k: np.array([r[k] for r in rows], dtype=str) for k in ["speaker", "session", "prompt", "word"]},
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
