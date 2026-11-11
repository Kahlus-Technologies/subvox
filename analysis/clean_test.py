"""Does better cleaning help? Test trimming and smoothing variants on Aayu's AirPods takes.
Scores: block-recorded set (146) with the within-word half split, and drill takes with leave-one-out
plus a time split (train on the first 60% of drill takes, test on the rest).
Takes with createdAt <= --since are block takes, later ones are drill takes.
Usage: python3 analysis/clean_test.py [--dataset PATH] [--since CREATED_AT]
"""
from clean_test_lib import parse_args, load_takes, ROT, make, scores

args = parse_args(__doc__, drill_since_file=True)
ALL, words = load_takes(args.dataset)
since = float("inf") if args.since is None else args.since

block = [t for t in ALL if t["createdAt"] <= since]; drill = [t for t in ALL if t["createdAt"] > since]
print(f"block takes {len(block)}, drill takes {len(drill)}, drill per word {[sum(t['label']==w for t in drill) for w in words]}\n")
variants = [
    ("no trimming", dict(frac=None)),
    ("Subvox default trim", dict(frac=0.15, pad=2)),
    ("tighter trim", dict(frac=0.30, pad=1)),
    ("tighter trim + smooth 3", dict(frac=0.30, pad=1, sm=3)),
    ("default trim + smooth 3", dict(frac=0.15, pad=2, sm=3)),
    ("default trim + per-take scale", dict(frac=0.15, pad=2, znorm=True)),
    ("tight + smooth + per-take scale", dict(frac=0.30, pad=1, sm=3, znorm=True)),
]
print(f"{'cleaning (rotation only)':34s} {'block: half-split':>18s} {'drill: leave-one-out':>21s} {'drill: half-split':>18s}")
for name, kw in variants:
    _, bh = scores(block, make(block, ROT, **kw))
    dl, dh = scores(drill, make(drill, ROT, **kw)) if len(drill) >= 10 else (float("nan"),) * 2
    print(f"{name:34s} {bh:>18.0%} {dl:>21.0%} {dh:>18.0%}")
