"""How far is each take from its nearest neighbour, in the app's units? Shows what max distance does."""
import numpy as np
from clean_test_lib import parse_args, load_takes, make, dmat
ALL, words = load_takes(parse_args(__doc__).dataset)
T = ALL; y = np.array([words.index(t["label"]) for t in T])
X = make(T, ["ax","ay","az","rx","ry","rz","pitch","roll","yaw"], frac=0.15, pad=2)  # app default: all motion
M = dmat(X); np.fill_diagonal(M, np.inf)
best = M.min(1); right = y[M.argmin(1)] == y
print("nearest-neighbour distance percentiles (10/50/90/99):", np.percentile(best, [10, 50, 90, 99]).round(2))
for cut in (1.0, 1.5, 2.0, 3.0, 4.0):
    acc = best <= cut
    print(f"max distance {cut}: answers {acc.mean():.0%} of takes; of those answered, right {right[acc].mean() if acc.any() else 0:.0%}")
