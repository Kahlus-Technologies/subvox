"""Writes dataset_small.json next to this file: 3 words x 6 takes of synthetic motion in the app's dataset.json format.

Two wears of 9 block-recorded takes, 10 minutes apart. createdAt 800000000 splits the takes 8 / 10 for
clean_test.py --since. Deterministic (seed 0). Regenerate with: python3 analysis/fixtures/make_fixture.py
"""
import json, pathlib, uuid
import numpy as np

RATE = 50.0  # AirPods Pro motion rate
WORDS = [("yes", 0, 2.0), ("no", 1, 3.0), ("stop", 2, 1.5)]  # label, rotation axis, cycles per take
rng = np.random.default_rng(0)

def take(axis, cycles, t0):
    n = int(30 + rng.integers(-4, 5))
    i = np.arange(n)
    wave = np.sin(np.pi * i / (n - 1)) * np.sin(2 * np.pi * cycles * i / n)
    rot = rng.normal(0, 0.01, (n, 3)); rot[:, axis] += 0.3 * wave
    acc = rng.normal(0, 0.002, (n, 3)); acc[:, axis] += 0.02 * wave
    att = np.cumsum(rot, 0) / RATE
    r = lambda v: round(float(v), 5)
    return [{"t": r(t0 + k / RATE), "ax": r(acc[k, 0]), "ay": r(acc[k, 1]), "az": r(acc[k, 2]),
             "rx": r(rot[k, 0]), "ry": r(rot[k, 1]), "rz": r(rot[k, 2]),
             "pitch": r(att[k, 0]), "roll": r(att[k, 1]), "yaw": r(att[k, 2]),
             "gx": 0.0, "gy": 0.0, "gz": -1.0, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0} for k in range(n)]

templates = []
created = 799999850.0  # seconds since 2001-01-01, like Swift's default Date encoding
for wear in range(2):
    for label, axis, cycles in WORDS:
        for _ in range(3):
            templates.append({"id": str(uuid.UUID(int=len(templates) + 1)).upper(), "label": label,
                              "samples": take(axis, cycles, 1000.0 + 5 * len(templates)), "createdAt": created})
            created += 20
    created += 600

out = pathlib.Path(__file__).with_name("dataset_small.json")
out.write_text(json.dumps({"vocabulary": [w for w, _, _ in WORDS], "templates": templates}, separators=(",", ":")))
print(f"wrote {out} ({len(templates)} takes)")
