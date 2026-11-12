"""Writes two synthetic datasets next to this file, in the app's dataset.json format. Deterministic.
Regenerate with: python3 analysis/fixtures/make_fixture.py

dataset_small.json: 3 words x 6 takes. Two wears of 9 block-recorded takes, 10 minutes apart. createdAt
800000000 splits the takes 8 / 10 for clean_test.py --since.
dataset_reseat.json (for reseat.py): 3 wears x 3 words x 4 takes, 10 minutes apart. The words differ mostly
in the direction of the motion, and each wear tilts the AirPods by a fixed rotation (accel, rotation rate and
gravity together), like re-seating them. A tilt confuses these words by construction: this checks the
scripts, it does not say how much real re-seats hurt.
"""
import json, pathlib, uuid
import numpy as np

RATE = 50.0  # AirPods Pro motion rate
WORDS = [("yes", 0, 2.0), ("no", 1, 3.0), ("stop", 2, 1.5)]  # label, rotation axis, cycles per take
rng = np.random.default_rng(0)

def samples(acc, rot, grav, t0):
    att = np.cumsum(rot, 0) / RATE
    r = lambda v: round(float(v), 5)
    return [{"t": r(t0 + k / RATE), "ax": r(acc[k, 0]), "ay": r(acc[k, 1]), "az": r(acc[k, 2]),
             "rx": r(rot[k, 0]), "ry": r(rot[k, 1]), "rz": r(rot[k, 2]),
             "pitch": r(att[k, 0]), "roll": r(att[k, 1]), "yaw": r(att[k, 2]),
             "gx": r(grav[k, 0]), "gy": r(grav[k, 1]), "gz": r(grav[k, 2]),
             "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0} for k in range(len(acc))]

def take(axis, cycles, t0):
    n = int(30 + rng.integers(-4, 5))
    i = np.arange(n)
    wave = np.sin(np.pi * i / (n - 1)) * np.sin(2 * np.pi * cycles * i / n)
    rot = rng.normal(0, 0.01, (n, 3)); rot[:, axis] += 0.3 * wave
    acc = rng.normal(0, 0.002, (n, 3)); acc[:, axis] += 0.02 * wave
    return samples(acc, rot, np.tile([0.0, 0.0, -1.0], (n, 1)), t0)

def write(name, vocabulary, templates):
    out = pathlib.Path(__file__).with_name(name)
    out.write_text(json.dumps({"vocabulary": vocabulary, "templates": templates}, separators=(",", ":")))
    print(f"wrote {out} ({len(templates)} takes)")

templates = []
created = 799999850.0  # seconds since 2001-01-01, like Swift's default Date encoding
for wear in range(2):
    for label, axis, cycles in WORDS:
        for _ in range(3):
            templates.append({"id": str(uuid.UUID(int=len(templates) + 1)).upper(), "label": label,
                              "samples": take(axis, cycles, 1000.0 + 5 * len(templates)), "createdAt": created})
            created += 20
    created += 600
write("dataset_small.json", [w for w, _, _ in WORDS], templates)

# dataset_reseat.json. Head frame: x forward, z up, gravity (0, 0, -1). Each word turns the head about an axis
# in the x-z plane (angle from x, degrees); a tilt about y swings those axes toward each other.
SEATED = [("yes", 0, 2.0), ("no", 40, 2.25), ("stop", 80, 2.0)]  # label, axis angle, cycles per take
TILTS = [(0, 1), (25, 1), (-25, 1)]  # per wear: degrees about the head's x (0) or y (1) axis
rng2 = np.random.default_rng(1)

def tilt(deg, axis):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg)); i, j = [(1, 2), (2, 0)][axis]
    R = np.eye(3); R[i, i] = R[j, j] = c; R[i, j] = -s; R[j, i] = s
    return R

def seated_take(angle, cycles, R, t0):
    n = int(30 + rng2.integers(-4, 5))
    i = np.arange(n)
    wave = rng2.uniform(0.8, 1.2) * np.sin(np.pi * i / (n - 1)) * np.sin(2 * np.pi * cycles * i / n)
    a = np.radians(angle + rng2.normal(0, 4))
    d = np.array([np.cos(a), 0, np.sin(a)]); up = np.array([-np.sin(a), 0, np.cos(a)])
    rot = rng2.normal(0, 0.01, (n, 3)) + 0.3 * np.outer(wave, d)
    acc = rng2.normal(0, 0.002, (n, 3)) + 0.02 * np.outer(wave, up)
    grav = R @ [0, 0, -1.0] + rng2.normal(0, 0.005, (n, 3))
    return samples(acc @ R.T, rot @ R.T, grav, t0)

templates = []
created = 810000000.0
for deg, axis in TILTS:
    for label, angle, cycles in SEATED:
        for _ in range(4):
            templates.append({"id": str(uuid.UUID(int=len(templates) + 1)).upper(), "label": label,
                              "samples": seated_take(angle, cycles, tilt(deg, axis), 2000.0 + 5 * len(templates)),
                              "createdAt": created})
            created += 20
    created += 600
write("dataset_reseat.json", [w for w, _, _ in SEATED], templates)
