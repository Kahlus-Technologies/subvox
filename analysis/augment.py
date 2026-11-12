"""Re-seat augmentations for one take's (T, 9) motion array, in the app's order: ax ay az rx ry rz pitch roll yaw.
Pure numpy. Each step takes the array and a numpy Generator and returns a new array. Only rotate() cares which
columns are vectors, so a (T, 6) accel + rotation-rate array works too. Plan step 3 in
docs/research/prior-work-and-training-plan.md (issue #17); reseat.py --augment scores them.
"""
import numpy as np
from clean_test_lib import rotvec, turn

def rotate(x, rng, deg=10.0):
    """Turn accel and rotation rate by one random 3D rotation of up to `deg` degrees, like a slightly different seat."""
    axis = rng.normal(size=3)
    return turn(x, rotvec(axis / np.linalg.norm(axis) * np.radians(rng.uniform(-deg, deg))))

def time_warp(x, rng, frac=0.15):
    """Stretch or squeeze the take in time by up to `frac` (linear interpolation of every channel)."""
    n = len(x); p = np.linspace(0, n - 1, max(2, round(n * rng.uniform(1 - frac, 1 + frac))))
    return np.stack([np.interp(p, np.arange(n), c) for c in x.T], 1)

def scale(x, rng, lo=0.8, hi=1.2):
    """Multiply the whole take by one random gain, like mouthing louder or softer."""
    return x * rng.uniform(lo, hi)

def drift(x, rng, size=0.1, knots=4):
    """Add slow wander to each channel: straight lines through `knots` random points, `size` times the
    channel's std, with zero mean so the take stays centred."""
    n = len(x); at = np.linspace(0, n - 1, knots)
    d = np.stack([np.interp(np.arange(n), at, rng.normal(size=knots)) for _ in x.T], 1) * size * x.std(0)
    return x + d - d.mean(0)

STEPS = {"rotation": rotate, "time warp": time_warp, "magnitude": scale, "drift": drift}

def augment(x, rng, steps=tuple(STEPS)):
    """Apply the named steps, in STEPS order."""
    for name, f in STEPS.items():
        if name in steps: x = f(x, rng)
    return x
