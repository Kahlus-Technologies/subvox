"""Checks for augment.py and the issue #17 self-check: python3 -m unittest discover -s analysis -p 'test_*.py'"""
import pathlib, sys, unittest
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import augment, reseat
from clean_test_lib import load_takes, raw

FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures/dataset_reseat.json"


class StepsTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.x = self.rng.normal(size=(40, 9))

    def test_rotation_turns_accel_and_rotation_rate_together_by_at_most_10_degrees(self):
        for _ in range(20):
            y = augment.rotate(self.x, self.rng)
            R = np.linalg.lstsq(self.x[:, 0:3], y[:, 0:3], rcond=None)[0].T
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
            np.testing.assert_allclose(y[:, 3:6], self.x[:, 3:6] @ R.T, atol=1e-9)
            self.assertLessEqual(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))), 10 + 1e-6)
            np.testing.assert_array_equal(y[:, 6:], self.x[:, 6:])

    def test_time_warp_changes_length_by_at_most_15_percent(self):
        for _ in range(50):
            self.assertTrue(34 <= len(augment.time_warp(self.x, self.rng)) <= 46)

    def test_magnitude_is_one_gain_between_0_8_and_1_2(self):
        g = augment.scale(self.x, self.rng) / self.x
        np.testing.assert_allclose(g, g[0, 0])
        self.assertTrue(0.8 <= g[0, 0] <= 1.2)

    def test_drift_is_slow_small_and_centred(self):
        d = augment.drift(self.x, self.rng) - self.x
        self.assertLess(np.abs(np.diff(d, axis=0)).max(), 0.05)  # the signal itself moves ~1 per sample
        self.assertLess(np.abs(d).max(), 0.5)
        np.testing.assert_allclose(d.mean(0), 0, atol=1e-12)


class RecoveryTest(unittest.TestCase):
    def test_normalisation_and_rotation_augmentation_recover_a_fixed_tilt_per_wear(self):
        T, words = load_takes(FIXTURE)
        wear = reseat.wears(T); y = np.array([words.index(t["label"]) for t in T])
        g = [np.mean([raw(t, ["gx", "gy", "gz"]).mean(0) for t, w in zip(T, wear) if w == k], 0) for k in range(3)]
        for k in (1, 2):  # the fixture tilts wears 1 and 2 by 25 degrees away from wear 0
            cos = g[k] @ g[0] / np.linalg.norm(g[k]) / np.linalg.norm(g[0])
            self.assertAlmostEqual(np.degrees(np.arccos(cos)), 25, delta=1)
        base = reseat.lowo(*reseat.features(T, wear, False, False), y, wear).mean()
        fixed = reseat.lowo(*reseat.features(T, wear, True, True), y, wear, ("rotation",), seed=0).mean()
        self.assertLess(base, 0.7)
        self.assertGreaterEqual(fixed, 0.95)


if __name__ == "__main__":
    unittest.main()
