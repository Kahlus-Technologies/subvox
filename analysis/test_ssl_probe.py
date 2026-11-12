"""Checks for ssl_probe.py: python3 -m unittest discover -s analysis -p 'test_*.py'. Skipped without torch; the
end-to-end run is skipped when torch.hub cannot load the encoder (offline with nothing cached)."""
import pathlib, subprocess, sys, tempfile, unittest
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    import torch
    import ssl_probe
except ImportError:
    torch = None

PEOPLE = [HERE / "fixtures" / f for f in ("dataset_reseat.json", "dataset_person_b.json")]


@unittest.skipUnless(torch, "needs torch: pip install -r analysis/requirements-ml.txt")
class ProbeTest(unittest.TestCase):
    def test_accel_is_resampled_to_30_hz_with_gravity_and_centred_in_a_still_window(self):
        t = np.arange(50) / 50.0
        x = np.zeros((50, 9)); x[:, 0] = np.sin(2 * np.pi * t)
        T = [{"samples": [{"gx": 0.0, "gy": 0.0, "gz": -1.0}] * 50}]
        a = ssl_probe.accel([x], T, np.array([0]), 150)[0]
        self.assertEqual(a.shape, (3, 150))
        lo = (150 - 30) // 2  # 0.98 s at 30 Hz is 30 samples
        np.testing.assert_allclose(a[0, lo:lo + 30], np.sin(2 * np.pi * np.arange(30) / 30), atol=0.01)
        np.testing.assert_allclose(a[:, :lo], np.tile([[0.0], [0.0], [-1.0]], lo))
        np.testing.assert_allclose(a[:, lo + 30:], np.tile([[0.0], [0.0], [-1.0]], 150 - lo - 30))

    def test_logistic_regression_separates_blobs_and_refits_every_shuffle(self):
        rng = np.random.default_rng(0)
        y = np.repeat([0, 1, 2], 10); E = rng.normal(size=(30, 8)); E[np.arange(30), y] += 4
        tr, q = np.arange(0, 30, 2), np.arange(1, 30, 2)
        rule = ssl_probe.lr_rule(E, tr, q, 3)
        pred, margin = rule(y)
        self.assertEqual(pred.tolist(), y[q].tolist())
        self.assertTrue(np.all((0 <= margin) & (margin <= 1)))
        S = np.array([rng.permutation(y) for _ in range(4)])
        np.testing.assert_array_equal(rule.many(S)[0], rule(S[0])[0])


@unittest.skipUnless(torch, "needs torch: pip install -r analysis/requirements-ml.txt")
class EndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ssl_probe.encoder("harnet5")
        except Exception as e:  # offline, rate-limited or blocked: nothing to test against
            raise unittest.SkipTest(f"torch.hub could not load harnet5: {e!r}")

    def test_end_to_end_on_two_synthetic_people(self):
        with tempfile.TemporaryDirectory() as d:
            results = pathlib.Path(d) / "RESULTS.md"
            args = [a for p in PEOPLE for a in ("--dataset", str(p))]
            out = subprocess.run([sys.executable, str(HERE / "ssl_probe.py"), *args, "--episodes", "10", "--shuffles", "20",
                                  "--results", str(results)], capture_output=True, text=True, check=True).stdout
            methods = [l.split(" | ")[1] for l in out.splitlines() if l.startswith("| leave-one-wear-out")]
            self.assertEqual(methods, ["DTW-1NN, 5 takes per word", "proto-net, 5-take prototypes",
                                       "harnet5 + logistic regression, all training takes", "harnet5, 5-take prototypes"])
            self.assertIn(out, results.read_text())


if __name__ == "__main__":
    unittest.main()
