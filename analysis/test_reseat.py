"""Checks for reseat.py: python3 -m unittest discover -s analysis -p 'test_*.py'"""
import pathlib, sys, unittest
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import reseat
from clean_test_lib import load_takes

FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures/dataset_reseat.json"


class WearsTest(unittest.TestCase):
    def test_gap_over_three_minutes_starts_a_new_wear(self):
        T = [{"createdAt": c} for c in (0, 20, 200, 381, 400)]  # gaps 20, 180, 181, 19 s
        self.assertEqual(reseat.wears(T).tolist(), [0, 0, 0, 1, 1])
        self.assertEqual(reseat.wears(T, gap=60).tolist(), [0, 0, 1, 2, 2])

    def test_wear_field_wins_over_gaps(self):
        T = [{"createdAt": 0, "wear": "a"}, {"createdAt": 1000, "wear": "a"}, {"createdAt": 1001, "wear": "b"}]
        self.assertEqual(reseat.wears(T).tolist(), [0, 0, 1])


class GravityFrameTest(unittest.TestCase):
    def test_gravity_frame_undoes_a_tilt_per_wear(self):
        """The fixture tilts wears 1 and 2 by +-25 degrees: templates stop transferring until the frame fixes it."""
        T, words = load_takes(FIXTURE)
        wear = reseat.wears(T); y = np.array([words.index(t["label"]) for t in T]); other = wear[:, None] != wear[None, :]
        acc = lambda gravity: np.mean(y[reseat.nearest(reseat.distances(T, wear, False, gravity), other)] == y)
        self.assertLess(acc(False), 0.7)
        self.assertEqual(acc(True), 1.0)


if __name__ == "__main__":
    unittest.main()
