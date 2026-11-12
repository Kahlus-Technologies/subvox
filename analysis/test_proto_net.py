"""Checks for proto_net.py: python3 -m unittest discover -s analysis -p 'test_*.py'. Skipped without torch."""
import pathlib, subprocess, sys, tempfile, unittest
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    import torch
    import proto_net
except ImportError:
    torch = None

PEOPLE = [HERE / "fixtures" / f for f in ("dataset_reseat.json", "dataset_person_b.json", "dataset_person_c.json")]


@unittest.skipUnless(torch, "needs torch: pip install -r analysis/requirements-ml.txt")
class ProtoNetTest(unittest.TestCase):
    def test_net_has_15k_to_50k_parameters_and_takes_any_length(self):
        net = proto_net.make_net().eval()
        self.assertTrue(15_000 <= proto_net.n_params(net) <= 50_000)
        for n in (37, 100):
            self.assertEqual(tuple(net(torch.zeros(4, 9, n)).shape), (4, 32))

    def test_window_centres_short_takes_and_keeps_the_middle_of_long_ones(self):
        x = np.arange(1, 11, dtype=float)[:, None].repeat(9, 1)
        w = proto_net.window(x, 14)
        self.assertEqual(w[:, 0].tolist(), [0, 0] + list(range(1, 11)) + [0, 0])
        self.assertEqual(proto_net.window(x, 4)[:, 0].tolist(), [4, 5, 6, 7])

    def test_decide_picks_the_nearest_word_and_the_apps_margin(self):
        D = np.array([[1.0, 4.0, 2.0], [3.0, 3.0, 9.0]])  # queries x enrolment takes of words 7, 7, 8
        word, margin = proto_net.decide(D, np.array([7, 7, 8]))
        self.assertEqual(word.tolist(), [7, 7])
        np.testing.assert_allclose(margin, [0.5, 1 - 3 / 9])

    def test_folds_hold_out_a_wear_or_a_person_and_enrol_from_what_is_left(self):
        who = np.repeat([0, 1], 24); local = np.tile(np.repeat([0, 1], 12), 2); wear = who * 2 + local
        y = np.tile(np.repeat([0, 1], 6), 4)
        F = proto_net.folds(y, who, wear, local)
        self.assertEqual([s for s, *_ in F], ["leave-one-wear-out"] * 4 + ["leave-one-person-out"] * 2)
        self.assertEqual([n for _, n, *_ in F], ["A0", "A1", "B0", "B1", "A", "B"])
        split, _, tr, en, q = F[2]  # B's first wear held out
        self.assertEqual(set(q), set(range(24, 36)))
        self.assertTrue(set(en) <= set(range(36, 48)) and len(en) == 10)
        self.assertFalse(set(tr) & set(q))
        split, _, tr, en, q = F[5]  # B held out: enrol its first 5 takes of each word, test the rest
        self.assertEqual(sorted(en), list(range(24, 29)) + list(range(30, 35)))
        self.assertEqual(set(tr), set(range(24)))
        self.assertEqual(len(q), 14)

    def test_training_is_seeded(self):
        T, y, who, wear, local, _ = proto_net.load(PEOPLE[:1], 180.0)
        X, sd = proto_net.features(T, wear); idx = np.arange(len(T))
        a, b = (proto_net.embed(proto_net.train(X, sd, y, idx, 3, 5), [x / s for x, s in zip(X, sd)]) for _ in range(2))
        np.testing.assert_array_equal(a, b)

    def test_end_to_end_on_the_synthetic_people(self):
        with tempfile.TemporaryDirectory() as d:
            results = pathlib.Path(d) / "RESULTS.md"
            args = [a for p in PEOPLE[:2] for a in ("--dataset", str(p))]
            out = subprocess.run([sys.executable, str(HERE / "proto_net.py"), *args, "--episodes", "10", "--shuffles", "20",
                                  "--results", str(results)], capture_output=True, text=True, check=True).stdout
            self.assertIn("72 takes", out)
            rows = [l for l in out.splitlines() if l.startswith("| leave-one-")]
            self.assertEqual([r.split(" | ")[:2] for r in rows],
                             [["| leave-one-wear-out", "DTW-1NN, 5 takes per word"], ["| leave-one-wear-out", "proto-net, 5-take prototypes"],
                              ["| leave-one-person-out", "DTW-1NN, 5 takes per word"], ["| leave-one-person-out", "proto-net, 5-take prototypes"]])
            self.assertIn(out, results.read_text())


if __name__ == "__main__":
    unittest.main()
