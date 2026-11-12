"""python3 -m unittest discover -s tools -p 'test_*.py'"""
import json, pathlib, subprocess, sys, tempfile, unittest
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_dataset

HEADER = "t,ax,ay,az,rx,ry,rz,pitch,roll,yaw,gx,gy,gz,qw,qx,qy,qz"
TOOL = pathlib.Path(__file__).resolve().parent / "build_dataset.py"
RATE = 50  # Hz
# Spoken-take word boundaries (seconds), deliberately off the 50 Hz sample grid.
WORDS = [("the", 0.61, 1.03), ("birch", 1.13, 1.59), ("canoe", 1.71, 2.37), ("slid", 2.52, 3.05),
         ("on", 3.19, 3.48), ("planks", 3.66, 4.41)]


def articulation(times, channels, seed=0):
    """Synthetic jaw motion: a smooth bump every ~0.1 s between 0.4 and 4.6 s, random size and sign per
    channel, flat before and after. A function of time, so it can be sampled on a warped clock."""
    rng = np.random.default_rng(seed)
    centers = np.arange(0.4, 4.6, 0.1)
    centers += rng.uniform(-0.03, 0.03, len(centers))
    widths = rng.uniform(0.04, 0.07, len(centers))
    amps = rng.normal(0, 1, (len(centers), channels))
    bumps = np.exp(-0.5 * ((np.asarray(times)[:, None] - centers) / widths) ** 2)
    return bumps @ amps


def warp(spoken_t, lead, slope, wobble):
    """Monotone map from spoken-take time to silent-take time: a different lead-in, a different pace, and a
    pace that drifts during the sentence."""
    return lead + slope * spoken_t + wobble * np.sin(2 * np.pi * spoken_t / 2.5)


def silent_copy(warp_args, seconds, channels, seed=1):
    """(t, motion) of a silent take: the spoken motion played back on the warped clock, 0.6x as strong,
    with sensor noise."""
    t = np.arange(0, seconds, 1 / RATE)
    grid = np.linspace(-2, 8, 20001)
    spoken_time = np.interp(t, warp(grid, *warp_args), grid)  # invert the monotone warp
    noise = np.random.default_rng(seed).normal(0, 0.02, (len(t), channels))
    return t, 0.6 * articulation(spoken_time, channels) + noise


def write_session(root, name, speaker, prompt, words, seconds=4.0, meta=True, with_words=True):
    d = root / name
    d.mkdir(parents=True)
    t = np.arange(0, seconds, 0.02)  # 50 Hz
    motion = np.column_stack([t] + [np.sin(t * (k + 1)) for k in range(16)])
    np.savetxt(d / "motion.csv", motion, delimiter=",", header=HEADER, comments="", fmt="%.6f")
    if meta:
        (d / "meta.json").write_text(json.dumps({"speaker": speaker, "prompt": prompt, "condition": "Normal speech"}))
    if with_words:
        (d / "words.json").write_text(json.dumps({"text": prompt, "words": words}))


class BuildDatasetTest(unittest.TestCase):
    def test_counts_and_slices(self):
        with tempfile.TemporaryDirectory() as tmp:
            mine, friend, out = pathlib.Path(tmp, "mine"), pathlib.Path(tmp, "friend/Sessions"), pathlib.Path(tmp, "out")
            write_session(mine, "s1-normal", "Aayu", "The birch canoe",
                          [{"text": "The", "start": 0.5, "end": 0.7}, {"text": "birch", "start": 0.8, "end": 1.2},
                           {"text": "canoe.", "start": 1.3, "end": 1.9}])
            write_session(mine, "s2-normal", "Aayu", "Glue the sheet",
                          [{"text": "Glue", "start": 1.0, "end": 1.4}, {"text": "the", "start": 1.5, "end": 1.6},
                           {"text": "sheet", "start": 9.0, "end": 9.5}])  # past the end of motion: dropped
            write_session(friend, "s3-normal", "Maya", "The canoe", [{"text": "the", "start": 0.2, "end": 0.4},
                                                                    {"text": "canoe", "start": 0.5, "end": 1.0}])
            write_session(mine, "s4-normal", "Aayu", "Not labelled", [], with_words=False)
            (mine / "s5-empty").mkdir()

            subprocess.run([sys.executable, TOOL, mine, friend.parent / "Sessions", "--out", out],
                           check=True, capture_output=True)

            summary = (out / "summary.txt").read_text()
            for line in ["sessions: 3", "speakers: 2", "minutes: 0.2", "words: 7", "unique words: 4",
                         "words dropped (no text or no motion in their window): 1", "sessions skipped: 2",
                         "  no motion.csv: 1", "  no words.json: 1"]:
                self.assertIn(line + "\n", summary)

            data = np.load(out / "words.npz")
            self.assertEqual(list(data["word"]), ["the", "birch", "canoe", "glue", "the", "the", "canoe"])
            self.assertEqual(list(data["speaker"]), ["Aayu"] * 5 + ["Maya"] * 2)
            self.assertEqual(data["prompt"][2], "The birch canoe")
            # "birch" spans 0.8..1.2 s at 50 Hz: 21 samples of the 9 channels, starting at t = 0.8.
            birch = data["motion"][data["offsets"][1]:data["offsets"][2]]
            self.assertEqual(birch.shape, (21, 9))
            self.assertEqual(birch.dtype, np.float32)
            self.assertAlmostEqual(float(birch[0, 0]), np.sin(0.8), places=5)
            self.assertEqual(data["offsets"][-1], len(data["motion"]))


class TransferTest(unittest.TestCase):
    """Silent take = time-warped copy of the spoken take: mapped word boundaries land within 2 samples."""

    def test_word_boundaries_within_two_samples(self):
        t = np.arange(0, 5, 1 / RATE)
        spoken = articulation(t, 9)
        spans = [(start, end) for _, start, end in WORDS]
        cases = {"slower, later start": ((0.4, 1.15, 0.15), 6.6), "faster, earlier start": ((-0.2, 0.8, 0.1), 4.2)}
        for name, (warp_args, seconds) in cases.items():
            with self.subTest(name):
                silent_t, silent = silent_copy(warp_args, seconds, 9)
                mapped = np.array(build_dataset.transfer(spans, t, spoken, silent_t, silent))
                truth = warp(np.array(spans), *warp_args)
                error = np.abs(mapped - truth) * RATE
                self.assertLessEqual(error.max(), 2, f"boundary errors in samples:\n{error.round(2)}")

    def test_dtw_path_recovers_a_repeated_sample(self):
        a = np.array([[0.0], [1.0], [2.0], [3.0]])
        b = np.array([[0.0], [1.0], [1.0], [2.0], [3.0]])
        self.assertEqual(build_dataset.dtw_path(a, b).tolist(), [[0, 0], [1, 1], [1, 2], [2, 3], [3, 4]])


class PairedSessionTest(unittest.TestCase):
    def test_silent_take_gets_its_spoken_partners_words(self):
        warp_args = (0.4, 1.15, 0.15)
        with tempfile.TemporaryDirectory() as tmp:
            root, out = pathlib.Path(tmp, "Sessions"), pathlib.Path(tmp, "out")
            spoken_dir, silent_dir = root / "s1-normal", root / "s2-silent"
            for d in [spoken_dir, silent_dir, root / "s3-silent"]:
                d.mkdir(parents=True)
            t = np.arange(0, 5, 1 / RATE)
            np.savetxt(spoken_dir / "motion.csv", np.column_stack([t, articulation(t, 16)]),
                       delimiter=",", header=HEADER, comments="", fmt="%.6f")
            silent_t, silent = silent_copy(warp_args, 6.6, 16)
            np.savetxt(silent_dir / "motion.csv", np.column_stack([silent_t, silent]),
                       delimiter=",", header=HEADER, comments="", fmt="%.6f")
            np.savetxt(root / "s3-silent" / "motion.csv", np.column_stack([silent_t, silent]),
                       delimiter=",", header=HEADER, comments="", fmt="%.6f")
            meta = {"speaker": "Maya", "prompt": "The birch canoe slid on the planks."}
            (spoken_dir / "meta.json").write_text(json.dumps({**meta, "condition": "Normal speech", "pairId": "P1"}))
            (silent_dir / "meta.json").write_text(json.dumps({**meta, "condition": "Silent mouthing", "pairId": "P1"}))
            (root / "s3-silent" / "meta.json").write_text(json.dumps({**meta, "condition": "Silent mouthing"}))
            # The last word is past the end of the spoken motion: dropped on both takes.
            (spoken_dir / "words.json").write_text(json.dumps(
                {"words": [{"text": w, "start": s, "end": e} for w, s, e in WORDS + [("smooth", 5.2, 5.6)]]}))
            # Phonon run on silent audio: made-up words that must never reach the dataset.
            bogus = json.dumps({"words": [{"text": "uh", "start": 1.0, "end": 1.4}]})
            (silent_dir / "words.json").write_text(bogus)
            (root / "s3-silent" / "words.json").write_text(bogus)

            subprocess.run([sys.executable, TOOL, root, "--out", out], check=True, capture_output=True)

            summary = (out / "summary.txt").read_text()
            for line in ["sessions: 2", "words: 12", "words on silent takes, timed from their paired spoken take: 6",
                         "words dropped (no text or no motion in their window): 2",
                         "sessions skipped: 1", "  silent take with no paired spoken take: 1"]:
                self.assertIn(line + "\n", summary)
            data = np.load(out / "words.npz")
            on_silent = data["session"] == "s2-silent"
            self.assertEqual(list(data["word"][on_silent]), [w for w, _, _ in WORDS])
            self.assertEqual(set(data["labels_from"][on_silent]), {"s1-normal"})
            self.assertEqual(set(data["labels_from"][~on_silent]), {""})
            truth = warp(np.array([(s, e) for _, s, e in WORDS]), *warp_args)
            mapped = np.column_stack([data["start"][on_silent], data["end"][on_silent]])
            self.assertLessEqual(np.abs(mapped - truth).max() * RATE, 2)
            # The silent rows slice the silent take's motion, not the spoken take's.
            first = np.flatnonzero(on_silent)[0]
            lo = np.searchsorted(silent_t, data["start"][first])
            row = data["motion"][data["offsets"][first]:data["offsets"][first + 1]]
            np.testing.assert_allclose(row[0], silent[lo, :9], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
