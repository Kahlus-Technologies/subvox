"""python3 -m unittest discover -s tools -p 'test_*.py'"""
import json, pathlib, subprocess, sys, tempfile, unittest
import numpy as np

HEADER = "t,ax,ay,az,rx,ry,rz,pitch,roll,yaw,gx,gy,gz,qw,qx,qy,qz"
TOOL = pathlib.Path(__file__).resolve().parent / "build_dataset.py"


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


if __name__ == "__main__":
    unittest.main()
