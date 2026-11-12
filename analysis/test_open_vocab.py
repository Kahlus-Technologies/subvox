"""python3 -m unittest discover -s analysis -p 'test_*.py'

Needs requirements.txt; the end-to-end test also needs requirements-ml.txt (torch) and is skipped without it.
All data here is synthetic (analysis/fixtures/make_sessions.py): it checks the wiring, not AirPods.
"""
import argparse, contextlib, importlib.util, io, json, pathlib, subprocess, sys, tempfile, unittest
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import open_vocab


class LexiconTest(unittest.TestCase):
    def test_cmudict_and_mfa_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = pathlib.Path(tmp, "lexicon.txt")
            f.write_text(";;; CMUdict\nTHE  DH AH0\nTHE(2)  DH IY0\nDON'T  D OW1 N T\n"
                         "real\t0.99\t0.12\t1.0\t0.8\tR IY1 L\nODD  QQ1 AA1\n")  # MFA .dict line; QQ is not ARPAbet
            self.assertEqual(open_vocab.load_lexicon(f), {"the": ["DH", "AH"], "don't": ["D", "OW", "N", "T"],
                                                          "real": ["R", "IY", "L"]})


class ResultsFileTest(unittest.TestCase):
    def test_block_goes_in_its_section_and_is_replaced_on_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = pathlib.Path(tmp, "RESULTS.md")
            f.write_text(f"# Results\n\n{open_vocab.HEADING}\n\nprose\n\n## Other analysis\n\nkeep me\n")
            open_vocab.write_results(f, "synthetic", "first run\n")
            open_vocab.write_results(f, "real", "real run\n")
            open_vocab.write_results(f, "synthetic", "second run\n")
            text = f.read_text()
            self.assertNotIn("first run", text)
            self.assertLess(text.index("prose"), text.index("second run"))
            self.assertLess(text.index("real run"), text.index("## Other analysis\n\nkeep me\n"))
            self.assertEqual(text.count(open_vocab.HEADING), 1)

    def test_new_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = pathlib.Path(tmp, "RESULTS.md")
            open_vocab.write_results(f, "real", "numbers\n")
            self.assertEqual(f.read_text(), f"# Results\n\n{open_vocab.HEADING}\n\n<!-- open_vocab.py real -->\n"
                                            "numbers\n<!-- /open_vocab.py real -->\n")


class SilentTakeTest(unittest.TestCase):
    def test_paired_silent_take_is_left_out(self):
        """A silent take borrows its spoken partner's words but has no phones.json; loading must not fail on it."""
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            t = np.arange(100) / 50
            motion = np.column_stack([t, np.sin(np.outer(t, np.arange(1, 10)))] + [np.zeros((100, 7))])
            words = [{"text": "the", "start": 0.2, "end": 0.5}, {"text": "canoe", "start": 0.6, "end": 1.2}]
            for name, condition in (("s1-normal", "Normal speech"), ("s1-silent", "Silent mouthing")):
                d = root / name
                d.mkdir()
                np.savetxt(d / "motion.csv", motion, delimiter=",", comments="",
                           header="t,ax,ay,az,rx,ry,rz,pitch,roll,yaw,gx,gy,gz,qw,qx,qy,qz")
                (d / "meta.json").write_text(json.dumps({"speaker": "A", "prompt": "The canoe", "pairId": "p1",
                                                         "condition": condition}))
                (d / "words.json").write_text(json.dumps({"words": words}))
            (root / "s1-normal" / "phones.json").write_text(json.dumps({"words": words, "phones": [
                {"phone": "DH", "start": 0.2, "end": 0.35}, {"phone": "AH0", "start": 0.35, "end": 0.5},
                {"phone": "K", "start": 0.6, "end": 0.8}, {"phone": "AH0", "start": 0.8, "end": 0.95},
                {"phone": "N", "start": 0.95, "end": 1.05}, {"phone": "UW1", "start": 1.05, "end": 1.2}]}))
            with contextlib.redirect_stdout(io.StringIO()) as log:
                rows, source = open_vocab.load(argparse.Namespace(npz=None, sessions=[root], lexicon=None))
            self.assertIn("left out 1 silent takes", log.getvalue())
            self.assertEqual([(r["session"], r["word"], r["phones"]) for r in rows],
                             [("s1-normal", "the", ["DH", "AH"]), ("s1-normal", "canoe", ["K", "AH", "N", "UW"])])


@unittest.skipUnless(importlib.util.find_spec("torch"), "needs torch: pip install -r analysis/requirements-ml.txt")
class EndToEndTest(unittest.TestCase):
    def test_synthetic_sessions_beat_chance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, lexicon, results = (pathlib.Path(tmp, n) for n in ("Sessions", "lexicon.txt", "RESULTS.md"))
            subprocess.run([sys.executable, HERE / "fixtures/make_sessions.py", root, "--lexicon", lexicon],
                           check=True, capture_output=True)
            for d in sorted(root.iterdir())[::10]:  # not aligned yet: these get phonemes from the lexicon
                (d / "phones.json").unlink()
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                rows = open_vocab.main(["--sessions", str(root), "--lexicon", str(lexicon), "--perms", "100",
                                        "--results", str(results)])
            log = log.getvalue()
            self.assertIn("word times from phones.json in 81 sessions, from words.json in the other 9", log)
            self.assertRegex(log, r"from the lexicon: [1-9]\d*, neither \(left out\): 0")
            self.assertEqual([(r["split"].split(" (")[0], r["model"]) for r in rows],
                             [("held-out sessions", "DTW-1NN"), ("held-out sessions", "CTC"),
                              ("held-out speakers", "DTW-1NN"), ("held-out speakers", "CTC")])
            for r in rows:
                with self.subTest(split=r["split"], model=r["model"]):
                    self.assertGreater(r["acc"], 2 * r["acc_chance"])
                    self.assertLess(r["acc_p"], 0.05)
                    self.assertLess(r["cer"], r["cer_chance"])
                    self.assertLess(r["cer_p"], 0.05)
            dtw_unseen, ctc_unseen = rows[0]["unseen"], rows[1]["unseen"]
            self.assertEqual(dtw_unseen["acc"], 0)  # nearest neighbour can only name words it has seen
            self.assertLess(ctc_unseen["cer"], ctc_unseen["cer_chance"])  # CTC spells words it never trained on
            self.assertLess(ctc_unseen["cer_p"], 0.05)
            text = results.read_text()
            self.assertIn("**Synthetic fixture, not recordings.**", text)
            self.assertIn("| held-out speakers | CTC | ", text)


if __name__ == "__main__":
    unittest.main()
