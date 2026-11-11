"""python3 -m unittest discover -s tools -p 'test_*.py'"""
import json, pathlib, subprocess, sys, tempfile, unittest, wave
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_alignment

HEADER = "t,ax,ay,az,rx,ry,rz,pitch,roll,yaw,gx,gy,gz,qw,qx,qy,qz"


def bursts(times, t, width=0.15):
    """Smooth bumps at the given times (seconds), sampled at t."""
    return sum(np.exp(-0.5 * ((t - c) / width) ** 2) for c in times)


class BestLagTest(unittest.TestCase):
    def test_recovers_injected_lag_within_one_sample(self):
        rng = np.random.default_rng(0)
        t = np.arange(0, 30, 1 / 50)
        speech = bursts(rng.uniform(1, 29, 25), t)
        for lag in [-37, -5, 0, 4, 12, 50]:
            motion = np.roll(speech, lag) + rng.normal(0, 0.05, len(t))
            found, r = check_alignment.best_lag(speech, motion, 50)
            self.assertLessEqual(abs(found - lag), 1, f"injected {lag}, found {found}")
            self.assertGreater(r, 0.8)


class SessionTest(unittest.TestCase):
    def test_session_with_motion_200ms_after_the_audio(self):
        """Audio starts 0.5 s into the motion clock; jaw motion trails each spoken word by 200 ms."""
        rng = np.random.default_rng(1)
        words = rng.uniform(1, 14, 12)
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp, "2026-11-11T050000-normal")
            d.mkdir()
            sr, offset = 16000, 0.5
            ta = np.arange(0, 15, 1 / sr)
            pcm = (0.5 * bursts(words - offset, ta) * np.sin(2 * np.pi * 220 * ta) * 32767).astype("<i2")
            with wave.open(str(d / "a.wav"), "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr), w.writeframes(pcm.tobytes())
            subprocess.run(["afconvert", "-f", "caff", "-d", "LEI16", str(d / "a.wav"), str(d / "audio.caf")], check=True)
            (d / "a.wav").unlink()
            tm = np.arange(0, 16, 1 / 50) + rng.normal(0, 0.002, 800)  # jittery ~50 Hz like CoreMotion
            rx = 0.3 * bursts(words + 0.2, tm) + rng.normal(0, 0.01, len(tm))
            cols = [tm, *np.zeros((3, len(tm))), rx, *np.zeros((12, len(tm)))]
            np.savetxt(d / "motion.csv", np.column_stack(cols), delimiter=",", header=HEADER, comments="", fmt="%.6f")
            (d / "meta.json").write_text(json.dumps({"audioStartOffsetSeconds": offset}))

            lag, r = check_alignment.check(d)
            self.assertAlmostEqual(lag, 0.2, delta=0.02)
            out = subprocess.run([sys.executable, check_alignment.__file__, str(d)], capture_output=True, text=True)
            self.assertEqual(out.returncode, 1)
            self.assertIn("FLAGGED", out.stdout)


if __name__ == "__main__":
    unittest.main()
