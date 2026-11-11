"""Check that a session's audio and motion line up, so words.json labels land on the right motion.

words.json times are shifted onto the motion clock by meta.json audioStartOffsetSeconds. This places the
audio on the motion clock with that same offset, then cross-correlates the audio loudness (RMS envelope at
50 Hz) with head-motion energy (norm of the rotation rate) over lags of -1 s..+1 s. The best lag should be
near 0. A positive lag means motion comes after the audio. Sessions with |lag| > 100 ms are flagged.

Usage: python3 tools/check_alignment.py SESSION_OR_FOLDER [...]
    A folder of sessions (e.g. ~/Documents/Subvox/Sessions) checks each one and prints a summary.
Needs numpy and macOS afconvert (to read audio.caf).
"""
import argparse, json, pathlib, subprocess, sys, tempfile, wave
import numpy as np

RATE = 50  # Hz, the grid both signals are put on
MAX_LAG = 1.0  # seconds
FLAG = 0.100  # seconds


def best_lag(audio, motion, max_lag):
    """Lag in samples (motion later than audio is positive) with the highest correlation, and that correlation."""
    best = (0, -np.inf)
    for lag in range(-max_lag, max_lag + 1):
        a, m = (audio[:len(audio) - lag], motion[lag:]) if lag >= 0 else (audio[-lag:], motion[:len(motion) + lag])
        n = min(len(a), len(m))
        a, m = a[:n], m[:n]
        if n < 2 or a.std() == 0 or m.std() == 0:
            continue
        r = float(np.corrcoef(a, m)[0, 1])
        if r > best[1]:
            best = (lag, r)
    return best


def audio_envelope(path):
    """(RMS per 1/RATE s frame, frame start times in seconds from the start of the audio file)."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = pathlib.Path(tmp) / "audio.wav"
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(path), str(wav)],
                       check=True, capture_output=True)
        with wave.open(str(wav)) as w:
            sr = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768
    hop = sr // RATE
    frames = x[:len(x) // hop * hop].reshape(-1, hop)
    return np.sqrt((frames ** 2).mean(axis=1)), np.arange(len(frames)) / RATE


def check(d):
    """(lag seconds, correlation) for one session folder, or raise ValueError saying why it can't be checked."""
    audio = d / "audio.wav" if (d / "audio.wav").exists() else d / "audio.caf"
    if not (d / "motion.csv").exists() or not audio.exists():
        raise ValueError("needs motion.csv and audio")
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    # Subvox Dictate sessions (audio.wav) are on the audio clock already.
    offset = 0.0 if audio.name == "audio.wav" else meta.get("audioStartOffsetSeconds")
    if offset is None:
        raise ValueError("no recorded audio offset (crashed session)")
    m = np.loadtxt(d / "motion.csv", delimiter=",", skiprows=1, usecols=(0, 4, 5, 6), ndmin=2)
    env, env_t = audio_envelope(audio)
    grid = np.arange(m[0, 0], m[-1, 0], 1 / RATE)
    motion = np.interp(grid, m[:, 0], np.linalg.norm(m[:, 1:], axis=1))
    loud = np.interp(grid, env_t + offset, env, left=0, right=0)
    lag, r = best_lag(loud, motion, int(MAX_LAG * RATE))
    return lag / RATE, r


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=pathlib.Path)
    args = p.parse_args()
    sessions = []
    for path in args.paths:
        path = path.expanduser()
        sessions += [path] if (path / "motion.csv").exists() else sorted(q for q in path.iterdir() if q.is_dir())
    lags, flagged, skipped = [], [], 0
    for d in sessions:
        try:
            lag, r = check(d)
        except (ValueError, OSError, subprocess.CalledProcessError) as e:
            skipped += 1
            print(f"{d.name}  SKIPPED: {e}")
            continue
        lags.append(lag)
        bad = abs(lag) > FLAG
        if bad:
            flagged.append(d.name)
        print(f"{d.name}  lag {lag * 1000:+.0f} ms  peak r {r:.2f}{'  FLAGGED' if bad else ''}")
    if len(sessions) > 1:
        median = f", median lag {np.median(lags) * 1000:+.0f} ms" if lags else ""
        print(f"\n{len(lags)} checked, {len(flagged)} flagged (|lag| > {FLAG * 1000:.0f} ms), {skipped} skipped{median}")
    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
