"""Writes synthetic talk-mode session folders for analysis/open_vocab.py. They are not recordings.

Each folder has what the app and tools write for a real Collect session: motion.csv (the app's 17 columns at
~50 Hz, t from 0), meta.json (SessionRecorder's keys), events.csv, words.json (Phonon-style word times on the
motion clock, a few tens of ms off the phone alignment) and phones.json (tools/align_phones.py's format,
stressed ARPAbet). As in Collect, every session is one sentence and every speaker reads the same sentences in
the same order. Speakers are named "Synthetic A", "Synthetic B", ...; open_vocab.py labels their results synthetic.

The signal: every phoneme has a fixed 6-D motion pattern (phonemes that look alike on the lips, like P/B/M,
share most of it), held for the phone's duration and smoothed into its neighbours, mixed by a per-speaker
matrix and a per-session seat matrix, plus noise and slow head drift. Acceleration and rotation rate carry
it; attitude is the integrated rotation. Deterministic for a given --seed.

Usage: python3 analysis/fixtures/make_sessions.py OUT [--speakers 3] [--sentences 30] [--seed 0] [--lexicon FILE]
    OUT must be empty or new. --lexicon FILE also writes the pronunciations in CMUdict format.
"""
import argparse, datetime, json, pathlib, sys
import numpy as np

RATE = 50.0
HEADER = "t,ax,ay,az,rx,ry,rz,pitch,roll,yaw,gx,gy,gz,qw,qx,qy,qz"
LOOK_ALIKE = ["P B M", "F V", "TH DH", "T D N L", "S Z", "SH ZH CH JH", "K G NG HH", "R ER", "W UW UH OW",
              "IY IH Y EY", "AA AE AH AY AW", "AO OY", "EH"]
# CMUdict pronunciations, most common words first (sentences draw words with Zipf-like weights).
LEXICON = """the DH AH0|a AH0|of AH1 V|to T UW1|and AH0 N D|in IH0 N|is IH1 Z|it IH1 T|was W AA1 Z|on AA1 N
for F AO1 R|with W IH1 DH|his HH IH1 Z|her HH ER1|at AE1 T|by B AY1|from F R AH1 M|they DH EY1|we W IY1
you Y UW1|she SH IY1|he HH IY1|this DH IH1 S|that DH AE1 T|not N AA1 T|but B AH1 T|all AO1 L|out AW1 T
up AH1 P|boy B OY1|girl G ER1 L|man M AE1 N|cat K AE1 T|dog D AO1 G|fish F IH1 SH|boat B OW1 T|lake L EY1 K
tree T R IY1|house HH AW1 S|road R OW1 D|water W AO1 T ER0|light L AY1 T|dark D AA1 R K|blue B L UW1
green G R IY1 N|red R EH1 D|small S M AO1 L|large L AA1 R JH|cold K OW1 L D|warm W AO1 R M|fast F AE1 S T
slow S L OW1|sat S AE1 T|ran R AE1 N|took T UH1 K|gave G EY1 V|made M EY1 D|found F AW1 N D|saw S AO1
held HH EH1 L D|bread B R EH1 D|cheese CH IY1 Z|juice JH UW1 S|salt S AO1 L T|rope R OW1 P|stone S T OW1 N
thumb TH AH1 M|shoe SH UW1|chair CH EH1 R|garden G AA1 R D AH0 N|morning M AO1 R N IH0 NG|young Y AH1 NG
evening IY1 V N IH0 NG|yellow Y EH1 L OW0|measure M EH1 ZH ER0|voice V OY1 S|sing S IH1 NG|wood W UH1 D
north N AO1 R TH|over OW1 V ER0|under AH1 N D ER0|river R IH1 V ER0|paper P EY1 P ER0|table T EY1 B AH0 L
quick K W IH1 K|bright B R AY1 T"""
LEX = dict((w, p.split()) for w, p in (e.split(None, 1) for e in LEXICON.replace("\n", "|").split("|")))


def session(rng, sentence, shape, mix, pace):
    """Motion rows, words.json and phones.json contents for one spoken sentence."""
    phones, words, clock = [], [], rng.uniform(0.4, 0.8)
    for w in sentence:
        start = clock
        for p in LEX[w]:
            d = pace * (rng.uniform(0.08, 0.16) if p[-1].isdigit() else rng.uniform(0.05, 0.1))
            phones.append((p, clock, clock + d))
            clock += d
        words.append((w, start, clock))
        clock += rng.uniform(0.02, 0.12)
    n = int((clock + 0.4) * RATE)
    t = np.maximum.accumulate(0.01 + np.arange(n) / RATE + rng.normal(0, 0.002, n))  # jittery, like CoreMotion
    target = np.zeros((n, 6))
    for p, a, b in phones:
        target[(t >= a) & (t < b)] = shape[p.rstrip("012")]
    k = np.hanning(7)[1:-1]
    target = np.stack([np.convolve(target[:, c], k / k.sum(), "same") for c in range(6)], 1)
    seat = np.eye(6) + 0.1 * rng.normal(size=(6, 6))  # each session is a slightly different fit in the ear
    x = target @ (seat @ mix).T + 0.3 * rng.normal(size=(n, 6))
    rot = 0.3 * x[:, 3:] + 0.01 * np.cumsum(rng.normal(size=(n, 3)), 0)
    att = rng.normal(0, 0.1, 3) + np.cumsum(rot, 0) / RATE
    motion = np.column_stack([t, 0.02 * x[:, :3], rot, att, np.tile([0, 0, -1.0, 1, 0, 0, 0], (n, 1))])
    shown = [w.capitalize() if i == 0 else w for i, w in enumerate(sentence)]
    shown[-1] += "."
    text = " ".join(shown)
    asr = []  # ASR word times sit a few tens of ms off the phone alignment
    for s, (_, a, b) in zip(shown, words):
        a = round(a + rng.normal(0, 0.03), 2)
        asr.append({"text": s, "start": a, "end": max(round(b + rng.normal(0, 0.03), 2), round(a + 0.04, 2))})
    words_json = {"text": text, "words": asr, "model": "synthetic", "clock": "start/end are on the motion.csv t clock"}
    phones_json = {"phones": [{"phone": p, "start": round(a, 3), "end": round(b, 3)} for p, a, b in phones],
                   "words": [{"text": w, "start": round(a, 3), "end": round(b, 3)} for w, a, b in words],
                   "aligner": "synthetic, in MFA english_us_arpa format", "oov": [],
                   "clock": "start/end are on the motion.csv t clock"}
    return motion, text, words_json, phones_json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=pathlib.Path)
    p.add_argument("--speakers", type=int, default=3)
    p.add_argument("--sentences", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lexicon", type=pathlib.Path)
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} is not empty; give a new or empty folder")
    rng = np.random.default_rng(args.seed)
    vocab = list(LEX)
    weights = 1 / (np.arange(len(vocab)) + 3.0)
    sentences = [list(rng.choice(vocab, rng.integers(5, 9), p=weights / weights.sum())) for _ in range(args.sentences)]
    shape = {}
    for group in LOOK_ALIKE:
        base = rng.normal(size=6)
        shape.update({ph: base + 0.4 * rng.normal(size=6) for ph in group.split()})
    for s in range(args.speakers):
        speaker = f"Synthetic {chr(ord('A') + s)}"
        mix = (np.eye(6) + 0.3 * rng.normal(size=(6, 6))) * rng.uniform(0.7, 1.3, 6)
        pace = rng.uniform(0.85, 1.15)
        for i, sentence in enumerate(sentences):
            started = datetime.datetime(2026, 11, 1 + s, 10) + datetime.timedelta(seconds=20 * i)
            d = args.out / f"{started.strftime('%Y-%m-%dT%H%M%S')}-000-normal"  # the app's folder names
            d.mkdir(parents=True)
            motion, text, words_json, phones_json = session(rng, sentence, shape, mix, pace)
            np.savetxt(d / "motion.csv", motion, delimiter=",", header=HEADER, comments="", fmt="%.6f")
            (d / "events.csv").write_text("start,end,kind,label,confidence\n")
            (d / "words.json").write_text(json.dumps(words_json, indent=1))
            (d / "phones.json").write_text(json.dumps(phones_json, indent=1))
            (d / "meta.json").write_text(json.dumps({
                "audioDevice": "none (synthetic)", "audioSampleRate": 48000, "audioStartOffsetSeconds": 0,
                "condition": "Normal speech", "durationSeconds": round(motion[-1, 0], 3), "events": 0,
                "motionSamples": len(motion), "note": "Synthetic session from analysis/fixtures/make_sessions.py, not a recording.",
                "prompt": text, "speaker": speaker, "startUptime": 1000.0 + 20 * i,
                "startedAt": started.strftime("%Y-%m-%dT%H:%M:%SZ")}, indent=2, sort_keys=True))
    if args.lexicon:
        lines = [";;; Synthetic fixture pronunciations, CMUdict format: WORD  PHONES"]
        lines += [f"{w.upper()}  {' '.join(ph)}" for w, ph in sorted(LEX.items())] + ["THE(2)  DH IY0"]
        args.lexicon.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.speakers * len(sentences)} synthetic sessions to {args.out}")


if __name__ == "__main__":
    main()
