"""Phone-level timings for Subvox sessions: phones.json next to words.json.

Montreal Forced Aligner places each of English's ~40 ARPAbet phonemes (AA, B, CH, ...) in time, given
the audio and the transcript. That gives the exact window of AirPods motion for every sound, which the
sound-based model (#11) trains on. Word boundaries come out of the same alignment, tighter than the
ASR's own word times.

Setup once: micromamba create -p ~/.local/mfa -c conda-forge montreal-forced-aligner python=3.11
            for k in acoustic dictionary g2p; do ~/.local/mfa/bin/mfa model download $k english_us_arpa; done
Run:        python3 tools/align_phones.py            (every session with words.json and no phones.json)

phones.json: {"phones": [{"phone": "K", "start": s, "end": s}, ...], "words": [...], "clock": ...}
Times use the same clock as words.json and motion.csv. Words missing from the pronunciation dictionary
get one guessed by MFA's G2P model; any word still left without phones is listed as "oov".
"""
import json, os, pathlib, re, shutil, subprocess, sys, tempfile

root = pathlib.Path(os.environ.get("SUBVOX_SESSIONS", pathlib.Path.home() / "Documents/Subvox/Sessions"))
mfa = pathlib.Path(os.environ.get("MFA", pathlib.Path.home() / ".local/mfa/bin/mfa"))
dictionary = pathlib.Path.home() / "Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict"


def tiers(textgrid):
    """Minimal TextGrid reader: {tier name: [(start, end, label)]}, skipping empty intervals."""
    out, name = {}, None
    for block in re.split(r"\n\s*item \[\d+\]:", textgrid)[1:]:
        name = re.search(r'name = "(.*?)"', block).group(1)
        out[name] = [(float(a), float(b), t) for a, b, t in
                     re.findall(r'xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "(.*?)"', block) if t.strip()]
    return out


def align(sessions):
    env = dict(os.environ, PATH=f"{mfa.parent}:{os.environ['PATH']}")  # MFA shells out to openfst/kaldi binaries
    with tempfile.TemporaryDirectory() as tmp:
        corpus, out = pathlib.Path(tmp, "corpus"), pathlib.Path(tmp, "out")
        corpus.mkdir()
        offsets, vocab = {}, set()
        for d in sessions:
            words = json.loads((d / "words.json").read_text())
            audio = d / "audio.wav" if (d / "audio.wav").exists() else d / "audio.caf"
            wav = corpus / f"{d.name}.wav"
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", audio, wav], check=True)
            # words.json times are on the motion clock; the audio file starts at the session's audio offset.
            meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
            offsets[d.name] = meta.get("audioStartOffsetSeconds") or 0
            text = re.sub(r"[^\w' ]", " ", words["text"]).lower()
            (corpus / f"{d.name}.lab").write_text(text)
            vocab |= set(text.split())
        # Unknown words would align as one "spn" blob with no phones: guess their pronunciation first.
        known = {line.split(None, 1)[0] for line in dictionary.read_text().splitlines() if line.strip()}
        lexicon = pathlib.Path(tmp, "lexicon.dict")
        shutil.copy(dictionary, lexicon)
        if missing := sorted(vocab - known):
            pathlib.Path(tmp, "oov.txt").write_text("\n".join(missing))
            subprocess.run([mfa, "g2p", pathlib.Path(tmp, "oov.txt"), "english_us_arpa", pathlib.Path(tmp, "oov.dict"),
                            "--quiet"], check=True, capture_output=True, env=env)
            with lexicon.open("a") as f:
                f.write("\n" + pathlib.Path(tmp, "oov.dict").read_text())
        r = subprocess.run([mfa, "align", corpus, lexicon, "english_us_arpa", out, "--clean",
                            "--single_speaker", "--quiet", "--use_mp", "false"], capture_output=True, text=True, env=env)
        if r.returncode:
            sys.exit("mfa failed: " + r.stderr[-800:])
        for d in sessions:
            grid = out / f"{d.name}.TextGrid"
            if not grid.exists():
                print(d.name, "SKIPPED: no alignment (audio and transcript disagree too much)")
                continue
            t, off = tiers(grid.read_text()), offsets[d.name]
            phones = [{"phone": p, "start": round(a + off, 3), "end": round(b + off, 3)} for a, b, p in t["phones"]]
            words = [{"text": w, "start": round(a + off, 3), "end": round(b + off, 3)} for a, b, w in t["words"]]
            (d / "phones.json").write_text(json.dumps({
                "phones": phones, "words": words, "aligner": "MFA english_us_arpa",
                # Words the dictionary lacks get one "spn" (spoken noise) phone instead of real ones.
                "oov": [w["text"] for w in words
                        if any(p["phone"] == "spn" and p["start"] >= w["start"] and p["end"] <= w["end"] for p in phones)],
                "clock": "start/end are on the motion.csv t clock"}, indent=1))
            print(d.name, len(phones), "phones,", len(words), "words")


if __name__ == "__main__":
    if not mfa.exists():
        sys.exit(f"MFA not found at {mfa}; see setup in this file's docstring")
    todo = [d for d in sorted(root.iterdir()) if d.is_dir() and (d / "words.json").exists() and not (d / "phones.json").exists()]
    if todo:
        align(todo)
