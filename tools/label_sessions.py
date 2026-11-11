"""Write Phonon-2 words.json for every session that needs it.

Subvox sessions (audio.caf) get labels if they have none. Subvox Dictate sessions (audio.wav) are always
relabelled with Phonon-2, the most accurate model we run locally; the app's own labels are kept as
words.app.json. Dictate sessions use the audio clock already (offset 0)."""
import json, os, pathlib, subprocess, sys, tempfile

root = pathlib.Path(os.environ.get("SUBVOX_SESSIONS", pathlib.Path.home() / "Documents/Subvox/Sessions"))
phonon = pathlib.Path(sys.executable).parent / "phonon"

def end_aligned_offset(d):
    """No meta.json (crash, #6): audio and motion stop together, so offset = last motion t - audio length.
    On a healthy session this matches the app's recorded offset to 3 ms."""
    info = subprocess.run(["afinfo", d / "audio.caf"], capture_output=True, text=True).stdout
    frames = int(info.split(" valid frames")[0].rsplit(" ", 1)[-1])
    rate = float(info.split("Hz")[0].rsplit(",", 1)[-1].split()[-1])
    last_t = float((d / "motion.csv").read_text().strip().rsplit("\n", 1)[-1].split(",")[0])
    return last_t - frames / rate

def audio_of(d):
    return d / "audio.wav" if (d / "audio.wav").exists() else d / "audio.caf"

def needs_label(d):
    words = d / "words.json"
    if not audio_of(d).exists():
        return False
    if not words.exists():
        return True
    return d.name.endswith("-dictation") and "Phonon" not in json.loads(words.read_text()).get("model", "")

def label(d):
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {"recovered": True}
    offset = meta.get("audioStartOffsetSeconds")
    estimated = offset is None and meta.get("recovered")
    if estimated:  # crashed session: no meta.json, or the one the app recovers at launch
        offset = end_aligned_offset(d)
        print(d.name, f"WARNING: no recorded audio offset, estimated {offset:+.3f}s by aligning the ends")
    elif offset is None:
        print(d.name, "WARNING: no recorded audio offset, assuming 0")
        offset = 0
    out = subprocess.run([phonon, "transcribe", "--json", audio_of(d)], capture_output=True, text=True)
    if out.returncode or not out.stdout.strip():  # CAF left unfinalized by a crash: libsndfile rejects it, CoreAudio reads it
        with tempfile.TemporaryDirectory() as tmp:
            wav = pathlib.Path(tmp) / "audio.wav"
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", audio_of(d), wav], check=True)
            out = subprocess.run([phonon, "transcribe", "--json", wav], capture_output=True, text=True)
    if out.returncode or not out.stdout.strip():
        raise RuntimeError("phonon failed: " + out.stderr.strip()[-200:])
    j = json.loads(out.stdout)
    if estimated:
        j["offsetEstimated"] = "end-aligned (crashed session)"
    for w in j["words"]:
        w["start"] += offset
        w["end"] += offset
    j["clock"] = "start/end are on the motion.csv t clock"
    if (d / "words.json").exists() and not (d / "words.app.json").exists():
        (d / "words.json").rename(d / "words.app.json")
    (d / "words.json").write_text(json.dumps(j, indent=1))
    print(d.name, len(j["words"]), f"words (offset {offset:+.3f}s):", j["text"][:60])

for d in sorted(p for p in root.iterdir() if p.is_dir()):
    if needs_label(d):
        try:
            label(d)
        except Exception as e:  # one bad session must not stop the rest
            print(d.name, "SKIPPED:", e)
