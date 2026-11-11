"""Write words.json for every session that has audio.caf but no labels yet (same output as the app)."""
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

def label(d):
    meta = d / "meta.json"
    offset = (json.loads(meta.read_text()).get("audioStartOffsetSeconds") or 0) if meta.exists() else end_aligned_offset(d)
    out = subprocess.run([phonon, "transcribe", "--json", d / "audio.caf"], capture_output=True, text=True)
    if out.returncode or not out.stdout.strip():  # CAF left unfinalized by a crash: libsndfile rejects it, CoreAudio reads it
        with tempfile.TemporaryDirectory() as tmp:
            wav = pathlib.Path(tmp) / "audio.wav"
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", d / "audio.caf", wav], check=True)
            out = subprocess.run([phonon, "transcribe", "--json", wav], capture_output=True, text=True)
    if out.returncode or not out.stdout.strip():
        raise RuntimeError("phonon failed: " + out.stderr.strip()[-200:])
    j = json.loads(out.stdout)
    if not meta.exists():
        j["offsetEstimated"] = "end-aligned (no meta.json)"
    for w in j["words"]:
        w["start"] += offset
        w["end"] += offset
    j["clock"] = "start/end are on the motion.csv t clock"
    (d / "words.json").write_text(json.dumps(j, indent=1))
    print(d.name, len(j["words"]), f"words (offset {offset:+.3f}s):", j["text"][:60])

for d in sorted(p for p in root.iterdir() if p.is_dir()):
    if (d / "audio.caf").exists() and not (d / "words.json").exists():
        try:
            label(d)
        except Exception as e:  # one bad session must not stop the rest
            print(d.name, "SKIPPED:", e)
