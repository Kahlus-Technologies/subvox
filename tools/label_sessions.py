"""Write words.json for every session that has audio.caf but no labels yet (same output as the app)."""
import json, pathlib, subprocess, sys

root = pathlib.Path.home() / "Documents/Subvox/Sessions"
phonon = pathlib.Path(sys.executable).parent / "phonon"
for d in sorted(root.iterdir()):
    if not (d / "audio.caf").exists() or (d / "words.json").exists():
        continue
    meta = d / "meta.json"  # missing after a crash (#6): assume audio started with motion
    offset = (json.loads(meta.read_text()).get("audioStartOffsetSeconds") or 0) if meta.exists() else 0
    out = subprocess.run([phonon, "transcribe", "--json", d / "audio.caf"], capture_output=True, text=True)
    if out.returncode or not out.stdout.strip():  # CAF left unfinalized by a crash: libsndfile rejects it, CoreAudio reads it
        wav = d / "audio.recovered.wav"
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", d / "audio.caf", wav], check=True)
        out = subprocess.run([phonon, "transcribe", "--json", wav], capture_output=True, text=True)
    j = json.loads(out.stdout)
    if not meta.exists():
        j["offsetAssumed"] = True
    for w in j["words"]:
        w["start"] += offset
        w["end"] += offset
    j["clock"] = "start/end are on the motion.csv t clock"
    (d / "words.json").write_text(json.dumps(j, indent=1))
    print(d.name, len(j["words"]), "words:", j["text"][:60])
