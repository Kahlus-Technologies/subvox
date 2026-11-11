"""Write words.json for every session that has audio.caf but no labels yet (same output as the app)."""
import json, pathlib, subprocess, sys

root = pathlib.Path.home() / "Documents/Subvox/Sessions"
phonon = pathlib.Path(sys.executable).parent / "phonon"
for d in sorted(root.iterdir()):
    if not (d / "audio.caf").exists() or (d / "words.json").exists():
        continue
    offset = json.loads((d / "meta.json").read_text()).get("audioStartOffsetSeconds") or 0
    out = subprocess.run([phonon, "transcribe", "--json", d / "audio.caf"], capture_output=True, text=True)
    j = json.loads(out.stdout)
    for w in j["words"]:
        w["start"] += offset
        w["end"] += offset
    j["clock"] = "start/end are on the motion.csv t clock"
    (d / "words.json").write_text(json.dumps(j, indent=1))
    print(d.name, len(j["words"]), "words:", j["text"][:60])
