# Talk-mode labels (local Phonon-2)

Subvox runs `tools/.venv/bin/phonon` (Phonon-2, the model inside Detta: NVIDIA Parakeet, 164 MB,
runs locally) after every session recording that has audio, and writes `words.json` next to
`motion.csv`. Word start/end times are on the motion.csv `t` clock.

Setup (once):

    cd tools && uv venv --python 3.12 .venv && source .venv/bin/activate
    uv pip install fermion-research mlx mlx-audio mlx-lm soundfile scipy zstandard

Label sessions recorded before this existed: `tools/.venv/bin/python tools/label_sessions.py`

## Phone timings (Montreal Forced Aligner)

`tools/align_phones.py` writes `phones.json` (each ARPAbet phoneme's start/end, same clock as motion.csv)
for every labelled session. Setup and usage are in the file's docstring. Unknown words get a G2P-guessed
pronunciation so nothing collapses into a phone-less `spn`. Checked on 2 sessions: 0 unknown words left,
39 of the 39 ARPAbet phonemes seen, vowels land on audio energy peaks and stops in dips, word starts
within 35-50 ms (median) of Phonon's.
