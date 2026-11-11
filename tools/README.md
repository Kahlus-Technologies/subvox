# Talk-mode labels (local Phonon-2)

Subvox runs `tools/.venv/bin/phonon` (Phonon-2, the model inside Detta: NVIDIA Parakeet, 164 MB,
runs locally) after every session recording that has audio, and writes `words.json` next to
`motion.csv`. Word start/end times are on the motion.csv `t` clock.

Setup (once):

    cd tools && uv venv --python 3.12 .venv && source .venv/bin/activate
    uv pip install fermion-research mlx mlx-audio mlx-lm soundfile scipy zstandard

Label sessions recorded before this existed: `tools/.venv/bin/python tools/label_sessions.py`
