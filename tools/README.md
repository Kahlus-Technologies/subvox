# Talk-mode labels (local Phonon-2)

Subvox runs `tools/.venv/bin/phonon` (Phonon-2, the model inside Detta: NVIDIA Parakeet, 164 MB,
runs locally) after every session recording that has audio, and writes `words.json` next to
`motion.csv`. Word start/end times are on the motion.csv `t` clock. Silent-mouthing sessions are not
transcribed: their audio has no speech, so Phonon could only invent words.

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

## Word dataset for training

`python3 tools/build_dataset.py ~/Documents/Subvox/Sessions path/to/unzipped/export/Sessions --out subvox-dataset`
turns every session with `motion.csv` and `words.json` into `subvox-dataset/words.npz` (one row per spoken
word with its motion slice) and `subvox-dataset/summary.txt` (sessions, speakers, minutes, words, and
sessions skipped and why). Needs numpy only. Test: `python3 -m unittest discover -s tools -p 'test_*.py'`.

Silent takes never use their own `words.json`. With pairing on (the default), Collect records each sentence
spoken, then mouthed silently, as two sessions whose `meta.json` share a `pairId`. The silent take gets the
spoken take's words: `build_dataset.py` DTW-aligns the two motion streams (accel and rotation-rate channels,
each z-scored within its take) and moves every word start/end through the warping path onto the silent
take's clock. Those rows have `labels_from` set to the spoken session's folder name; rows labelled from their
own audio have `labels_from` "". Silent takes with no spoken partner are skipped, and `summary.txt` counts
the transferred words. On synthetic takes (silent = time-warped, weaker, noisy copy of spoken) the mapped
boundaries land within 1 sample (20 ms) of the truth. On real AirPods takes this still needs checking.

## Check audio-to-motion alignment

`python3 tools/check_alignment.py ~/Documents/Subvox/Sessions` cross-correlates each session's audio
loudness with head-motion energy and prints the lag between them. Near 0 means `words.json` labels land on
the right motion; sessions off by more than 100 ms are flagged (exit code 1). Needs numpy and macOS
`afconvert`. Test: `python3 -m unittest discover -s tools -p 'test_*.py'`.
