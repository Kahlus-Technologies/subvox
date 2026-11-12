# Analysis results

Each section says whether its numbers are real or synthetic. Real numbers come from the owner's recordings, which are not in the repo. Synthetic numbers come from the fixtures in `analysis/fixtures/` (regenerate with `python3 analysis/fixtures/make_fixture.py`). They check that a script runs and that a fix undoes a problem injected on purpose. They say nothing about accuracy on real takes.

## Re-seat drift, real data (#14, #17): the owner's runs

Copied from the owner's comments on #14 and #17. These came from `analysis/reseat_test.py` (5f5b949), not `reseat.py`: DTW-1NN on all 9 motion channels, trained on wear 0 and tested on wear 1 only. Wears were split by createdAt gaps over 3 min: wear 0 has 201 takes, wear 1 has 29 (AirPods re-seated), wear 2 has 4 (ignored).

Same 172 training takes (wear 0 minus its last 29), two test sets of 29:

| Test set (n=29) | Accuracy |
|---|---|
| Same wear (last 29 takes of wear 0) | 58.6% |
| Re-seated (wear 1) | 55.2% |
| Difference | -3.4 points, 95% bootstrap CI -28 to +21 |

Fixes, trained on all 201 takes of wear 0 and tested on wear 1. Compare these rows with each other, not with the table above, which used fewer training takes.

| Fix | Accuracy on wear 1 (n=29) |
|---|---|
| None (global z, like the app) | 69.0% (20/29) |
| Per-wear z | 72.4% |
| Gravity frame | 72.4% |
| Per-wear z + gravity frame | 72.4% |
| Augmentation (rotation, time warp, gain) | 65.5% |
| All three | 65.5% |
| Chance | 20% (P(>=20/29) = 1.6e-8) |

The owner's augmentation in `reseat_test.py` is not the one in `augment.py`: 2 copies per training take, a rotation about each axis drawn with an 8 degree standard deviation, time warp and gain of 0.85-1.15, no drift.

Caveats from the owner: n=29 and a single re-seat. Each fix moves the score by one take, which is noise at this size, and the CI on the re-seat drop (-28 to +21 points) cannot rule out a large drop. With a 120 s gap threshold, wear 0 splits 147/54, so it may not be one seating. Answering the question needs 3-4 deliberate re-seats of about 25 takes each (#16).

## Open vocabulary (#11)

Can a model spell words from AirPods head motion when it never trained on those words, or can it only match words it has seen? `analysis/open_vocab.py` answers that on talk-mode sessions.

**No real-data result yet.** The plan is to run this after 5+ people x 10+ min of talk-mode data (docs/research/prior-work-and-training-plan.md, step 7), and the research notes expect over 60% CER even then. The only numbers below come from synthetic sessions. They show that the pipeline runs and finds a signal that was planted on purpose. They say nothing about AirPods.

Two baselines, and every word is scored by a model that never saw its session (or its speaker):

- DTW-1NN gives a test word the word of its nearest training word. It can only name words that occur in training.
- CTC is a small 1D-CNN + GRU (about 80k parameters) that spells a word's ARPAbet phonemes from its motion, so it can spell words it never trained on. For word accuracy, its spelling is snapped to the closest pronunciation among all the dataset's words, test words included. That makes CTC word accuracy a closed-set score over a word list the model is handed, like zero-shot classification with known class names. The CER columns are the open-vocabulary measure.

Motion is z-scored per speaker over all of that speaker's words, test words included but without labels, so a new speaker is normalised from their own motion. Silent takes from paired Collect are left out. Chance is the same predictions scored against the test targets shuffled within each fold, 200 times, as in `score.py`. CER counts phoneme edits per target phoneme, so it can pass 100% when guesses are longer than the targets. "Unseen words" are test words that never occur in training, and their columns are the open-vocabulary question itself. "Held-out sessions" cuts each speaker's sessions into 5 blocks in time order. Collect starts every new speaker at the same first sentence and goes through the list in order, so a block's sentences are missing from training for everyone, and that is where unseen words come from. For the same reason, held-out speakers mostly tests words that other speakers said.

### Run it on real exports

    pip install -r analysis/requirements.txt -r analysis/requirements-ml.txt
    python3 tools/align_phones.py   # phones.json for ~/Documents/Subvox/Sessions; MFA setup is in the file
    SUBVOX_SESSIONS=path/to/unzipped/export/Sessions python3 tools/align_phones.py
    python3 analysis/open_vocab.py --sessions ~/Documents/Subvox/Sessions path/to/unzipped/export/Sessions \
        --results analysis/RESULTS.md

Sessions without `phones.json` need `--lexicon FILE`, a CMUdict-format pronunciation file you supply (nothing is downloaded or bundled). A `words.npz` from `tools/build_dataset.py` works too (`--npz subvox-dataset/words.npz --lexicon FILE`), but then phonemes come only from the lexicon and word times from `words.json`. A real-data run writes its table under "Real data" below.

### Real data

<!-- open_vocab.py real -->
Pending: no real talk-mode exports have been run yet.
<!-- /open_vocab.py real -->

### Synthetic fixture (pipeline check only)

`analysis/fixtures/make_sessions.py` writes 3 speakers x 30 one-sentence sessions in the app's file formats, drawing words from an 86-word lexicon (70 of them come up with the default seed). Each phoneme is a fixed motion pattern (look-alike phonemes such as P/B/M share most of it), smoothed into its neighbours, mixed per speaker and per session, plus noise and head drift. Both models should beat chance here; if they didn't, the pipeline would be broken. Regenerate with:

    python3 analysis/fixtures/make_sessions.py /tmp/subvox-synth   # needs a new or empty folder
    python3 analysis/open_vocab.py --sessions /tmp/subvox-synth --results analysis/RESULTS.md

<!-- open_vocab.py synthetic -->
**Synthetic fixture, not recordings.** Run 2026-11-11: 3 speakers, 90 sessions, 579 words (70 unique); phonemes from phones.json for 579 words, from the lexicon for 0. Took 40 s.

    python3 analysis/open_vocab.py --sessions /tmp/subvox-synth --results analysis/RESULTS.md

| split | model | words | seen in training | word accuracy (chance) p | phoneme CER (chance) p | unseen words: accuracy | unseen words: CER |
|---|---|---|---|---|---|---|---|
| held-out sessions (5 time blocks per speaker) | DTW-1NN | 579 | 78% | 63% (5%) p=0.005 | 35% (105%) p=0.005 | 0% (0%) p=1.000 | 90% (105%) p=0.005 |
| held-out sessions (5 time blocks per speaker) | CTC | 579 | 78% | 53% (5%) p=0.005 | 39% (98%) p=0.005 | 5% (1%) p=0.010 | 71% (94%) p=0.005 |
| held-out speakers | DTW-1NN | 579 | 100% | 54% (3%) p=0.005 | 47% (107%) p=0.005 | - | - |
| held-out speakers | CTC | 579 | 100% | 34% (4%) p=0.005 | 58% (98%) p=0.005 | - | - |
<!-- /open_vocab.py synthetic -->

## Appended runs

`reseat.py` and `proto_net.py` append their output below when given `--results analysis/RESULTS.md`. Each heading is the exact command. A command with `--dataset analysis/fixtures/...` ran on synthetic data. A command without `--dataset` ran on the app's real dataset.

Synthetic fixture `dataset_reseat.json`: 3 wears of 12 takes (3 words x 4 takes), 10 minutes apart. Wear 0 is upright, and wears 1 and 2 tilt the AirPods by +25 and -25 degrees about the head's left-right axis, applied to accel, rotation rate and gravity together. The words differ mostly in the direction of the motion, so the tilt confuses them by construction. 25 degrees was chosen so the drop is easy to see, not taken from a measured re-seat.

With `--augment`, a second table gives leave-one-wear-out accuracy for each augmentation step from `augment.py` (#17) under each fix. On this fixture the gravity frame alone undoes the tilt, so the augmentations have nothing left to fix. Without it, a rotation of up to 10 degrees cannot cover a 25 degree tilt, and all four steps together lower the baseline. The fixture cannot say whether augmentation helps on real re-seats.

### `python3 analysis/reseat.py --dataset analysis/fixtures/dataset_reseat.json --augment --results analysis/RESULTS.md`

36 takes, words ['no', 'stop', 'yes'], 3 wears of [12, 12, 12] takes (gap > 180 s or wear field). DTW-1NN on accel + rotation rate.

| fix | leave-one-wear-out | by held-out wear | within-wear leave-one-out | chance (shuffled labels) | p vs chance |
|---|---|---|---|---|---|
| none (global z, like the app) | 50% (18/36) | 8/12, 5/12, 5/12 | 100% (36/36) | 31% | 0.029 |
| per-wear z | 67% (24/36) | 10/12, 6/12, 8/12 | 100% (36/36) | 31% | 0.001 |
| gravity frame | 100% (36/36) | 12/12, 12/12, 12/12 | 100% (36/36) | 31% | 0.001 |
| per-wear z + gravity frame | 100% (36/36) | 12/12, 12/12, 12/12 | 100% (36/36) | 31% | 0.001 |

Leave-one-wear-out with 4 augmented copies of each training take (augment.py, seed 0). Columns are the fixes above.

| augmentation | none (global z, like the app) | per-wear z | gravity frame | per-wear z + gravity frame |
|---|---|---|---|---|
| none | 50% (18/36) | 67% (24/36) | 100% (36/36) | 100% (36/36) |
| rotation ±10° | 50% (18/36) | 64% (23/36) | 100% (36/36) | 100% (36/36) |
| time warp ±15% | 56% (20/36) | 72% (26/36) | 100% (36/36) | 100% (36/36) |
| magnitude 0.8-1.2 | 58% (21/36) | 69% (25/36) | 100% (36/36) | 100% (36/36) |
| slow drift | 50% (18/36) | 72% (26/36) | 100% (36/36) | 100% (36/36) |
| all four | 33% (12/36) | 67% (24/36) | 100% (36/36) | 100% (36/36) |

Prototypical CNN vs DTW-1NN (#18). No real multi-wear or multi-person data exists yet, so `proto_net.py` has only run on synthetic fixtures. It takes one `--dataset` file per person. The three fixture people share the words of `dataset_reseat.json`, which is person A. B and C (`dataset_person_b.json`, `dataset_person_c.json`) turn each word's axis by 4-6 degrees, mouth 10% faster or slower and 20-25% smaller or bigger, and each of their seats adds a turn of up to 25 degrees about the gravity axis to a tilt. The gravity frame cannot undo that turn. Each person has 3 wears of 12 takes (3 words x 4 takes).

### `python3 analysis/proto_net.py --dataset analysis/fixtures/dataset_reseat.json --dataset analysis/fixtures/dataset_person_b.json --dataset analysis/fixtures/dataset_person_c.json --results analysis/RESULTS.md`

108 takes, words ['no', 'stop', 'yes']. People: A = analysis/fixtures/dataset_reseat.json (36 takes, 3 wears); B = analysis/fixtures/dataset_person_b.json (36 takes, 3 wears); C = analysis/fixtures/dataset_person_c.json (36 takes, 3 wears). Features: 9 channels, gravity frame + per-wear z. Proto-net: 36,228 parameters (1D CNN, 2 residual SE blocks, 32-number embedding), 300 episodes per fold of up to 10 words x (5 support + 3 query) takes, augmentation: rotation, time warp, magnitude, drift, seed 0.

| split | method | accuracy | by held-out wear or person | rejected (margin < 0.05) | accuracy on the rest | chance (shuffled labels) | p vs chance |
|---|---|---|---|---|---|---|---|
| leave-one-wear-out | DTW-1NN, 5 takes per word | 96% (104/108) | A0 12/12, A1 12/12, A2 12/12, B0 8/12, B1 12/12, B2 12/12, C0 12/12, C1 12/12, C2 12/12 | 14% | 100% (93/93) | 33% | 0.001 |
| leave-one-wear-out | proto-net, 5-take prototypes | 100% (108/108) | A0 12/12, A1 12/12, A2 12/12, B0 12/12, B1 12/12, B2 12/12, C0 12/12, C1 12/12, C2 12/12 | 0% | 100% (108/108) | 33% | 0.001 |
| leave-one-person-out | DTW-1NN, 5 takes per word | 98% (62/63) | A 21/21, B 20/21, C 21/21 | 3% | 100% (61/61) | 33% | 0.001 |
| leave-one-person-out | proto-net, 5-take prototypes | 100% (63/63) | A 21/21, B 21/21, C 21/21 | 0% | 100% (63/63) | 33% | 0.001 |

On this fixture the proto-net beats DTW-1NN by 4 of 108 takes leave-one-wear-out (100% vs 96%) and by 1 of 63 leave-one-person-out (100% vs 98%). DTW's leave-one-wear-out misses are all in B's first wear. Both are near the ceiling because the fixture's three words differ mainly in the direction of the motion. The run shows that training, enrolment and both splits work end to end. It does not show that the net is better on real takes. The reject rates are not comparable between the two: the margin is a ratio of distances, and the net is trained to push prototypes apart. Per-wear z and the gravity frame use every take of a wear, held-out ones included, but no labels, as in reseat.py. Chance shuffles the labels with each fold's trained net fixed, so its p-value covers enrolment and scoring, not training; retraining for each of 1,000 shuffles would take hours. With `--no-augment` the net scored 106/108 and 63/63, two takes fewer leave-one-wear-out, which is too few for the fixture to say whether augmentation helps it. Whether to keep the net is still open. Real data decides: rerun with one dataset.json per person once multi-wear recordings (#16) exist, and keep the net only if it beats DTW there.
