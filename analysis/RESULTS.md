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

## Appended runs

`reseat.py --results analysis/RESULTS.md` appends its output below. Each heading is the exact command. A command with `--dataset analysis/fixtures/...` ran on synthetic data. A command without `--dataset` ran on the app's real dataset.

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
