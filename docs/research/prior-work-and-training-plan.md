# Prior work and a training plan for low-data silent speech from AirPods motion

Date: 2026-11-11. Method: STORM-style synthesis (multiple expert perspectives asking questions, grounded in sources). Sources: the [supernaiter/silent-speech-review](https://github.com/supernaiter/silent-speech-review) corpus (152 structured reviews), a Consensus research thread ([consensus-thread.md](consensus-thread.md), [View this in Consensus](https://consensus.app/search/research-brief-engineering-context-we-are-building/9Td9xNipRISGbVbI1YZOKg/)), Consensus searches, and Meta's published sEMG work. The `knowledge-storm` package was not run (no working LLM API key in this environment); the outline and questions below follow its method by hand.

## TL;DR

1. **Yes, people have done almost exactly this.** Stony Brook's earable twin-IMU line (MuteIt 2022, Jawthenticate 2023, Unvoiced 2024, QuietSync with Microsoft 2024) decodes silently mouthed words from jaw motion picked up by an IMU at the ear. OpenEarable (2026) did 14 silent commands from in-ear IMU, 20 users, 92% F1. Nobody has published it on stock AirPods `CMHeadphoneMotionManager` data.
2. **What works there is small vocabulary, per-user calibration, and priors.** MuteIt: 100 command words, 20 users, 94.8%, using a jaw-kinematics + phoneme particle filter, not a learned classifier. Unvoiced: personalised, 20 audible samples per item, GPT-4 fills missing words. QuietSync: 12 commands, 94.2%, 5-shot fine-tune for new users.
3. **Open vocabulary from a 50 Hz ear IMU has no supporting evidence.** Even 8-channel face EMG, which is far richer, collapses to 78.6% CER on unseen sentences after multi-subject pretraining (NYU 2026). Our 76% CER is in line with that, not a bug.
4. **The single biggest lever is data from other people, then a short per-user calibration.** Meta's sEMG work: generic models need 100+ users to mean anything; personalising a 6,400-user model with 20 minutes still gave 16% median improvement. NYU EMG: pretraining on 26 other people took CER from 74.4% to 21.7%, and 3 minutes of calibration was as good as 13.
5. **Re-seating is the known killer, and it is fixable with multi-session training.** SilentWear: silent accuracy 77.5% within days, 59.3% on a held-out day; incremental fine-tuning recovers 10+ points with under 10 minutes. AESSI: training on 44 sessions from 24 people gave 92% on a later day with no calibration. EchoSpeech: re-donned glasses, 6-8 min fine-tune gives 9.5% WER on 31 commands.

## Perspectives and the questions they asked

**Earables researcher.** Is ear-IMU jaw motion enough signal? Evidence: yes for commands. MuteIt shows jaw motion alone is "not distinctive enough" for a naive classifier but works with jaw-anatomy priors and phoneme decomposition. EarSpy shows earphone accelerometers carry speech, but via bone-conduction vibration, which needs sample rates far above 50 Hz and voiced speech. Implication: Subvox only sees the slow jaw/head component, so it is a jaw-kinematics problem, not a vibration problem.

**Low-data ML engineer.** What architecture with ~150 takes? Evidence converges on tiny models: SilentWear's 15k-parameter CNN; SE-ResNet 1D for headphone EMG (96% on 10 words); Siamese / metric few-shot for continuous EMG+EEG. LipLearner gets few-shot custom commands from contrastive pretraining plus a linear head. DTW-1NN (what we have) is a legitimate baseline for 5-10 takes per word; a learned embedding should beat it once there are several sessions and more than one person.

**Foundation-model skeptic.** Will an IMU foundation model help? HAR self-supervised pretraining (Oxford UK Biobank 700k person-days, open weights; HUNT4; TransfHAR from CMU/Google) clearly helps few-label activity recognition, and TransfHAR reaches 86.7% on new wrist gestures with 5 examples. But those models are wrist/body accelerometers at 30-100 Hz on gross motion. A 2026 benchmark found general EEG foundation models gave no gain on speech decoding. Net: worth one cheap experiment (frozen encoder + linear probe), not a plan.

**Speech-transfer researcher.** Can audio supervise the motion? Gaddy & Klein transfer audio targets from vocalised to silent sessions by DTW alignment, which is the core trick that made silent EMG work. LipLearner's Voice2Lip registers silent commands from spoken ones. emg2speech aligns EMG to self-supervised speech units. Caveat: articulation differs between voiced and silent speech (ultrasound study 2025), so train on silent data with labels transferred from paired voiced takes. This is exactly issue #13.

**Big-lab perspective.** Meta (Nature 2025, open code `facebookresearch/generic-neuromotor-interface`, dataset `emg2qwerty`): cross-user variation is larger than cross-session; generic model alone was over 50% CER on emg2qwerty and fine-tuning brought it to 6.95%. Google (Project Relate style on-device ASR personalisation): 50 utterances per speaker cut WER 71% for disordered speech. Apple has published AirPods biosignal work (respiration and heart rate from in-ear audio, using speech foundation models), nothing on silent speech from the motion API. No DeepMind work found that applies.

**Product realist.** What can ship? Every working wearable system is 10-100 commands, user-calibrated, with an LM or task constraint on top. Unvoiced's 0.14 WER on unseen words leans heavily on GPT-4 over a small task space.

## Datasets and code worth using

| What | Why | Link |
|---|---|---|
| Gaddy & Klein silent EMG (open vocab, voiced + silent paired) | Reference implementation of audio-target transfer to silent sessions | github.com/dgaddy/silent_speech |
| Meta generic neuromotor interface + emg2qwerty | Training recipe and scaling/personalisation curves | github.com/facebookresearch/generic-neuromotor-interface |
| Oxford SSL wearables (UK Biobank pretrained ResNet) | Off-the-shelf accelerometer encoder for a linear-probe test | github.com/OxWearables/ssl-wearables |
| OpenEarable silent command dataset (14 cmds, 20 users, in-ear IMU) | Closest public-ish data to ours, if released | PerCom Workshops 2026 |
| silent-speech-review `data/papers.csv` | 152 papers with sensors, vocab, eval mode | this repo's sources |

Nothing public is recorded with AirPods `CMHeadphoneMotionManager`, so our own multi-person data is the dataset.

## Ranked plan for Subvox

1. **Change the target to a command vocabulary (10-30 words) for the product path.** Keep open vocabulary as a research track only. Probability this reaches over 85% on a fresh wear for one user: 55-70%. Most likely failure: 50 Hz rotation/accel cannot separate visemes like p/b/m, so confusable pairs must be avoided in the vocabulary.
2. **Collect across wears and people before changing the model.** 5-10 takes per word per wear, at least 3 wears per person, 5+ people. Evaluate leave-one-wear-out and leave-one-person-out (issues #8, #14). This alone is what moved every comparable system.
3. **Augment for re-seating.** Random small 3D rotation of the accel/rotation vectors (simulates a different seat), ±15% time warp, magnitude scaling, drift noise. Normalise per wear (z-score each channel) and rotate into a gravity-aligned frame using `gx,gy,gz`.
4. **Replace DTW-1NN with a tiny 1D CNN embedding (15k-50k params) trained with a prototypical / metric loss** across all people, then enrol a new user with 5 takes per word as prototypes. Keep DTW as the baseline it must beat on the leave-one-wear-out split.
5. **Pretraining experiment (cheap, optional):** frozen Oxford SSL encoder or a masked-reconstruction pretrain on all unlabeled AirPods motion we record, then linear probe. Keep only if it beats step 4 on held-out wears.
6. **Silent labels from voiced pairs (issue #13)** and a constrained decoder (CTC or word-sequence Viterbi with a small grammar or LLM rescoring) for short command phrases.
7. **Open-vocabulary track:** only after 5+ people x 10+ min of paired voiced/silent data. Expect over 60% CER. Treat as research.

## Honest caveats

- Several headline numbers above are author-reported with known issues flagged by the review repo (e.g. the six-axis IMU Conformer paper's 97% is word accuracy on 16 words and 2 sentences, with figure/text inconsistencies).
- No study isolates a stock consumer earbud's processed 50 Hz motion stream. Signal ceiling is the main unknown and only our own data can answer it.
