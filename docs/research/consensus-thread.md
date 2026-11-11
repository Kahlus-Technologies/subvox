# Consensus research thread: silent speech from earbud IMU with tiny data

Thread: [View this in Consensus](https://consensus.app/search/research-brief-engineering-context-we-are-building/9Td9xNipRISGbVbI1YZOKg/)

The full cited document (20 papers, with inline [N.M] quote markers and the References list) lives in that thread. Its conclusions, verbatim headings:

- Prior Wearable Inertial and Craniofacial Silent Speech Systems: EarSpy, QuietSync, skin-conformal IMU, EarCommand, eyewear SSI, soft magnetic skin. Re-seating degrades template alignment in every configuration.
- Few-Shot Adaptation, Biosignal Modeling, and Big-Lab Paradigms: cross-subject pretraining + short calibration; incremental fine-tuning recovers over 10% after re-attachment with under 10 min; LMs correct sequence hypotheses; silent articulation differs from voiced.
- IMU Foundation Models and Pretraining Strategies: oneHAR, MASTER, IMU2CLIP, Time2Lang.
- Concrete Recommendations for Subvox: (1) 1D CNN + SE blocks with prototypical/metric loss, 15k-50k params; (2) 10-30 command vocabulary; (3) 5-10 takes per word across 2+ seatings; (4) rotation jitter, ±15% time warp, drift noise; (5) masked pretraining on public HAR corpora.
- Evidence caveat: no evidence a 50 Hz earphone IMU alone supports open-vocabulary decoding.

Key references (DOIs):
- Srivastava et al. 2024, QuietSync, 10.1145/3678957.3685720
- Srivastava et al. 2022, MuteIt, 10.1145/3550281
- Cao et al. 2024, EarSpy, 10.1109/tmc.2023.3333214
- Jin et al. 2022, EarCommand, 10.1145/3534613
- Spacone et al. 2026, SilentWear, 10.1109/jsen.2026.3742328
- Inoue et al. 2026, around-ear EEG large-scale training, 10.1088/1741-2552/ae54d0
- Satterlee et al. 2025, sentence-level EMG/EEG with LM, 10.3390/s25196168
- Tang et al. 2025, textile EMG headphones, 10.1109/tim.2025.3583386
- Gosztolya et al. 2025, silent vs voiced articulation, 10.21437/ssw.2025-24
- Tomanek et al. 2021, on-device ASR personalisation, 10.48550/arxiv.2106.10259
- Bian et al. 2026, HAR foundation model survey, 10.1145/3810230
- Zhu et al. 2025, MASTER, 10.1145/3749511
- Yuan et al. 2024, SSL on 700k person-days, 10.1038/s41746-024-01062-3
- Kaifosh et al. 2025, Meta generic neuromotor interface, 10.1038/s41586-025-09255-w

_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders' consent._
