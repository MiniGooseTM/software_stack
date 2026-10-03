# v10: v9 + full-reversal commands (archived 2026-10-03) - CURRENT

Same as v9 (0.3 m/s / 1.5 rad/s, 3 command levels, CoM up to 8 mm, actor 128-64-32, commands
divided by (0.3, 1.5)) plus: 20% of mid-episode command changes are full reversals
(opposite direction at 75-100% of the speed limit, no turning), and levelling up also needs
85% survival over the episodes that had a reversal. v9 fell on most joystick flips
(+0.3 -> -0.3 m/s): it braked so hard it pitched to ~30 deg. Trained from scratch, 30M steps.

Curriculum: stage 3 by ~4M, end levels [5, 5, 5, 5, 5, 5, 5, 6, 5, 5] (v9: 4-6).

Full reversals (+-0.3 m/s flipped every 0.7 / 1 / 1.5 / 3 s, stage-3 noise, latency,
mass/friction, 40 runs each, no pushes):

| | falls (of 160) | max pitch |
|---|---|---|
| v7 @30M | 0 | 16-18 deg |
| v9 @29M (20 runs each) | 41 of 80 (v9 @30M fell 5/5 even without noise) | 20-27 deg |
| v10 @27M / 28M / 29M | 20 / 28 / 17 | 16-21 deg |
| **v10 @30M** | **2** | 15-17 deg |

Other tests (v9 env; flat 100 episodes, slopes 240 at level 6, new seeds):

| | held +0.1 / +0.2 / +0.3 / -0.3 m/s (8 seeds) | flat surv | slopes surv (0-0.1 / 0.1-0.2 / 0.2-0.3) | still drift 0 / 0.2 rad | CoM falls /120 |
|---|---|---|---|---|---|
| v7 @30M | 0.11 / 0.20 / 0.29 / -0.24 | 100% | 67% (89 / 80 / 31%) | 2.6 / 8.1 cm/s | 10 |
| v9 @30M | 0.10 / 0.21 / 0.31 / -0.28 | 98% | 60% (88 / 69 / 25%) | 0.5 / 3.8 cm/s | 16 |
| v10 @29M | 0.08 / 0.18 / 0.28 / -0.29 | 97% | 59% (85 / 65 / 26%) | 0.5 / 4.2 cm/s | 10 |
| **v10 @30M** | 0.07 / 0.16 / 0.24 / -0.29 | **100%** | **64% (91 / 78 / 22%)** | **0.4 / 4.2 cm/s** | **6** |

Weakness: forward speed undershoots by ~20% (+0.3 -> +0.24, backward -0.3 -> -0.29), the same
with and without noise. Checkpoints that track forward speed better fall more on reversals.
