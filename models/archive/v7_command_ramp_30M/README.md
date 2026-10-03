# v7: 3-stage curriculum with command ramp, 30M steps (archived 2026-10-03)

Superseded by v10 (`../v10_reversals_30M`). v7 was trained with commands divided by
(0.4, 1.5) and a 64-32 actor: with the current code (commands divided by (0.3, 1.5)) it sees
every command 33% too large, so run it with its `config_snapshot.py` settings.

Trained from scratch for 30M steps (10 envs). New 3-stage curriculum (stabilization, command
ramp in 4 levels up to 0.4 m/s / 1.5 rad/s, then 6 slope+push levels), level-up needs survival,
speed ratio (stage 2: 95% / 0.90 / still <= 0.02 m/s; stage 3: 90% / 0.85 / still <= 0.05 m/s),
commands scaled by the full range in the observation, wheel-feasible commands, narrow speed
kernel 0.005. Actor 88 -> 64 -> 32 -> 2. Config: `config_snapshot.py`.

At 30M all envs were in stage 3, levels [6, 4, 3, 4, 4, 6, 5, 4, 5, 5].

Results (flat: 100 episodes, full command range; slopes: stage 3 at level 6, 240 episodes with
slopes spread over 0-0.3 rad; still drift: v = 0 for 500 steps with noise/latency/CoM/mass):

| | held 0.15 / 0.3 / 0.4 / -0.3 m/s | flat surv / v err | slopes surv (0-0.1 / 0.1-0.2 / 0.2-0.3) | still drift 0 / 0.3 rad |
|---|---|---|---|---|
| v7 @29M | 0.17 / 0.31 / 0.39 / -0.28 | 84% / 0.065 | 60% (79 / 70 / 31%) | 4.2 / 4.7 cm/s |
| **v7 @30M** | 0.16 / 0.29 / 0.35 / -0.24 | **99% / 0.063** | **68% (91 / 81 / 32%)** | 3.6 / 4.5 cm/s |

Best survival of all checkpoints from 25M to 45M (resuming to 45M lowered survival).
Weaknesses: 0.4 m/s commands reach ~0.35 m/s, backward commands ~80%, slopes above 0.2 rad
survive only ~1/3 of the time, and standing still drifts at ~4 cm/s (no position feedback).
