# RL Curriculum Learning Strategy for Two-Wheeled Balancing Robot

This document outlines the curriculum learning strategy for training a two-wheeled self-balancing robot using Proximal Policy Optimization (PPO). 

To prevent catastrophic forgetting this strategy uses a **unified reward function** from timestep zero. The curriculum advances strictly by progressively widening the probability distribution of the commanded velocities ($v_{cmd}$ and $\omega_{cmd}$), rather than changing the reward formulas.

---

## 1. Unified Reward Function Formulation

The total reward $R_t$ at any timestep is a weighted sum of survival bonuses, tracking performance, and energy/smoothness penalties. We use squared exponential (Gaussian) kernels to bound the tracking rewards strictly between $[0, 1]$, which prevents gradient explosions and ensures stable PPO value estimation.

### 1.1 Survival & Posture
*   **Alive Bonus:** A constant reward provided as long as the robot has not fallen (e.g., $|\theta| < \theta_{max}$). This prevents the agent from terminating early to avoid accumulating penalties.
    $$r_{alive} = 1.0$$
*   **Pitch Rate Penalty:** Penalizes pitching motion (wobble), not the lean angle itself. The angle the robot balances at depends on where its centre of mass is and on the slope, and leaning is how it accelerates, so pulling the angle towards 0 fought both (v1 of this reward did, see `models/archive/v1_pitch_angle_reward/`).
    $$r_{\dot\theta} = -\dot{\theta}^2$$

### 1.2 Command Tracking
*   **Linear Velocity Tracking:** Replaces the need for a global $x, y$ position penalty. When $v_{cmd} = 0$, this explicitly acts as an anti-drift penalty.
    $$r_{v} = (1 - \beta)\exp\left(-\frac{(v_x - v_{cmd})^2}{\sigma_v}\right) + \beta\exp\left(-\frac{(v_x - v_{cmd})^2}{\sigma_{v,wide}}\right)$$
    The narrow kernel rewards precision near the command; the wide one still rewards getting closer when far from it. With the narrow kernel alone, v2 learned to crawl at 0.05 m/s for 0.5 m/s commands, because speeding up earned nothing until it was within ~0.15 m/s.
*   **Angular Velocity Tracking:** Encourages following the yaw rate command. Narrow kernel only: a wide one (tried in v4) made turning sloppy, and turning never had the crawl problem.
    $$r_{\omega} = \exp\left(-\frac{(\omega_z - \omega_{cmd})^2}{\sigma_\omega}\right)$$

### 1.3 Control Penalties
*   **Action Effort:** Penalizes high motor torque/PWM commands to save battery and reduce motor wear.
    $$r_{action} = -\sum_{i \in \{L, R\}} a_{i, t}^2$$
*   **Action Smoothness:** Heavily penalizes high-frequency oscillations (jitter) between consecutive motor commands.
    $$r_{smooth} = -\sum_{i \in \{L, R\}} (a_{i, t} - a_{i, t-1})^2$$

### 1.4 Total Reward Equation
$$R_t = w_1 r_{alive} + w_2 r_{\dot\theta} + w_3 r_{v} + w_4 r_{\omega} + w_5 r_{action} + w_6 r_{smooth}$$

---

## 2. Recommended Weights and Variances

These values serve as a robust starting point for an IsaacLab / Isaac Gym environment.

| Component | Weight ($w_i$) | Kernel Variance ($\sigma$) | Rationale |
| :--- | :--- | :--- | :--- |
| **Alive Bonus** | $w_1 = 2.0$ | N/A | Must be large enough to offset all potential penalties so the robot wants to survive. |
| **Pitch Rate** | $w_2 = 0.1$ | N/A | A 1 rad/s wobble costs 0.1 per step, a 3 rad/s transient 0.9: small next to the alive bonus, so it damps oscillation without discouraging the lean needed to track velocity. |
| **Linear Vel** | $w_3 = 2.0$ | $\sigma_v = 0.005$, $\sigma_{v,wide} = 0.25$, $\beta = 0.5$ | High priority (raised from 1.5, at which v5 drove at ~60% of the commanded speed). Reward kept: 0.02 m/s error 96%, 0.05 m/s 80%, 0.1 m/s 55%, 0.3 m/s 35%, 0.45 m/s 22%. The wide kernel keeps a gradient far from the command (narrow kernel alone: ~0% beyond 0.3 m/s and v2 crawled); the narrow one was 0.02 until v7, which kept 96% at 0.04 m/s error, so small commands and the last 10% of speed were not worth chasing. |
| **Angular Vel**| $w_4 = 1.0$ | $\sigma_\omega = 0.3$ | Slightly lower priority than standing upright and moving forward. Reward kept: 0.2 rad/s error 88%, 0.5 rad/s 43%, 1 rad/s 4%. |
| **Action** | $w_5 = 0.01$ | N/A | Kept small to prevent the robot from turning off motors and falling just to save energy. |
| **Smoothness** | $w_6 = 0.05$ | N/A | Critical for sim-to-real transfer to prevent motor destruction on the physical hardware. |

---

## 3. The Curriculum

The reward function remains identical across all stages. The curriculum only changes what the environment samples: first the command range, then the environment. It is an ordered list of stages in `src/self_balancing_robot/curriculum.py` (`CURRICULUM`). Each `Stage` holds everything that defines it: command range, reversal probability, slope, pushes, CoM shift, mass and friction ranges, sensor latency and noise, how magnitudes are sampled, and the test to leave it. To add, remove or retune a stage, edit that list; nothing else changes.

**Moving on.** The curriculum is global: all (thousands of) parallel GPU environments train at the same stage, and episodes that finish at it are pooled across environments in windows of 1,000. Each full window is checked against the stage's test; when it passes, new episodes start at the next stage of the list (the last stage is never left). Episodes still running at that moment finish at their old stage and are not counted. Up to v10 (SB3, 8 CPU environments) each environment advanced on its own over its last 20 episodes; with 4,096 environments each one finishes only a few dozen episodes per run, too few for that.

**A stage's test** (criteria set to `None` in a stage are not checked):
*   **Minimum episodes:** at least `min_episodes` episodes were played at the stage, so a lucky early window cannot pass.
*   **Survival:** the fraction of episodes reaching the timeout is at least `success_rate`.
*   **Speed ratio:** $\sum v_x \operatorname{sign}(v_{cmd}) / \sum |v_{cmd}|$ is at least `speed_ratio`, counting commands of at least 1/3 of the stage's maximum speed (and at least 0.05 m/s) held for at least 1 s (the acceleration after each new command is not counted). An absolute error threshold let v5 advance driving at ~60% of the command.
*   **Standing still:** on zero speed commands (held at least 1 s) the mean $|v_x|$ is at most `still_speed_limit`. v4 and v6 drifted at 0.07–0.10 m/s on slopes.
*   **Reversals:** at least `reversal_success_rate` of the episodes that had a reversal survive (about a third of the episodes have one).

**Sampling.** Command speed and yaw rate, slope, push and CoM shift are drawn per stage as 0 with probability 10% (stand still, flat ground, no push), in the band the stage added (from the previous stage's maximum up to its own) with 40%, and uniform below that band otherwise, so the policy practises the new difficulty without forgetting easier ones. A quantity the stage does not raise is 0 or uniform up to its maximum. Signs and directions are random.

**Reversals.** Where a stage sets `reversal_probability` (20% from the first command stage on), each command change during an episode is, with that probability, a full reversal instead: the opposite direction of the current command at 75–100% of the stage's speed limit, with no turning. With random commands only 0.3–0.6% of changes were full reversals, and v9 fell on most joystick flips (+0.3 → −0.3 m/s) because it braked so hard it pitched to ~30°, past the ~29° where the base touches the ground.

**Commands the wheels can deliver.** Every command is scaled down (speed and turn rate together) so that the outer wheel needs at most the full command speed of 0.4 m/s: $|v_{cmd}| + |\omega_{cmd}| \cdot b/2 \le 0.4$ m/s with half track $b/2 = 5.35$ cm (`goose.xml`; 6.45 cm on the earlier `XML-test.xml`). At 0.4 m/s and 1.5 rad/s the outer wheel would need ~0.5 m/s, more than the motors give while balancing; every v4 fall on flat ground came from such commands. The robot's joystick code must apply the same limit.

### The default stages

| # | Stage | What is new | Test to leave it |
|---|---|---|---|
| 0 | `balance` | Zero command, flat ground, nominal physics, perfect sensors | 95% survive |
| 1 | `commands 0.1 m/s` | $\|v_{cmd}\| \le 0.1$ m/s, $\|\omega_{cmd}\| \le 0.625$ rad/s, 20% reversals | 2,000 episodes, 95% survive, speed ratio 0.90, still within 0.02 m/s, 85% of reversals survive |
| 2 | `commands 0.2 m/s` | up to 0.2 m/s, 1.25 rad/s | as stage 1 |
| 3 | `commands 0.3 m/s` | up to 0.3 m/s, 1.875 rad/s | as stage 1 |
| 4 | `commands 0.4 m/s` | full range: 0.4 m/s, 2.5 rad/s | as stage 1 |
| 5 | `randomization 1/6` | mass ±10%, friction ±40%, 0–2 steps of motor/IMU/encoder latency, sensor noise; slope 0.05 rad, push 0.12 N, CoM shift 1.3 mm | 5,000 episodes, 90% survive, speed ratio 0.85, still within 0.05 m/s, 85% of reversals survive |
| 6–10 | `randomization 2/6` … `6/6` | slope, push and CoM shift up by 1/6 each, to 0.3 rad, 0.7 N, 8 mm | as stage 5 |

*   **Balance:** the robot learns to balance the inverted pendulum. Because $v_{cmd} = 0$, the $r_v$ tracking reward acts as a strict anti-drift penalty, keeping the robot hovering in its starting location.
*   **Commands:** the robot learns to start, brake and reverse, then aggressive acceleration and cornering, never more than one step beyond what it already masters. Commands are resampled every 3 to 5 seconds. Earlier versions had two fixed command steps (±0.05 m/s, then ±0.4 m/s): v5 and v6 passed the first by barely moving (~20% of the commanded speed) and the second had to unlearn it. A first try with 8 steps from 0.05 m/s never passed the first: at 0.05 m/s the speed signal is lost in balancing wobble and gearbox friction, and the policy drifted back to standing still (36% → 4% of the command from 3M to 5M steps). Up to v8 the range was 0.4 m/s (about 30% of the motor torque left for balancing; at 0.5 m/s only ~14%, and v4 fell more often); 0.3 m/s leaves headroom for the speed bias of a CoM shift: v8 with +1 cm drove ~0.15 m/s too fast, saturated the motors and fell. The network tilts as much as it needs to track the speed; only the rate of pitching is penalized. (v4, the best speed tracker so far, scored 0.83 at the full range.)
*   **Randomization:** this final part solidifies the policy; the network learns robust recovery maneuvers under velocity demands. The speed ratio is a bit lower than for the command stages (slopes cost speed uphill); v6, with 85% survival and no speed check, drifted back to ~70% of the commanded speed. At every episode start: chassis mass, floor friction (coefficient 0.48 to 1.12), CoM position (the nominal CoM is 4.9 cm above the axle; 8 mm needs ~9° of lean), terrain slope in a random direction (the base touches the ground at 0.5 rad of tilt relative to the surface, so steeper slopes cannot be driven straight up or down; v2, trained up to 0.5 rad, never survived slopes above 0.3 rad) and the motor, IMU and encoder latencies. IMU and encoder noise at every timestep. Random horizontal pushes on the chassis for 0.1 s every 2 to 4 seconds.

---

## 4. Episode Termination Conditions

An episode must immediately terminate, returning a reward of $0$ for the remainder of the theoretical timestep limit, if any of the following occur:

1.  **Pitch Failure:** $|\theta| > \theta_{max}$ or $|\phi_{roll}| > \theta_{max}$ radians ($\theta_{max} = 0.6$). The robot has fallen beyond the point of physical recovery.
2.  **Base Collision:** Any part of the chassis (other than the wheels) touches the ground.
3.  **Spinning:** $|\omega_z| > 3$ rad/s, twice the largest yaw command. On the GPU, PPO sometimes learned to balance while spinning in place at ~8 rad/s (a barely trained robot stays up longer that way), and the narrow yaw-rate kernel gives no gradient back from that far away.
4.  **Timeout:** The episode reaches the maximum length to prevent infinite loops and force buffer resets.