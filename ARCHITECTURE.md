# Lightweight RL Controller Architectures for 2-Wheeled Self-Balancing Robot

This document outlines two training pipelines for a highly efficient, proprioceptive-only neural network policy designed for real-time deployment on an ESP32 microcontroller. Both methods rely on leveraging privileged simulation data to train a lightweight, history-based feedforward network (MLP).

## 1. Common State and Action Spaces

### 1.1 The Action Space ($A_t$)
*   **Dimensions:** 2
*   **Vector:** $[pwm_{left}, pwm_{right}]$ normalized to $[-1, 1]$.
*   **Meaning:** Each output is a signed motor command. `-1` means full reverse,
    `0` means zero command, and `+1` means full forward. The policy output is
    therefore not a PWM timer count or a voltage in volts; it is a dimensionless
    normalized command that is converted to hardware-specific PWM duty cycle at
    deployment time.

### 1.2 The Privileged State ($S_t$) - Teacher/Critic Only
Used exclusively during simulation training. It is fully Markovian (no history window) because it contains exact physical truths.
*   **Proprioceptive Core:** the same 11 states as $O_t$ (see 1.3), but noise-free and without sensor latency, plus the true forward and lateral chassis velocity.
*   **Privileged Variables:** Exact Center of Mass (CoM) offset $(x, y, z)$, chassis mass deviation, floor friction deviation, current external push force $(F_{forward}, F_{lateral})$ in the robot's heading frame, motor/IMU/encoder latency, and terrain slope $(forward, lateral)$. All are normalized to roughly $[-1, 1]$ by their randomization range.
*   **Size:** 25 (`PRIVILEGED_FIELDS` in `envs/balance_env.py`).

### 1.3 The Proprioceptive History ($O_t$) - Student/Actor Only
Used by the ESP32 to infer the privileged variables implicitly. 
*   **Base Observation:** 11 states, in this order (`ACTOR_FRAME_FIELDS` in `envs/balance_env.py`):
    $[v_{x,cmd}, \omega_{z,cmd}, roll, pitch, \dot{roll}, \dot{pitch}, \dot{yaw}, \omega_{L}, \omega_{R}, a_{L,t-1}, a_{R,t-1}]$.
    Commands divided by the full command range ($v_{x,cmd} / 0.4$ m/s, $\omega_{z,cmd} / 2.5$ rad/s, so in $[-1, 1]$; the joystick command must first be limited so the outer wheel needs at most 0.4 m/s, as in training), angles in rad (IMU attitude estimate), rates in rad/s (gyro, body frame), wheel velocities divided by the motor no-load speed (19.4 rad/s), and $a_{t-1}$ is the policy's own previous output.
*   **Temporal Window:** $H = 8$ frames (`EnvironmentConfig.history_length`, keep within 5 to 10).
*   **Input Vector:** Concatenated history $[O_t, O_{t-1}, ..., O_{t-H+1}]$, newest first, resulting in an input size of $11 \times H = 88$. At startup the window is filled with copies of the first frame.

---

## 2. Option A: Expert Distillation via DAgger (Imitation Learning)

This pipeline trains a heavy Expert to solve the task, then uses it to label trajectories for a lightweight Student policy to mimic.

### 2.1 Architectures
*   **The Teacher (Expert Policy):** `Linear(Priv_Size) -> ReLU -> Linear(128) -> ReLU -> Linear(64) -> ReLU -> Output(2)`
    *   Takes the privileged state $S_t$ to maximize task rewards.
*   **The Student (ESP32 Policy):** `Linear(11 * H) -> ReLU -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Output(2)`
    *   Takes the history window $O_t$ to minimize action deviation from the Teacher.

### 2.2 Training Pipeline
1.  **Phase 1 - Expert RL Training:** Train the Teacher via PPO in a randomized simulation environment (varying mass, friction, and applying random external pushes). Freeze weights upon convergence.
2.  **Phase 2 - DAgger Loop:** 
    *   Let the Student control the simulated robot.
    *   As the Student drives, pass the simulation's privileged state $S_t$ to the frozen Teacher in the background to calculate the *ideal* action $A_{teacher}$.
    *   Record $(O_{student}, A_{teacher})$ into a rolling dataset buffer.
    *   Retrain the Student using Mean Squared Error (MSE) loss against the Teacher's labels.
    *   Repeat until the Student's survival rate matches the Teacher's.

---

## 3. Option B: Direct RL via Asymmetric Actor-Critic (Recommended)

This pipeline trains the lightweight ESP32 policy directly using Reinforcement Learning. It bypasses Behavioral Cloning entirely, eliminating covariance shift because the Actor learns from its own exploration.

### 3.1 Architectures
*   **The Critic (Value Function):** `Linear(Priv_Size) -> ReLU -> Linear(128) -> ReLU -> Linear(64) -> ReLU -> Output(1)`
    *   Evaluates the "goodness" of a state. It receives the privileged state $S_t$ to accurately map the value of the robot's current physical situation (e.g., knowing exactly how strong an external push is).
*   **The Actor (ESP32 Policy):** `Linear(11 * H) -> ReLU -> Linear(128) -> ReLU -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Output(2)`
    *   Models up to v7 used `Linear(64) -> Linear(32)` (7.8k weights); the current actor has 21.8k (87 KB as float32, ~22k multiply-adds per step).
    *   Takes the proprioceptive history $O_t$ and outputs actions. It is completely blind to the exact privileged data but receives policy gradients guided by the Critic's accurate value estimations.

### 3.2 Training Pipeline
1.  **Unified PPO Loop:** The Actor and Critic are trained simultaneously in simulation.
2.  **Domain Randomization & Noise:** Heavily randomize mass, friction, and sensor noise (IMU latency and encoder latency). Continuously apply random physical impulses to the chassis.
3.  **Optimization:** 
    *   The Critic rapidly learns an accurate value function because it can "see" the exact friction and disturbances.
    *   The Actor is forced to learn how to infer those disturbances from its history window $O_t$ to satisfy the Critic's gradients. 
    *   Because the Actor drives the rollout itself, it naturally experiences edge cases and learns its own optimal recovery maneuvers.
4.  **Extraction:** Discard the Critic. Extract the Actor for deployment.

### 3.3 Training Stack (GPU)
*   **Simulation:** MuJoCo MJX (JAX) runs thousands of copies of the robot in parallel on one GPU (`envs/balance_env.py`). The robot model is `assets/goose.xml`, placed on a tiltable ground plane by `assets/env_v0.xml`; the MJX copy drops the visual meshes and caps the constraint solver at 4 iterations (trajectories match MuJoCo's default of 100 to ~1e-6; 2.6x faster in a batch). The same environment runs on the CPU for evaluation, teleop, export and tests.
*   **PPO:** Brax PPO (`train.py`), 4,096 environments, 32-step rollouts, 16 minibatches, 4 epochs, learning rate 3e-4, $\gamma = 0.99$, $\lambda = 0.95$, clip 0.2, entropy 0.01. On an RTX 4050 laptop GPU it runs ~165k environment steps per second in training (the earlier SB3 setup ran on 8 CPU cores).
*   **Networks:** as in 3.1, ReLU. The action distribution is a plain Gaussian (no tanh) with a learned state-independent std (initially 0.61), so the deterministic action is the linear output that the firmware clips. Weights start as in SB3 (orthogonal, action layer gain 0.01), so training starts near zero action. Some runs still learned to balance while spinning in place at ~8 rad/s, so spinning faster than 3 rad/s now ends the episode (REWARDS.md section 4).
*   **Episodes:** a finished environment starts a fresh episode at once (new slope, physics, heading and command at the current curriculum stage), and timeouts are bootstrapped (Brax's `truncation`). The curriculum is global (`envs/wrappers.py`, REWARDS.md section 3).

---

## 4. Hardware Deployment (ESP32)

Only the actor runs on the robot; the critic is discarded. The deployed policy is `models/policy.bin` (v10, trained with SB3 on the earlier model `XML-test.xml`); new policies come from GPU training runs in `models/runs/` (3.3). Everything below must match training exactly: a sign, unit or order mismatch in a single input is enough to make the robot fall.

### 4.1 Network and Cost
*   **Layers:** `x[88] -> Linear(128) -> ReLU -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Linear(2)`. The last layer has **no activation**.
*   **Size:** 21,794 parameters, 87 KB as `float32`. Keep them `const` so they stay in flash.
*   **Cost:** ~22k multiply-adds per step, roughly 0.3-0.5 ms on a 240 MHz ESP32 with its FPU. Measure it on the board.
*   **No input normalization:** the network takes the scaled frame of 4.4 directly (training uses no observation normalization).

### 4.2 Control Loop Timing
*   **Rate:** 100 Hz (10 ms per step: `EnvironmentConfig.control_period_s`, 5 physics substeps of 2 ms). Run the loop on a hardware timer, not `delay()`.
*   **Latency tolerated:** training delayed the motor command, the IMU and the encoders by 0-2 steps each (up to 20 ms). Keep the time from reading the sensors to writing the PWM well below one step.
*   **Each step:**
    1. Read the IMU and the encoders.
    2. Build the new 11-value frame (4.4) and shift it into the front of the history.
    3. Run the forward pass (4.5).
    4. Clip the two outputs to [-1, 1], write them as PWM (4.6) and keep them as the previous action for the next frame.

### 4.3 Axes and Signs
The sim's body frame is **x = towards the right wheel (along the axle), y = forward, z = up**. The real IMU axes must be remapped to this frame:

| Quantity | Positive when | Source |
|---|---|---|
| `pitch` | the top leans **backward** (driving off forward makes it positive) | attitude estimate, rad |
| `roll` | the top leans towards the **right** wheel | attitude estimate, rad |
| `pitch_rate` | pitch increasing (rotation about +x, the axle towards the right wheel) | gyro, rad/s |
| `roll_rate` | roll increasing (rotation about +y, forward) | gyro, rad/s |
| `yaw_rate` | turning **left** (counter-clockwise seen from above) | gyro z, body frame, rad/s |
| wheel velocity | that wheel **rolls the robot forward** (both wheels, despite the mirrored motors) | encoder, rad/s |
| duty / action | that wheel is driven **forward** | output |
| `v_cmd` / `omega_cmd` | drive forward / turn left | joystick |

Check every row on the real robot before the first balancing test (4.8).

### 4.4 Building the Observation
One frame, in this order (`ACTOR_FRAME_FIELDS` in `envs/balance_env.py`):

| # | Field | Scaling |
|---|---|---|
| 0 | `v_cmd` | divided by 0.4 m/s |
| 1 | `omega_cmd` | divided by 2.5 rad/s |
| 2 | `roll` | rad |
| 3 | `pitch` | rad |
| 4 | `roll_rate` | rad/s |
| 5 | `pitch_rate` | rad/s |
| 6 | `yaw_rate` | rad/s |
| 7 | left wheel velocity | divided by 19.4 rad/s (motor no-load speed) |
| 8 | right wheel velocity | divided by 19.4 rad/s |
| 9 | previous left action | clipped duty from the last step |
| 10 | previous right action | clipped duty from the last step |

*   **Angles:** roll and pitch are the tilt of gravity in the body frame, from a fused attitude estimate (e.g. complementary or Mahony filter on accelerometer and gyro). Training added 0.01 rad of noise; a raw accelerometer angle is far noisier while the robot accelerates.
*   **Rates:** raw gyro in the body frame (training noise 0.05 rad/s). Remove the gyro bias at startup.
*   **Wheel velocities:** from the encoders, in rad/s of the wheel (training noise 0.3 rad/s). Use the encoder count difference over the 10 ms step, or a short moving average.
*   **History:** the input is the last 8 frames, **newest first**: `x[0..10]` = current frame, `x[11..21]` = previous frame, ..., `x[77..87]` = 7 steps ago. Shift with `memmove` or a ring buffer that is read newest first.
*   **Startup:** set the previous action to 0 and fill all 8 slots with copies of the first frame.

### 4.5 Joystick Commands
Apply the same limits as training before scaling:
1. Clamp `v_cmd` to +-0.4 m/s and `omega_cmd` to +-2.5 rad/s.
2. Keep the outer wheel within 0.4 m/s: if `|v| + |omega| * 0.0535 > 0.4` (half track 5.35 cm), multiply **both** by `0.4 / (|v| + |omega| * 0.0535)`.
3. Divide by 0.4 and 2.5 for the frame.

Instant full reversals (+0.3 -> -0.3 m/s) are part of training (v10), so no ramp is required. A slew-rate limit (e.g. 0.6 m/s^2) is still a cheap extra safety margin.

### 4.6 Forward Pass and Motor Output
Weights are exported row-major as `W[out][in]`, so each layer is `y = W x + b`:

```c
#include "policy_weights.h"   // generated by `make export` (4.7)

static void dense(const float *W, const float *b, const float *x, float *y,
                  int n_out, int n_in, bool relu) {
    for (int o = 0; o < n_out; o++) {
        float acc = b[o];
        const float *row = W + o * n_in;
        for (int i = 0; i < n_in; i++) acc += row[i] * x[i];
        y[o] = (relu && acc < 0.0f) ? 0.0f : acc;
    }
}

void policy_step(const float x[88], float duty[2]) {
    float h0[128], h1[64], h2[32], out[2];
    dense(&POLICY_W0[0][0], POLICY_B0, x,  h0, 128, 88,  true);
    dense(&POLICY_W1[0][0], POLICY_B1, h0, h1, 64,  128, true);
    dense(&POLICY_W2[0][0], POLICY_B2, h1, h2, 32,  64,  true);
    dense(&POLICY_W3[0][0], POLICY_B3, h2, out, 2,  32,  false);  // no activation
    for (int i = 0; i < 2; i++)
        duty[i] = fminf(fmaxf(out[i], -1.0f), 1.0f);      // 0 = left, 1 = right
}
```

*   **The output is a duty, not a speed setpoint:** in simulation, duty `d` drives the DC-motor model `torque = kv * (d * 19.4 rad/s - wheel_speed)`, which is what a PWM duty of `d` on the H-bridge does to a real motor (the voltage is `d * V_supply`). Write `|duty|` as the PWM compare value and `sign(duty)` to the direction pins. Do **not** put a wheel-speed PID in between.
*   **Feedback:** the clipped duty is also the "previous action" in the next frame.
*   **Supply voltage:** training assumed the no-load speed of 19.4 rad/s, i.e. the nominal battery voltage. If the battery sags, scale the duty by `V_nominal / V_battery` (clipped to [-1, 1]) so the motor sees the voltage the policy expects.

### 4.7 Exporting the Weights
`make export` (or `.venv/bin/python -m self_balancing_robot.export --model <run dir> --output <file.h>`) writes `models/policy_weights.h` from the newest training run in `models/runs/`. Before writing, it checks that its plain forward pass (NumPy, float32) reproduces the trained policy's actions (Brax's network in JAX) on simulated inputs. The header contains:
*   `POLICY_W0..W3[out][in]` and `POLICY_B0..B3[out]`, plus `POLICY_WEIGHTS`, `POLICY_BIASES`, `POLICY_LAYER_IN` and `POLICY_LAYER_OUT` tables, so a loop over `POLICY_NUM_LAYERS` also works if the network size changes.
*   The constants from 4.2-4.5 (`POLICY_FRAME_SIZE`, `POLICY_HISTORY_LENGTH`, `POLICY_CONTROL_PERIOD_S`, `POLICY_MAX_LINEAR_COMMAND`, `POLICY_MAX_ANGULAR_COMMAND`, `POLICY_HALF_TRACK_M`, `POLICY_WHEEL_NO_LOAD_SPEED`, `POLICY_TILT_LIMIT_RAD`) and the frame field order.

The arrays are `static const`: include the header in one source file only. It is ~400 KB of text but compiles to 85 KB.

The same export also writes the **policy file** `models/policy.bin` (format in the `export.py` docstring). The robot firmware (`ESP32_software/robot`) uses this instead of the header: copy it to `robot/main/policy.bin` and rebuild. Along with the weights and the constants above, the file holds:
*   A CRC-32 of the frame field names. The firmware refuses a policy whose fields, history length, control period or limits differ from its own observation.
*   8 simulated observations and the trained policy's actions for them. The firmware checks its forward pass against these at boot (4.9, step 1).
*   A CRC-32 of the whole file, which also identifies the policy in the firmware log.

### 4.8 What the Policy Was Trained to Handle
*   Mass +-10%, CoM shift up to 8 mm per axis, floor friction +-40%, slopes up to 0.3 rad, pushes up to 1 N for 0.1 s, sensor noise and 0-2 steps of latency (the last stage of `CURRICULUM` in `curriculum.py`).
*   Known weaknesses (`models/archive/v10_reversals_30M/README.md`): forward speed undershoots by ~20%, a forward CoM shift turns into a speed offset (no integral term), and survival on slopes above 0.2 rad is low.
*   Training ends an episode when roll or pitch passes 0.6 rad, the robot spins faster than 3 rad/s, or the chassis touches the ground: past that the policy has never been trained, so cut the motors there.

### 4.9 Bring-Up Checklist
1. **Forward pass:** log a few real 88-value inputs, run them through the Python model (`Actor.load(path).predict(obs)` from `actor.py`, for a run directory or a `policy.bin`) and compare with the ESP32 outputs (should agree to ~1e-5).
2. **Signs:** with the robot held, tilt it, turn it and spin each wheel by hand, and check every row of 4.3 in the logged frame.
3. **Motors:** a small positive duty on each wheel must roll the robot forward.
4. **Timing:** measure the loop period (10 ms) and the sensor-to-PWM time.
5. **First run:** on a tether or with a hand close, zero command, motors cut past 0.6 rad.
