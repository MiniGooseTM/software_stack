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
    Commands divided by the full command range ($v_{x,cmd} / 0.3$ m/s, $\omega_{z,cmd} / 1.5$ rad/s, so in $[-1, 1]$; the joystick command must first be limited so the outer wheel needs at most 0.3 m/s, as in training), angles in rad (IMU attitude estimate), rates in rad/s (gyro, body frame), wheel velocities divided by the motor no-load speed (19.4 rad/s), and $a_{t-1}$ is the policy's own previous output.
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

---

## 4. Hardware Deployment (ESP32)

Regardless of which training pipeline is chosen, the resulting ESP32 policy is identical in structure: a highly efficient feedforward MLP.

*   **Export:** Export the trained Actor/Student weights as a flat `.h` C++ array.
*   **Inference Constraints:** Utilize only standard matrix multiplications and ReLUs (no transcendental functions like `tanh` or `sigmoid`). This ensures execution times well under `<1ms` on a single-core MCU.
*   **Execution Loop:** 
    1. Read IMU and Encoder states.
    2. Shift the new 11-state array into the front of the observation buffer (dropping the oldest frame), using the same units and scaling as in training.
    3. Run the MLP forward pass manually using Eigen.
    4. Output scaled PWM to the motor drivers.