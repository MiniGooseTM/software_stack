# Lightweight RL Controller Architectures for 2-Wheeled Self-Balancing Robot

This document outlines two training pipelines for a highly efficient, proprioceptive-only neural network policy designed for real-time deployment on an ESP32 microcontroller. Both methods rely on leveraging privileged simulation data to train a lightweight, history-based feedforward network (MLP).

## 1. Common State and Action Spaces

### 1.1 The Action Space ($A_t$)
*   **Dimensions:** 2
*   **Vector:** $[v_{left\_motor\_cmd}, v_{right\_motor\_cmd}]$ normalized to $[-1, 1]$.

### 1.2 The Privileged State ($S_t$) - Teacher/Critic Only
Used exclusively during simulation training. It is fully Markovian (no history window) because it contains exact physical truths.
*   **Proprioceptive Core:** $v_{x, cmd}, \omega_{z, cmd}$, Pitch, Pitch Rate, Yaw Rate, Left/Right wheel velocities, $a_{t-1}$.
*   **Privileged Variables:** Exact Center of Mass (CoM) offset $(x, y, z)$, exact floor friction coefficient, current external push forces $(F_x, F_y, F_z)$, exact motor delay/lag.

### 1.3 The Proprioceptive History ($O_t$) - Student/Actor Only
Used by the ESP32 to infer the privileged variables implicitly. 
*   **Base Observation:** 11 states (Commands, IMU rates/angles, Wheel velocities, previous actions).
*   **Temporal Window:** $H = 5$ to $10$ frames.
*   **Input Vector:** Concatenated history $[O_t, O_{t-1}, ..., O_{t-H}]$ resulting in an input size of $11 \times H$.

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
*   **The Actor (ESP32 Policy):** `Linear(11 * H) -> ReLU -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Output(2)`
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
    2. Shift the new 11-state array into the observation buffer (dropping the oldest frame).
    3. Run the MLP forward pass manually using Eigen.
    4. Output scaled PWM to the motor drivers.