# RL Curriculum Learning Strategy for Two-Wheeled Balancing Robot

This document outlines the curriculum learning strategy for training a two-wheeled self-balancing robot using Proximal Policy Optimization (PPO). 

To prevent catastrophic forgetting this strategy uses a **unified reward function** from timestep zero. The curriculum advances strictly by progressively widening the probability distribution of the commanded velocities ($v_{cmd}$ and $\omega_{cmd}$), rather than changing the reward formulas.

---

## 1. Unified Reward Function Formulation

The total reward $R_t$ at any timestep is a weighted sum of survival bonuses, tracking performance, and energy/smoothness penalties. We use squared exponential (Gaussian) kernels to bound the tracking rewards strictly between $[0, 1]$, which prevents gradient explosions and ensures stable PPO value estimation.

### 1.1 Survival & Posture
*   **Alive Bonus:** A constant reward provided as long as the robot has not fallen (e.g., $|\theta| < \theta_{max}$). This prevents the agent from terminating early to avoid accumulating penalties.
    $$r_{alive} = 1.0$$
*   **Pitch Regularization:** Encourages the chassis to stay near vertical, but uses a soft kernel so the robot can still pitch slightly to accelerate.
    $$r_{pitch} = \exp\left(-\frac{\theta^2}{\sigma_\theta}\right)$$

### 1.2 Command Tracking
*   **Linear Velocity Tracking:** Replaces the need for a global $x, y$ position penalty. When $v_{cmd} = 0$, this explicitly acts as an anti-drift penalty.
    $$r_{v} = \exp\left(-\frac{(v_x - v_{cmd})^2}{\sigma_v}\right)$$
*   **Angular Velocity Tracking:** Encourages following the yaw rate command.
    $$r_{\omega} = \exp\left(-\frac{(\omega_z - \omega_{cmd})^2}{\sigma_\omega}\right)$$

### 1.3 Control Penalties
*   **Action Effort:** Penalizes high motor torque/PWM commands to save battery and reduce motor wear.
    $$r_{action} = -\sum_{i \in \{L, R\}} a_{i, t}^2$$
*   **Action Smoothness:** Heavily penalizes high-frequency oscillations (jitter) between consecutive motor commands.
    $$r_{smooth} = -\sum_{i \in \{L, R\}} (a_{i, t} - a_{i, t-1})^2$$

### 1.4 Total Reward Equation
$$R_t = w_1 r_{alive} + w_2 r_{pitch} + w_3 r_{v} + w_4 r_{\omega} + w_5 r_{action} + w_6 r_{smooth}$$

---

## 2. Recommended Weights and Variances

These values serve as a robust starting point for an IsaacLab / Isaac Gym environment.

| Component | Weight ($w_i$) | Kernel Variance ($\sigma$) | Rationale |
| :--- | :--- | :--- | :--- |
| **Alive Bonus** | $w_1 = 2.0$ | N/A | Must be large enough to offset all potential penalties so the robot wants to survive. |
| **Pitch** | $w_2 = 0.5$ | $\sigma_\theta = 0.05$ | Keeps the robot upright without strictly forbidding the tilt required for acceleration. |
| **Linear Vel** | $w_3 = 1.5$ | $\sigma_v = 0.25$ | High priority. A variance of 0.25 means a 0.5 m/s error still yields ~36% of the reward. |
| **Angular Vel**| $w_4 = 1.0$ | $\sigma_\omega = 0.5$ | Slightly lower priority than standing upright and moving forward. |
| **Action** | $w_5 = 0.01$ | N/A | Kept small to prevent the robot from turning off motors and falling just to save energy. |
| **Smoothness** | $w_6 = 0.05$ | N/A | Critical for sim-to-real transfer to prevent motor destruction on the physical hardware. |

---

## 3. The Curriculum Steps

The reward function remains identical across all steps. The curriculum is advanced by changing the environment's command sampling distribution. 

### Step 1: Upright Stabilization (Hover)
*   **Command Distribution:** $v_{cmd} = 0.0$ m/s, $\omega_{cmd} = 0.0$ rad/s (Fixed).
*   **Objective:** The robot learns to balance the inverted pendulum. Because $v_{cmd} = 0$, the $r_v$ tracking reward acts as a strict anti-drift penalty, keeping the robot hovering in its starting location.
*   **Advancement Condition:** 95% of episodes reach the maximum timeout (e.g., 1000 steps) without falling.

### Step 2: Micro-Mobility
*   **Command Distribution:** $v_{cmd} \sim \mathcal{U}(-0.05, 0.05)$ m/s, $\omega_{cmd} \sim \mathcal{U}(-0.3, 0.3)$ rad/s.
*   **Command Resampling:** Randomize the commands every 3 to 5 seconds during the episode.
*   **Objective:** The robot learns to intentionally break its perfectly vertical stance to initiate small movements, and more importantly, learns how to brake and reverse direction dynamically.
*   **Advancement Condition:** Average velocity tracking error drops below 0.01 m/s.

### Step 3: Full Dynamic Tracking
*   **Command Distribution:** $v_{cmd} \sim \mathcal{U}(-0.3, 0.3)$ m/s, $\omega_{cmd} \sim \mathcal{U}(-1.0, 1.0)$ rad/s.
*   **Objective:** The robot learns aggressive acceleration and "high-speed" cornering. At these speeds, the $r_{pitch}$ reward naturally conflicts with $r_v$. The network will learn the optimal Pareto frontier (tilting exactly enough to track the speed without violating the safety threshold).

### Step 4: Domain Randomization & Disturbance
*   **Command Distribution:** Same as Step 3.
*   **Environmental Injection:** 
    *   Randomize chassis mass ($\pm 10\%$) at episode start.
    *   Randomize the position of the center of mass ($10\%$) at episode start.
    *   Randomize floor friction at episode start.
    *   Randomize terrain slope at episode start.
    *   Randomize IMU, encoder and motors latency at episode start.
    *   Randomize IMU and encoder noise at every timestep.
    *   Apply random lateral and longitudinal impulse forces ($F_x, F_y$) to the chassis every few seconds.
*   **Objective:** This final step solidifies the policy. The network learns robust recovery maneuvers under velocity demands.

---

## 4. Episode Termination Conditions

An episode must immediately terminate, returning a reward of $0$ for the remainder of the theoretical timestep limit, if any of the following occur:

1.  **Pitch Failure:** $|\theta| > \theta_{cmd}$ radians. The robot has fallen beyond the point of physical recovery.
2.  **Base Collision:** Any part of the chassis (other than the wheels) touches the ground.
3.  **Timeout:** The episode reaches the maximum length to prevent infinite loops and force buffer resets.