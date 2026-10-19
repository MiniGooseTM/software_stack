# Changes on `policy_test` relative to `gpu_training`

`policy_test` branches off the tip of `gpu_training` (`19f5c64`); `gpu_training` has no commits since. There are 5 commits, all dated 2026-10-11:

| Commit | Message |
|---|---|
| `1901d1e` | added wheel torque to domain randomization, increased probability of full stop commands |
| `4133330` | updated tests |
| `7fcf396` | test policy |
| `32e34f9` | changed curriculum and added randomization techniques |
| `85571d7` | the new policy |

The branch raises the command range, makes the curriculum longer and more gradual, adds three new domain-randomization sources (wheel torque kicks, battery voltage, explicit stop commands), lets each curriculum stage set its own reward weights, and adds a newly trained policy.

## 1. Command range: 0.3 m/s → 0.5 m/s

- `FULL_COMMAND` goes from `(0.3, 1.5)` to `(0.5, 3.0)` (m/s, rad/s) in `curriculum.py`.
- `yaw_rate_limit_rad_s` (the termination for spinning) goes from 3.0 to 6.0 to make room for the higher turn rate commands.
- The exported header now reads `POLICY_MAX_LINEAR_COMMAND 0.5` and `POLICY_MAX_ANGULAR_COMMAND 3.0`, so **the ESP32 firmware has to scale joystick commands to the new range.**
- In the README and the `teleop` help text, the "clean simulation" stage name changes from `"commands 0.3 m/s"` to `"commands 0.5 m/s"`.

## 2. Curriculum (`curriculum.py`)

The curriculum grows from 10 to 12 stages:

```
balance
commands 0.1 / 0.2 / 0.3 / 0.4 / 0.5 m/s     (was 0.1 / 0.2 / 0.3)
randomization 1/6 … 6/6
```

### Command stages
- The range grows in fifths of `FULL_COMMAND`, i.e. steps of 0.1 m/s and 0.6 rad/s.
- **Sensor noise ramps in alongside**: IMU angle, IMU rate and encoder noise reach k/5 of their full values (0.01 rad, 0.05 rad/s, 0.3 rad/s) at command stage k. Before, all noise switched on at once at `randomization 1/6`. In run `20261010-165211`, survival there dropped from ~100% to 35%.
- **Latency ramps in**: up to 1 step from 0.2 m/s and up to 2 steps from 0.4 m/s. Before, it also switched on all at once at `randomization 1/6`.
- **New `stop_probability = 0.2`**: a command that isn't a reversal is a full stop `(0, 0)` with this probability. Before, only ~1% of commands were full stops, and earlier policies drifted while standing still.

### Randomization stages
- The maximum push drops from 1.0 N to **0.5 N**. A 0.1 s push of 1 N gave the 0.27 kg robot ~0.37 m/s, which is more than the old full command.
- Wheel torque kicks are new (section 3), growing in sixths up to 0.02 N·m.
- Battery voltage is new (section 3), drawn from (10.5, 12.5) V in every randomization stage.
- `speed_ratio` (the speed-tracking threshold to leave a stage) is lowered from 0.85 to **0.75**.
- Noise and latency are no longer set here because the command stages already set them.
- Slope (up to 0.3 rad), CoM shift (up to 8 mm), mass (±10%) and friction (±40%) are unchanged.
- The constants `PUSH_MAX`, `COM_OFFSET_MAX`, `WHEEL_KICK_MAX`, `IMU_ANGLE_NOISE`, `IMU_RATE_NOISE` and `ENCODER_NOISE` replace the hard-coded numbers.

## 3. New domain randomization (`envs/balance_env.py`)

- **Wheel torque kicks** model bumps, stiction and gearbox torque errors.
  - Each wheel gets its own torque pulse, applied through `qfrc_applied` on the wheel joints.
  - A pulse lasts 0.05 s, and the next one comes every 1–3 s.
  - The magnitude uses the same band sampling as the other banded quantities (`BANDED["wheel_kick"]`).
- **Battery voltage** is drawn per episode and scales the motors relative to `NOMINAL_VOLTAGE = 11.1 V` (3S LiPo).
  - The no-load speed (`actuator_ctrlrange`) and stall torque (`actuator_forcerange`) both scale with the voltage.
  - The commanded wheel speed (duty × no-load speed) is scaled the same way.
  - Over 10.5–12.5 V, this gives −5% to +13% of nominal motor speed and torque.
- The push logic moves into a generic `_pulse()` helper, which both the pushes and the wheel kicks now use. `_sample_banded()` takes a `shape` argument so it can draw an independent value per wheel.
- **The critic's privileged observation has 3 new fields**: `battery_voltage_deviation`, `wheel_kick_left` and `wheel_kick_right`. The actor's input is unchanged (88 inputs).

## 4. Rewards

- **Each stage can now set its own reward weights.** `Stage` has a `rewards: RewardConfig` field, which defaults to `DEFAULT_REWARD_CONFIG`.
  - `stage_table()` flattens the reward fields into per-stage columns (`smoothness_weight`, …), and the environment reads the weights of the current stage inside JAX.
  - `SelfBalancingRobotEnv` no longer takes a `reward_config` argument.
  - `validate()` now checks the reward fields too, rejecting negative values.
- Default weights:
  - `action_weight` goes from 0.01 to **0.05**.
  - `smoothness_weight` goes from 0.5 to **1.0**.
- `randomization 5/6` and `6/6` use `smoothness_weight = 5.0`. Before, smoothness was <1% of the reward. It isn't raised from the start because early on it would mostly penalize exploration noise.

## 5. Network and training (`config.py`)

- The critic goes from `(128, 64)` to **`(128, 64, 32)`**. The actor stays `(128, 64, 32)`.
- `total_timesteps` goes from 200M to **250M**.

## 6. Trained policy

- `models/policy.bin` and `models/policy_weights.h` are replaced by run `models/runs/20261011-171132`. The architecture is the same (88 → 128 → 64 → 32 → 2, 90 KB).
- The same policy is also saved as `models_share/more_randomization_speedy/` (the two `policy.bin` files are identical).

## 7. Tests

- New tests cover:
  - stage-specific reward weights
  - the full-stop command probability
  - voltage scaling of motor speed and torque
  - when wheel kicks happen and that they hit the wheel joints
  - wheel kicks being off in stages that don't have them
  - the reward columns in the stage table
  - validation of the new interval and voltage fields
- Existing tests are updated for 12 stages, the 0.5 m/s range and the new band floors.

**Test status** (`pytest tests/test_curriculum.py tests/test_balance_env.py`): 46 passed, **2 failed**. Both failures come from the tests not matching the final curriculum:

| Test | Expects | Code has |
|---|---|---|
| `test_curriculum.py::test_default_curriculum_keeps_the_tuned_progression` | randomization `speed_ratio == 0.85` | `0.75` |
| `test_balance_env.py::test_last_stages_weigh_smoothness_more` | last two smoothness weights `[5.0, 10.0]` | `[5.0, 5.0]` |

## 8. Loose ends

- The README's teleop paragraph still says the speed limit is "100% = 0.3 m/s and 1.5 rad/s" and mentions limiting the outer wheel to 0.3 m/s. These should be 0.5 m/s and 3.0 rad/s now.
- The comment above `FULL_COMMAND` in `curriculum.py` still explains why 0.3 m/s was chosen ("0.5 m/s left ~14% … and v4 fell more"), which contradicts the new value.
- In the `RANDOMIZATION_5/6` comment, "14 at a weight of 10" suggests that 10 was the intended last-stage weight. The test expects 10 too.
