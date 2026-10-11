"""Quantitative evaluation suites used by the main ``evaluate`` command."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from self_balancing_robot.config import (
    DEFAULT_ENVIRONMENT_CONFIG,
    EnvironmentConfig,
)
from self_balancing_robot.envs.balance_env import SelfBalancingRobotEnv
from self_balancing_robot.metrics import run_episode, summarize


def _episode_seed(seed: int, index: int, episodes: int) -> int:
    return seed + index * episodes


def evaluate_balance(env: SelfBalancingRobotEnv, model, episodes: int, seed: int) -> dict:
    """Measure position drift and absolute pitch error with zero velocity command."""
    env.hold_command(0.0, 0.0)
    results = []
    for episode in range(episodes):
        env.set_curriculum_stage(1)
        trace = run_episode(env, model, seed=seed + episode)
        displacement = np.hypot(
            trace["position_x"] - trace["initial_position"][0],
            trace["position_y"] - trace["initial_position"][1],
        )
        metrics = summarize(trace)
        results.append(
            {
                "survived": bool(trace["survived"]),
                "duration_s": float(trace["time"][-1]),
                "mean_position_drift_m": float(displacement.mean()),
                "final_position_drift_m": float(displacement[-1]),
                "position_drift_rate_m_s": float(displacement[-1] / trace["time"][-1]),
                "mean_absolute_pitch_error_rad": float(np.mean(np.abs(trace["pitch"]))),
                "mean_absolute_pitch_error_deg": metrics["abs_pitch_deg"],
                "end": trace["end"],
            }
        )
    return {
        "episodes": episodes,
        "mean_survival_rate": float(np.mean([result["survived"] for result in results])),
        "mean_position_drift_m": float(
            np.mean([result["mean_position_drift_m"] for result in results])
        ),
        "mean_final_position_drift_m": float(
            np.mean([result["final_position_drift_m"] for result in results])
        ),
        "mean_position_drift_rate_m_s": float(
            np.mean([result["position_drift_rate_m_s"] for result in results])
        ),
        "mean_absolute_pitch_error_rad": float(
            np.mean([result["mean_absolute_pitch_error_rad"] for result in results])
        ),
        "mean_absolute_pitch_error_deg": float(
            np.mean([result["mean_absolute_pitch_error_deg"] for result in results])
        ),
        "per_episode": results,
    }


def evaluate_tracking(
    env: SelfBalancingRobotEnv,
    model,
    episodes: int,
    seed: int,
    settle_seconds: float,
) -> dict:
    """Measure steady-state MAE across signed linear and angular command setpoints."""
    cases = (
        ("linear", 0.1, 0.0),
        ("linear", -0.1, 0.0),
        ("linear", 0.2, 0.0),
        ("linear", -0.2, 0.0),
        ("angular", 0.0, 0.5),
        ("angular", 0.0, -0.5),
        ("angular", 0.0, 1.0),
        ("angular", 0.0, -1.0),
    )
    case_results = []
    errors: dict[str, list[np.ndarray]] = {"linear": [], "angular": []}
    for case_index, (component, linear, angular) in enumerate(cases):
        env.set_curriculum_stage(2)
        env.set_difficulty_level(env.max_level)
        env.hold_command(linear, angular)
        samples = []
        survived = []
        for episode in range(episodes):
            env.set_curriculum_stage(2)
            env.set_difficulty_level(env.max_level)
            trace = run_episode(
                env,
                model,
                seed=_episode_seed(seed, case_index, episodes) + episode,
            )
            valid = trace["time"] >= settle_seconds
            if component == "linear":
                error = np.abs(trace["forward_velocity"][valid] - trace["v_cmd"][valid])
            else:
                error = np.abs(trace["yaw_rate"][valid] - trace["omega_cmd"][valid])
            if error.size:
                samples.append(error)
            survived.append(bool(trace["survived"]))
        combined = np.concatenate(samples) if samples else np.asarray([], dtype=float)
        if combined.size:
            errors[component].append(combined)
        case_results.append(
            {
                "component": component,
                "v_command_m_s": linear,
                "omega_command_rad_s": angular,
                "steady_state_mae": float(combined.mean()) if combined.size else None,
                "steady_state_mae_unit": "m/s" if component == "linear" else "rad/s",
                "sample_count": int(combined.size),
                "survival_rate": float(np.mean(survived)),
            }
        )

    def overall_mae(component: str) -> float | None:
        return (
            float(np.concatenate(errors[component]).mean()) if errors[component] else None
        )

    return {
        "episodes_per_command": episodes,
        "settle_seconds_excluded": settle_seconds,
        "linear_velocity_mae_m_s": overall_mae("linear"),
        "angular_velocity_mae_rad_s": overall_mae("angular"),
        "per_command": case_results,
    }


def _sensor_scenarios(rates_hz: list[float], control_hz: float) -> list[dict]:
    scenarios = [
        {
            "name": "baseline",
            "imu_interval": 1,
            "encoder_interval": 1,
            "imu_rate_hz": control_hz,
            "encoder_rate_hz": control_hz,
        }
    ]
    for rate in rates_hz:
        if rate <= 0 or rate >= control_hz:
            raise ValueError(
                f"Sensor rates must be positive and below control rate ({control_hz:g} Hz)"
            )
        interval = round(control_hz / rate)
        achieved_rate = control_hz / interval
        if not np.isclose(achieved_rate, rate, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"{rate:g} Hz is not an integer divisor of the {control_hz:g} Hz control rate"
            )
        for name, imu_interval, encoder_interval in (
            ("imu", interval, 1),
            ("encoders", 1, interval),
            ("imu_and_encoders", interval, interval),
        ):
            scenarios.append(
                {
                    "name": f"{name}_{rate:g}hz",
                    "imu_interval": imu_interval,
                    "encoder_interval": encoder_interval,
                    "imu_rate_hz": control_hz / imu_interval,
                    "encoder_rate_hz": control_hz / encoder_interval,
                }
            )
    return scenarios


def evaluate_robustness(
    model,
    episodes: int,
    seed: int,
    max_steps: int,
    sensor_rates_hz: list[float],
    base_environment_config: EnvironmentConfig = DEFAULT_ENVIRONMENT_CONFIG,
) -> dict:
    """Compare full-rate sensors with slower IMU, encoder, and combined sensor updates."""
    probe = SelfBalancingRobotEnv(environment_config=base_environment_config)
    control_hz = 1.0 / probe.dt
    probe.close()
    scenarios = _sensor_scenarios(sensor_rates_hz, control_hz)
    results = []
    for scenario_index, scenario in enumerate(scenarios):
        env_config = replace(
            base_environment_config,
            imu_update_interval_steps=scenario["imu_interval"],
            encoder_update_interval_steps=scenario["encoder_interval"],
            max_episode_steps=max_steps,
        )
        env = SelfBalancingRobotEnv(curriculum_stage=3, environment_config=env_config)
        env.set_difficulty_level(env.max_level)
        traces = []
        for episode in range(episodes):
            env.set_curriculum_stage(3)
            env.set_difficulty_level(env.max_level)
            traces.append(
                run_episode(
                    env,
                    model,
                    seed=_episode_seed(seed, scenario_index, episodes) + episode,
                    max_steps=max_steps,
                )
            )
        summaries = [summarize(trace) for trace in traces]
        result = {
            **scenario,
            "episodes": episodes,
            "survival_rate": float(np.mean([trace["survived"] for trace in traces])),
            "mean_duration_s": float(np.mean([trace["time"][-1] for trace in traces])),
            "mean_absolute_pitch_error_deg": float(
                np.mean([summary["abs_pitch_deg"] for summary in summaries])
            ),
            "linear_velocity_mae_m_s": float(
                np.mean([summary["linear_error"] for summary in summaries])
            ),
            "angular_velocity_mae_rad_s": float(
                np.mean([summary["angular_error"] for summary in summaries])
            ),
            "end_reasons": {
                reason: sum(trace["end"] == reason for trace in traces)
                for reason in sorted({trace["end"] for trace in traces})
            },
        }
        results.append(result)
        env.close()

    baseline_survival = results[0]["survival_rate"]
    for result in results:
        result["survival_rate_change_vs_baseline"] = result["survival_rate"] - baseline_survival
    return {
        "control_rate_hz": control_hz,
        "episodes_per_scenario": episodes,
        "max_steps": max_steps,
        "scenarios": results,
    }


def run_quantitative_suite(
    suite: str,
    model,
    episodes: int,
    seed: int,
    max_steps: int,
    settle_seconds: float,
    sensor_rates_hz: list[float],
) -> dict:
    results = {
        "seed": seed,
        "suite": suite,
    }
    if suite in ("balance", "all"):
        env = SelfBalancingRobotEnv(
            curriculum_stage=1,
            environment_config=replace(
                DEFAULT_ENVIRONMENT_CONFIG, max_episode_steps=max_steps
            ),
        )
        try:
            results["balance"] = evaluate_balance(env, model, episodes, seed)
        finally:
            env.close()
    if suite in ("tracking", "all"):
        env = SelfBalancingRobotEnv(
            curriculum_stage=2,
            environment_config=replace(
                DEFAULT_ENVIRONMENT_CONFIG, max_episode_steps=max_steps
            ),
        )
        try:
            results["tracking"] = evaluate_tracking(
                env, model, episodes, seed, settle_seconds
            )
        finally:
            env.close()
    if suite in ("robustness", "all"):
        results["robustness"] = evaluate_robustness(
            model, episodes, seed, max_steps, sensor_rates_hz
        )
    return results
