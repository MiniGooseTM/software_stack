"""Evaluation rollouts, summary metrics and plots shared by training and evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from self_balancing_robot.envs.balance_env import SelfBalancingRobotEnv

TRACE_KEYS = ("v_cmd", "forward_velocity", "omega_cmd", "yaw_rate", "pitch")
METRIC_KEYS = ("reward", "survival", "linear_error", "angular_error", "abs_pitch_deg")


def run_episode(
    env: SelfBalancingRobotEnv,
    model,
    seed: int | None = None,
    max_steps: int | None = None,
    frames: list | None = None,
) -> dict:
    """Run one deterministic episode and return its time series.

    If ``frames`` is given, the rendered frames (one before the first step, then after
    every step) are appended to it.
    """
    observation, _ = env.reset(seed=seed)
    if frames is not None:
        frames.append(env.render())
    trace: dict[str, list] = {key: [] for key in (*TRACE_KEYS, "action", "reward")}
    info: dict = {}
    for _ in range(max_steps or env.max_episode_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        if frames is not None:
            frames.append(env.render())
        elif env.render_mode == "human":
            env.render()
        for key in TRACE_KEYS:
            trace[key].append(info[key])
        trace["action"].append(np.clip(action, -1.0, 1.0))
        trace["reward"].append(reward)
        if terminated or truncated:
            break
    result = {key: np.asarray(values) for key, values in trace.items()}
    result["time"] = np.arange(1, len(result["reward"]) + 1) * env.dt
    result["stage"] = info.get("curriculum_stage", env.curriculum_stage)
    result["end"] = info.get("terminated_reason") or "max_steps"
    result["survived"] = result["end"] in ("timeout", "max_steps")
    result["upright_limit"] = env.environment_config.upright_limit_radians
    return result


def summarize(trace: dict) -> dict[str, float]:
    return {
        "reward": float(trace["reward"].sum()),
        "survival": float(trace["survived"]),
        "linear_error": float(np.mean(np.abs(trace["forward_velocity"] - trace["v_cmd"]))),
        "angular_error": float(np.mean(np.abs(trace["yaw_rate"] - trace["omega_cmd"]))),
        "abs_pitch_deg": float(np.degrees(np.mean(np.abs(trace["pitch"])))),
    }


def plot_episode(trace: dict, output_path: Path, title: str = "") -> None:
    """Commanded vs actual velocity and yaw rate, pitch and motor commands over one episode."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = trace["time"]
    figure, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].step(time, trace["v_cmd"], where="post", color="k", ls="--", label="command")
    axes[0].plot(time, trace["forward_velocity"], label="actual")
    axes[0].set_ylabel("Forward velocity (m/s)")
    axes[1].step(time, trace["omega_cmd"], where="post", color="k", ls="--", label="command")
    axes[1].plot(time, trace["yaw_rate"], color="tab:orange", label="actual")
    axes[1].set_ylabel("Yaw rate (rad/s)")
    limit = np.degrees(trace["upright_limit"])
    axes[2].plot(time, np.degrees(trace["pitch"]), color="tab:green", label="pitch")
    axes[2].axhline(0.0, color="k", ls="--", lw=1, label="upright (reward reference)")
    for sign in (-1, 1):
        axes[2].axhline(sign * limit, color="tab:red", ls=":", lw=1)
    axes[2].set_ylabel("Pitch (deg)")
    axes[3].plot(time, trace["action"][:, 0], label="left")
    axes[3].plot(time, trace["action"][:, 1], label="right")
    axes[3].set_ylabel("Motor command (PWM)")
    axes[3].set_ylim(-1.05, 1.05)
    axes[3].set_xlabel("Time (s)")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize="small")
    metrics = summarize(trace)
    figure.suptitle(
        f"{title} stage {trace['stage']}, ended by {trace['end']} after {time[-1]:.1f} s\n"
        f"mean |v error| {metrics['linear_error']:.3f} m/s, "
        f"mean |yaw error| {metrics['angular_error']:.3f} rad/s, "
        f"mean |pitch| {metrics['abs_pitch_deg']:.1f} deg"
    )
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def plot_training_metrics(history: dict[str, list], output_path: Path) -> None:
    """Six-panel training dashboard: train rollouts vs held-out evaluation over timesteps."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = (
        ("reward", "Episode reward"),
        ("survival", "Episodes surviving to timeout (%)"),
        ("linear_error", "Mean |v - v_cmd| (m/s)"),
        ("angular_error", "Mean |yaw rate - omega_cmd| (rad/s)"),
        ("abs_pitch_deg", "Mean |pitch| (deg, 0 = upright)"),
    )
    steps = np.asarray(history.get("steps", []))
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for axis, (key, label) in zip(axes.flat, panels):
        scale = 100.0 if key == "survival" else 1.0
        for prefix, name in (("train_", "Train"), ("test_", "Eval")):
            values = history.get(prefix + key, [])
            if len(values) == len(steps) and len(steps):
                axis.plot(steps, np.asarray(values, dtype=float) * scale, label=name)
        axis.set_title(label)
        if key == "survival":
            axis.set_ylim(0, 105)
    stage_axis = axes.flat[-1]
    if history.get("stage_min"):
        stage_axis.step(steps, history["stage_min"], where="post", label="min (eval level)")
        stage_axis.step(steps, history["stage_mean"], where="post", label="mean")
    stage_axis.set_title("Curriculum: stage + fraction of its levels passed")
    stage_axis.set_yticks([1, 2, 3])
    stage_changes = [
        steps[i]
        for i in range(1, len(history.get("stage_min", [])))
        if int(history["stage_min"][i]) != int(history["stage_min"][i - 1])
    ]
    for axis in axes.flat:
        for step in stage_changes:
            axis.axvline(step, color="grey", ls=":", lw=1)
        axis.grid(True, alpha=0.3)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize="small")
    for axis in axes[-1]:
        axis.set_xlabel("Training timesteps")
    figure.suptitle("Training progress (dotted lines: curriculum stage advances)")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
