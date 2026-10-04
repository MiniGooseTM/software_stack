"""Evaluation rollouts, summary metrics, videos and plots shared by training and evaluation."""

from __future__ import annotations

import functools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from self_balancing_robot.actor import Actor, forward
from self_balancing_robot.envs.balance_env import ACTOR_KEY, SelfBalancingRobotEnv
from self_balancing_robot.render import render_frames

TRACE_KEYS = ("v_cmd", "forward_velocity", "omega_cmd", "yaw_rate", "pitch")
METRIC_KEYS = ("reward", "survival", "linear_error", "angular_error", "abs_pitch_deg")


@functools.cache
def _rollout_fn(env: SelfBalancingRobotEnv, steps: int, record_states: bool):
    """Jitted batch of deterministic episodes: (layers, keys, persistent) -> per-step arrays."""

    def episode(layers, key, persistent):
        state = env.reset_episode(key, persistent)

        def body(carry, _):
            state, alive = carry
            action = jnp.clip(forward(layers, state.obs[ACTOR_KEY]), -1.0, 1.0)
            command = state.info["command"]
            next_state = env.step(state, action)
            record = {
                "v_cmd": command[0],
                "omega_cmd": command[1],
                "forward_velocity": next_state.info["forward_velocity"],
                "yaw_rate": next_state.info["yaw_rate"],
                "pitch": next_state.info["pitch"],
                "action": action,
                "reward": next_state.reward,
                "alive": alive,
                "terminated": next_state.info["terminated"],
                "truncated": next_state.info["truncation"] > 0,
            }
            if record_states:
                record["qpos"] = next_state.data.qpos
                record["mocap_quat"] = next_state.data.mocap_quat
            return (next_state, alive & (next_state.done == 0)), record

        _, trace = jax.lax.scan(body, (state, jnp.array(True)), None, steps)
        return trace

    return jax.jit(jax.vmap(episode, in_axes=(None, 0, None)))


def run_episodes(
    env: SelfBalancingRobotEnv,
    actor: Actor,
    seeds: list[int],
    max_steps: int | None = None,
    stage: int | str | None = None,
    record_states: bool = False,
) -> list[dict]:
    """Run deterministic episodes in parallel (one per seed) and return their time series.

    ``stage`` (index or name) defaults to the env's start stage.
    """
    steps = max_steps or env.max_episode_steps
    persistent = env.initial_persistent()
    if stage is not None:
        persistent["stage"] = jnp.int32(env.stage_index(stage))
    keys = jnp.stack([jax.random.PRNGKey(seed) for seed in seeds])
    layers = [(jnp.asarray(w), jnp.asarray(b)) for w, b in actor.layers]
    batch = jax.device_get(_rollout_fn(env, steps, record_states)(layers, keys, persistent))
    traces = []
    for i in range(len(seeds)):
        trace = {key: np.asarray(value[i]) for key, value in batch.items()}
        length = int(trace.pop("alive").sum())
        terminated = bool(trace["terminated"][length - 1])
        truncated = bool(trace["truncated"][length - 1])
        result = {key: value[:length] for key, value in trace.items()}
        result["time"] = np.arange(1, length + 1) * env.dt
        result["stage"] = int(persistent["stage"])
        result["stage_name"] = env.curriculum[result["stage"]].name
        result["end"] = "fall" if terminated else "timeout" if truncated else "max_steps"
        result["survived"] = not terminated
        result["upright_limit"] = env.environment_config.upright_limit_radians
        traces.append(result)
    return traces


def run_episode(env: SelfBalancingRobotEnv, actor: Actor, seed: int = 0, **kwargs) -> dict:
    """Run one deterministic episode and return its time series."""
    return run_episodes(env, actor, [seed], **kwargs)[0]


def summarize(trace: dict) -> dict[str, float]:
    return {
        "reward": float(trace["reward"].sum()),
        "survival": float(trace["survived"]),
        "linear_error": float(np.mean(np.abs(trace["forward_velocity"] - trace["v_cmd"]))),
        "angular_error": float(np.mean(np.abs(trace["yaw_rate"] - trace["omega_cmd"]))),
        "abs_pitch_deg": float(np.degrees(np.mean(np.abs(trace["pitch"])))),
    }


def render_trace(
    env: SelfBalancingRobotEnv, trace: dict, width: int = 640, height: int = 480
) -> list[np.ndarray]:
    """Frames of an episode run with ``record_states=True`` (needs MUJOCO_GL=egl headless)."""
    return render_frames(
        env.mj_model,
        env.environment_config,
        trace["qpos"],
        trace["mocap_quat"],
        width,
        height,
    )


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
        f'{title} stage {trace["stage"]} "{trace["stage_name"]}", ended by {trace["end"]} after {time[-1]:.1f} s\n'
        f"mean |v error| {metrics['linear_error']:.3f} m/s, "
        f"mean |yaw error| {metrics['angular_error']:.3f} rad/s, "
        f"mean |pitch| {metrics['abs_pitch_deg']:.1f} deg"
    )
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def plot_training_metrics(
    history: dict[str, list], output_path: Path, stage_names: list[str] | None = None
) -> None:
    """Six-panel training dashboard over timesteps: deterministic episodes at the current
    curriculum stage and at the last stage, and the curriculum stage."""
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
        for prefix, name in (("stage_", "Current stage"), ("final_", "Last stage")):
            values = history.get(prefix + key, [])
            if len(values) == len(steps) and len(steps):
                axis.plot(steps, np.asarray(values, dtype=float) * scale, label=name)
        axis.set_title(label)
        if key == "survival":
            axis.set_ylim(0, 105)
    stage_axis = axes.flat[-1]
    stages = history.get("stage", [])
    if stages:
        stage_axis.step(steps, stages, where="post")
    stage_axis.set_title("Curriculum stage")
    if stage_names:
        stage_axis.set_yticks(range(len(stage_names)), stage_names, fontsize="small")
    stage_changes = [steps[i] for i in range(1, len(stages)) if stages[i] != stages[i - 1]]
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
