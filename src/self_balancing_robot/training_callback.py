from collections import defaultdict
from pathlib import Path

import imageio
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from self_balancing_robot.config import DEFAULT_TRAINING_CONFIG
from self_balancing_robot.envs.balance_env import SelfBalancingRobotEnv
from self_balancing_robot.metrics import (
    METRIC_KEYS,
    plot_episode,
    plot_training_metrics,
    run_episode,
    summarize,
)

PLOT_DIR = Path(__file__).resolve().parents[1] / DEFAULT_TRAINING_CONFIG.plot_dir


def _least_advanced(training_env) -> tuple[int, int]:
    """(stage, level) of the training environment furthest behind in the curriculum."""
    progress = training_env.get_attr("curriculum_progress")
    index = int(np.argmin(progress))
    return (
        training_env.get_attr("curriculum_stage", indices=index)[0],
        training_env.get_attr("difficulty_level", indices=index)[0],
    )


class TrainingMetricsCallback(BaseCallback):
    """Every ``eval_freq`` steps, compare training rollouts with held-out evaluation episodes.

    Writes ``training_metrics.png`` (reward, survival, velocity and yaw-rate tracking,
    pitch and curriculum stage over time) and ``eval_episode_latest.png`` (command vs
    response of one evaluation episode); at the end also ``eval_episode_final.png``.
    """

    def __init__(
        self,
        eval_freq: int = DEFAULT_TRAINING_CONFIG.evaluation_frequency,
        n_eval_episodes: int = DEFAULT_TRAINING_CONFIG.evaluation_episodes,
        max_eval_steps: int = DEFAULT_TRAINING_CONFIG.evaluation_max_steps,
        seed: int = 0,
        plot_dir: Path = PLOT_DIR,
        domain_randomization: bool = True,
    ):
        super().__init__()
        if eval_freq <= 0:
            raise ValueError("eval_freq must be positive")
        if n_eval_episodes <= 0:
            raise ValueError("n_eval_episodes must be positive")
        if max_eval_steps <= 0:
            raise ValueError("max_eval_steps must be positive")

        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.max_eval_steps = max_eval_steps
        self.seed = seed
        self.plot_dir = plot_dir
        self.eval_env = SelfBalancingRobotEnv(domain_randomization=domain_randomization)
        self.history: dict[str, list] = defaultdict(list)
        self._train_sums: dict[str, float] = defaultdict(float)
        self._train_count = 0
        self._next_eval_step = eval_freq

    def _on_training_start(self) -> None:
        self.history.clear()
        self._train_sums.clear()
        self._train_count = 0
        self._next_eval_step = self.eval_freq
        self._save_plot()

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            self._train_sums["linear_error"] += abs(info["forward_velocity"] - info["v_cmd"])
            self._train_sums["angular_error"] += abs(info["yaw_rate"] - info["omega_cmd"])
            self._train_sums["abs_pitch_deg"] += np.degrees(abs(info["pitch"]))
        self._train_count += len(self.locals["infos"])

        if self.num_timesteps >= self._next_eval_step:
            self._evaluate()
            while self._next_eval_step <= self.num_timesteps:
                self._next_eval_step += self.eval_freq
        return True

    def _on_training_end(self) -> None:
        if self._train_count:
            trace = self._evaluate()
            plot_episode(trace, self.plot_dir / "eval_episode_final.png", "Final policy,")
        self.eval_env.close()

    def _evaluate(self) -> dict:
        stages = self.training_env.get_attr("curriculum_stage")
        progress = self.training_env.get_attr("curriculum_progress")
        self.history["steps"].append(self.num_timesteps)
        self.history["stage_min"].append(min(progress))
        self.history["stage_mean"].append(float(np.mean(progress)))
        self.logger.record("curriculum/min_stage", min(stages))
        self.logger.record("curriculum/mean_stage", float(np.mean(stages)))
        # Stage + fraction of its levels passed (2.5 = halfway through the command ramp).
        self.logger.record("curriculum/min_progress", min(progress))
        self.logger.record("curriculum/mean_progress", float(np.mean(progress)))

        # Training rollouts since the last evaluation (reward/survival from the Monitor buffer).
        episodes = list(self.model.ep_info_buffer)
        train = {key: value / max(self._train_count, 1) for key, value in self._train_sums.items()}
        train["reward"] = np.mean([e["r"] for e in episodes]) if episodes else np.nan
        train["survival"] = (
            np.mean([e["l"] >= self.eval_env.max_episode_steps for e in episodes])
            if episodes
            else np.nan
        )
        self._train_sums.clear()
        self._train_count = 0

        # Held-out episodes at the hardest stage and level every training env has reached.
        # Setting it each time also clears the eval env's own curriculum window.
        stage, level = _least_advanced(self.training_env)
        self.eval_env.set_curriculum_stage(stage)
        self.eval_env.set_difficulty_level(level)
        traces = [
            run_episode(self.eval_env, self.model, self.seed + episode, self.max_eval_steps)
            for episode in range(self.n_eval_episodes)
        ]
        summaries = [summarize(trace) for trace in traces]
        for key in METRIC_KEYS:
            test_value = float(np.mean([summary[key] for summary in summaries]))
            self.history["train_" + key].append(float(train.get(key, np.nan)))
            self.history["test_" + key].append(test_value)
            self.logger.record("eval/" + key, test_value)

        self._save_plot()
        plot_episode(
            traces[0],
            self.plot_dir / "eval_episode_latest.png",
            f"After {self.num_timesteps:,} steps,",
        )
        return traces[0]

    def _save_plot(self) -> None:
        plot_training_metrics(self.history, self.plot_dir / "training_metrics.png")


class EvaluationVideoCallback(BaseCallback):
    """Record one deterministic evaluation episode as an MP4 every ``video_freq`` steps
    and once more when training ends."""

    def __init__(
        self,
        video_freq: int = DEFAULT_TRAINING_CONFIG.video_frequency,
        video_dir: Path = DEFAULT_TRAINING_CONFIG.video_path,
        max_steps: int = DEFAULT_TRAINING_CONFIG.evaluation_max_steps,
        seed: int = 0,
        verbose: int = 0,
        domain_randomization: bool = True,
    ):
        super().__init__(verbose)
        if video_freq <= 0:
            raise ValueError("video_freq must be positive")
        self.video_freq = video_freq
        self.video_dir = video_dir
        self.max_steps = max_steps
        self.seed = seed
        self.video_env = SelfBalancingRobotEnv(
            render_mode="rgb_array", domain_randomization=domain_randomization
        )
        self._next_video_step = video_freq

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_video_step:
            self._record_video(f"step_{self.num_timesteps:09d}")
            while self._next_video_step <= self.num_timesteps:
                self._next_video_step += self.video_freq
        return True

    def _on_training_end(self) -> None:
        self._record_video("final")
        self.video_env.close()

    def _record_video(self, name: str) -> Path:
        # Show the hardest stage and level every training environment has reached.
        stage, level = _least_advanced(self.training_env)
        self.video_env.set_curriculum_stage(stage)
        self.video_env.set_difficulty_level(level)
        observation, _ = self.video_env.reset(seed=self.seed)
        frames = [self.video_env.render()]
        for _ in range(self.max_steps):
            action, _ = self.model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = self.video_env.step(action)
            frames.append(self.video_env.render())
            if terminated or truncated:
                break
        self.video_dir.mkdir(parents=True, exist_ok=True)
        path = self.video_dir / f"{name}_stage{stage}_level{level}.mp4"
        imageio.mimsave(path, frames, fps=self.video_env.metadata["render_fps"])
        if self.verbose:
            print(f"Saved evaluation video to {path}")
        return path
