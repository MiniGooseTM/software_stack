from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from self_balancing_robot.envs.balance_env import SelfBalancingRobotEnv


class BalanceErrorCallback(BaseCallback):
    """Track training and held-out mean absolute pitch error during PPO training."""

    def __init__(
        self,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        max_eval_steps: int = 1_000,
        seed: int = 0,
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
        self.eval_env = SelfBalancingRobotEnv()
        self.train_steps: list[int] = []
        self.train_errors: list[float] = []
        self.test_steps: list[int] = []
        self.test_errors: list[float] = []
        self._train_error_sum = 0.0
        self._train_observation_count = 0
        self._next_eval_step = eval_freq

    def _on_step(self) -> bool:
        observations = np.asarray(self.locals["new_obs"])
        self._train_error_sum += float(np.abs(observations[:, 1]).sum())
        self._train_observation_count += observations.shape[0]

        if self.num_timesteps >= self._next_eval_step:
            self._record_errors()
            while self._next_eval_step <= self.num_timesteps:
                self._next_eval_step += self.eval_freq
        return True

    def _on_training_end(self) -> None:
        if self._train_observation_count:
            self._record_errors()
        self.eval_env.close()

    def _record_errors(self) -> None:
        if self._train_observation_count:
            self.train_steps.append(self.num_timesteps)
            self.train_errors.append(self._train_error_sum / self._train_observation_count)
            self._train_error_sum = 0.0
            self._train_observation_count = 0

        self.test_steps.append(self.num_timesteps)
        self.test_errors.append(self._evaluate_test_error())
        save_training_plot(
            self.train_steps,
            self.train_errors,
            self.test_steps,
            self.test_errors,
            Path(__file__).resolve().parents[1] / "plots" / "training_errors.png",
        )

    def _evaluate_test_error(self) -> float:
        episode_errors = []
        for episode in range(self.n_eval_episodes):
            observation, _ = self.eval_env.reset(seed=self.seed + episode)
            step_errors = []
            for _ in range(self.max_eval_steps):
                action, _ = self.model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, _ = self.eval_env.step(action)
                step_errors.append(abs(float(observation[1])))
                if terminated or truncated:
                    break
            episode_errors.append(float(np.mean(step_errors)))
        return float(np.mean(episode_errors))


def save_training_plot(
    train_steps: list[int],
    train_errors: list[float],
    test_steps: list[int],
    test_errors: list[float],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()
    axis.plot(train_steps, train_errors, label="Train")
    axis.plot(test_steps, test_errors, label="Test")
    axis.set_xlabel("Training timesteps")
    axis.set_ylabel("Mean absolute pitch error (rad)")
    axis.set_title("Balancing training and test error")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
