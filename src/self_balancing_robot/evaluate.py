import argparse
from pathlib import Path

from stable_baselines3 import PPO

from self_balancing_robot.config import DEFAULT_TRAINING_CONFIG
from self_balancing_robot.envs.balance_env import SelfBalancingRobotEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained balancing policy.")
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_TRAINING_CONFIG.evaluation_model_path
    )
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_TRAINING_CONFIG.evaluation_episodes
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show the MuJoCo viewer (requires a graphical display).",
    )
    args = parser.parse_args()

    env = SelfBalancingRobotEnv(render_mode="human" if args.render else None)
    model = PPO.load(args.model, env=env)
    for episode in range(args.episodes):
        observation, _ = env.reset()
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        print(
            f"episode={episode + 1} reward={total_reward:.2f} "
            f"stage={info['curriculum_stage']} pitch={info['pitch']:.3f} "
            f"v_error={abs(info['linear_velocity'] - info['v_cmd']):.3f}"
        )
    env.close()


if __name__ == "__main__":
    main()
