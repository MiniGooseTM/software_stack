import argparse
from pathlib import Path

from stable_baselines3 import PPO

from self_balancing_robot.envs.balance_env import SelfBalancingRobotEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained balancing policy.")
    parser.add_argument("--model", type=Path, default=Path("models/ppo_balance.zip"))
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    env = SelfBalancingRobotEnv(render_mode="human")
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
