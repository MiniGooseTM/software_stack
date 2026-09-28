"""
A simple script that runs the minigoose environment.
"""
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import self_balancing_robot.envs

def main():
    env = gym.make("MiniGoose-Balance-v0", render_mode="human")
    print(env.observation_space)
    print(env.action_space)
    # test random policy
    obs, _ = env.reset()
    total_reward = 0

    # Go for 2000 steps
    terminated = False
    truncated = False
    for step in range(2000):
        # Choose a random action
        action = env.action_space.sample()

        # Take a step with the given action
        obs, reward, terminated, truncated, _ = env.step(action)

        # Accumulate reward
        total_reward += reward

        # Show that we stopped
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    print(f"Final obs: {obs}")
    print(f"Final reward: {total_reward}")
    env.close()
if __name__ == "__main__":
    main()