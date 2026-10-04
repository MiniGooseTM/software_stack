import argparse
from pathlib import Path

import numpy as np

from self_balancing_robot.actor import Actor, resolve_model
from self_balancing_robot.config import DEFAULT_TRAINING_CONFIG
from self_balancing_robot.envs.balance_env import SelfBalancingRobotEnv
from self_balancing_robot.metrics import (
    METRIC_KEYS,
    plot_episode,
    render_trace,
    run_episodes,
    summarize,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained balancing policy.")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Training run directory or policy.bin file (default: the newest run).",
    )
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_TRAINING_CONFIG.evaluation_episodes
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        choices=range(1, 4),
        help="Curriculum stage to run at its last level: 1 = balance, 2 = full commands, "
        "3 = full commands with slopes, pushes, noise and latency.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed of the first episode.")
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Save a command-vs-response plot of every episode to this directory.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Save the first episode as an MP4 here (headless: MUJOCO_GL=egl).",
    )
    args = parser.parse_args()

    model = resolve_model(args.model)
    actor = Actor.load(model)
    env = SelfBalancingRobotEnv()
    # All episodes run in parallel, at the last level of the stage.
    traces = run_episodes(
        env,
        actor,
        [args.seed + episode for episode in range(args.episodes)],
        stage=args.stage,
        record_states=args.video is not None,
    )
    print(f"Evaluating {model} at stage {args.stage} level {traces[0]['level']}")
    summaries = []
    for episode, trace in enumerate(traces):
        summary = summarize(trace)
        summaries.append(summary)
        print(
            f"episode={episode + 1} end={trace['end']} "
            f"duration={trace['time'][-1]:.1f}s reward={summary['reward']:.1f} "
            f"mean|v_err|={summary['linear_error']:.3f}m/s "
            f"mean|yaw_err|={summary['angular_error']:.3f}rad/s "
            f"mean|pitch|={summary['abs_pitch_deg']:.1f}deg"
        )
        if args.plot_dir is not None:
            plot_episode(
                trace,
                args.plot_dir / f"episode_{episode + 1:02d}_stage{args.stage}.png",
                f"Episode {episode + 1},",
            )
    print(
        "mean over episodes: "
        + " ".join(
            f"{key}={np.mean([summary[key] for summary in summaries]):.3f}"
            for key in METRIC_KEYS
        )
    )
    if args.video is not None:
        import imageio

        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.video, render_trace(env, traces[0]), fps=round(1 / env.dt))
        print(f"Saved {args.video}")


if __name__ == "__main__":
    main()
