import argparse
import json
from pathlib import Path

import numpy as np

from self_balancing_robot.config import DEFAULT_TRAINING_CONFIG
from self_balancing_robot.envs.balance_env import SelfBalancingRobotEnv
from self_balancing_robot.evaluate_suite import run_quantitative_suite
from self_balancing_robot.metrics import METRIC_KEYS, plot_episode, run_episode, summarize
from self_balancing_robot.policy import load_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained balancing policy and run quantitative test suites."
    )
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_TRAINING_CONFIG.evaluation_model_path
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
    parser.add_argument("--seed", type=int, default=None, help="Seed of the first episode.")
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Save a command-vs-response plot of every episode to this directory.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show the MuJoCo viewer (requires a graphical display).",
    )
    parser.add_argument(
        "--suite",
        choices=("standard", "balance", "tracking", "robustness", "all"),
        default="standard",
        help="Evaluation mode. The default preserves the per-episode summary and plot behavior.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_TRAINING_CONFIG.evaluation_max_steps,
        help="Maximum steps per episode in quantitative suites.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help="Initial seconds excluded from velocity tracking MAE.",
    )
    parser.add_argument(
        "--sensor-rates-hz",
        type=float,
        nargs="+",
        default=[50.0, 25.0, 10.0],
        help="Reduced sensor rates tested by the robustness suite.",
    )
    parser.add_argument(
        "--domain-randomization",
        choices=("none", "full"),
        default=None,
        help="Toggle domain randomization independently of curriculum stage. "
        "For quantitative suites, defaults to full.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON results file for quantitative suites.",
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.max_steps <= 0 or args.settle_seconds < 0:
        parser.error("--max-steps must be positive and --settle-seconds cannot be negative")
    if args.suite == "standard" and args.output is not None:
        parser.error("--output is only available with a quantitative --suite")

    if args.suite != "standard":
        if args.render or args.plot_dir is not None:
            parser.error("--render and --plot-dir are only available with --suite standard")
        model = load_model(args.model, device="cpu")
        results = {
            "model": str(args.model),
            **run_quantitative_suite(
                args.suite,
                model,
                args.episodes,
                DEFAULT_TRAINING_CONFIG.seed if args.seed is None else args.seed,
                args.max_steps,
                args.settle_seconds,
                args.sensor_rates_hz,
                args.domain_randomization or "full",
            ),
        }
        rendered = json.dumps(results, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    stage = args.stage
    env = SelfBalancingRobotEnv(
        render_mode="human" if args.render else None,
        curriculum_stage=stage,
        domain_randomization=args.domain_randomization != "none",
    )
    model = load_model(args.model, env=env, device="cpu")
    env.set_difficulty_level(env.max_level)  # full commands (and slopes and pushes in stage 3)
    summaries = []
    for episode in range(args.episodes):
        seed = None if args.seed is None else args.seed + episode
        trace = run_episode(env, model, seed=seed)
        # Keep the requested stage even if the env's own curriculum would advance.
        env.set_curriculum_stage(args.stage)
        env.set_difficulty_level(env.max_level)
        summary = summarize(trace)
        summaries.append(summary)
        print(
            f"episode={episode + 1} stage={trace['stage']} end={trace['end']} "
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
    env.close()


if __name__ == "__main__":
    main()
