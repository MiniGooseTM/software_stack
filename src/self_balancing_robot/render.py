"""Render recorded episodes to MP4 videos, in a separate process during training.

Training saves the states of each recorded episode (``videos/<name>.npz``) and renders them in
a child process: a missing or broken OpenGL setup can abort the whole process (a headless
server without a display, or a MIG GPU slice, which has no graphics support), and that must
never stop training. The child tries the OpenGL backends one after the other. If none works,
the states files stay; render them on another machine with::

    python -m self_balancing_robot.render models/runs/<run>/videos/*.npz
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import warnings
from pathlib import Path

import imageio
import mujoco
import numpy as np

from self_balancing_robot.config import DEFAULT_ENVIRONMENT_CONFIG, EnvironmentConfig
from self_balancing_robot.envs.balance_env import load_model


def tracking_camera(model: mujoco.MjModel, cfg: EnvironmentConfig) -> mujoco.MjvCamera:
    """Camera that follows the robot base (behind and above it)."""
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = model.body("base_link").id
    camera.distance = cfg.camera_distance_m
    camera.elevation = cfg.camera_elevation_degrees
    camera.azimuth = cfg.camera_azimuth_degrees
    return camera


def render_frames(
    model: mujoco.MjModel,
    cfg: EnvironmentConfig,
    qpos: np.ndarray,
    mocap_quat: np.ndarray,
    width: int = 640,
    height: int = 480,
) -> list[np.ndarray]:
    """Frames of recorded states (needs an OpenGL context: MUJOCO_GL=egl or osmesa headless)."""
    data = mujoco.MjData(model)
    camera = tracking_camera(model, cfg)
    frames = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for state, slope in zip(qpos, mocap_quat):
            data.qpos[:] = state
            data.mocap_quat[:] = slope
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())
    return frames


def save_states(path: Path, trace: dict, model_file: str, fps: int) -> None:
    """Save what `render_file` needs from an episode run with ``record_states=True``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, qpos=trace["qpos"], mocap_quat=trace["mocap_quat"], model_file=model_file, fps=fps
    )


def render_file(states: Path) -> Path:
    """Render a states file to an MP4 next to it."""
    recorded = np.load(states)
    model = load_model(str(recorded["model_file"]))
    frames = render_frames(
        model, DEFAULT_ENVIRONMENT_CONFIG, recorded["qpos"], recorded["mocap_quat"]
    )
    video = states.with_suffix(".mp4")
    imageio.mimsave(video, frames, fps=int(recorded["fps"]))
    return video


class BackgroundRenderer:
    """Renders states files in child processes, one at a time, without blocking training.

    Tries the OpenGL backends in order (MUJOCO_GL if set, GLFW if there is a display, then EGL
    and OSMesa) and keeps the first that works for later videos.
    """

    def __init__(self) -> None:
        candidates = [os.environ.get("MUJOCO_GL"), "glfw" if os.environ.get("DISPLAY") else None]
        self.backends = list(dict.fromkeys(b for b in [*candidates, "egl", "osmesa"] if b))
        self.lock = threading.Lock()
        self.threads: list[threading.Thread] = []
        # Starting a child process while JAX runs makes it warn about fork; the child execs a
        # fresh Python straight away, which is safe.
        warnings.filterwarnings("ignore", message="os.fork", category=RuntimeWarning)

    def render(self, states: Path) -> None:
        thread = threading.Thread(target=self._render, args=(states,))
        thread.start()
        self.threads.append(thread)

    def wait(self) -> None:
        for thread in self.threads:
            thread.join()

    def _render(self, states: Path) -> None:
        with self.lock:
            while self.backends:
                backend = self.backends[0]
                result = subprocess.run(
                    [sys.executable, "-m", "self_balancing_robot.render", str(states)],
                    env={**os.environ, "MUJOCO_GL": backend, "JAX_PLATFORMS": "cpu"},
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    states.unlink()
                    print(f"Saved {states.with_suffix('.mp4')}", flush=True)
                    return
                lines = (result.stderr or result.stdout).strip().splitlines() or ["no output"]
                print(f"Video: MUJOCO_GL={backend} does not work here ({lines[-1]})", flush=True)
                self.backends.pop(0)
            print(
                f"Video: no OpenGL backend works here; render {states} elsewhere with "
                "python -m self_balancing_robot.render <file.npz>",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("states", type=Path, nargs="+", help="States files (videos/*.npz).")
    for states in parser.parse_args().states:
        print(f"Saved {render_file(states)}")


if __name__ == "__main__":
    main()
