"""Drive a trained policy with the keyboard in a live MuJoCo window.

Hold the arrow keys to command the robot: up/down for forward/backward speed and
left/right for turning. The policy only receives the (v_cmd, omega_cmd) command,
exactly like on the real robot.
"""

import argparse
import time
from pathlib import Path
from typing import NamedTuple

import glfw
import mujoco
import numpy as np

from self_balancing_robot.config import (
    DEFAULT_CURRICULUM_CONFIG,
    DEFAULT_RANDOMIZATION_CONFIG,
    DEFAULT_TRAINING_CONFIG,
)
from self_balancing_robot.envs.balance_env import SelfBalancingRobotEnv
from self_balancing_robot.policy import load_model

ARROW_KEYS = (glfw.KEY_UP, glfw.KEY_DOWN, glfw.KEY_LEFT, glfw.KEY_RIGHT)
# The viewer toggles geom groups on 0-4; on many layouts "=" or "+" is Shift + a digit.
DIGIT_KEYS = tuple(range(glfw.KEY_0, glfw.KEY_9 + 1))


class CameraView(NamedTuple):
    name: str
    azimuth: float | None  # relative to the robot heading; None = free mouse view
    elevation: float
    distance: float


# Camera views cycled with V. All of them follow the robot.
CAMERA_VIEWS = (
    CameraView("chase", 0.0, -25.0, 1.0),
    CameraView("top-down", 0.0, -89.0, 1.5),
    CameraView("side", 90.0, -5.0, 0.8),
    CameraView("free (mouse)", None, -25.0, 1.0),
)


class PhysicsField(NamedTuple):
    label: str
    unit: str
    training_range: str
    com_axis: int | None = None  # index into the CoM offset for X / Y / Z


# Physics parameters that can be typed in: key -> field.
PHYSICS_KEYS = {
    glfw.KEY_G: PhysicsField(
        "slope", "rad", f"training range +-{DEFAULT_RANDOMIZATION_CONFIG.slope_radians:g}"
    ),
    glfw.KEY_F: PhysicsField("friction", "coefficient", "training range 0.48 to 1.12"),
    glfw.KEY_X: PhysicsField("CoM x (axle)", "mm", "training range +-4.9", com_axis=0),
    glfw.KEY_Y: PhysicsField("CoM y (forward)", "mm", "training range +-4.9", com_axis=1),
    glfw.KEY_Z: PhysicsField("CoM z (up)", "mm", "training range +-4.9", com_axis=2),
}
HELP = (
    "Arrows: drive (hold)\n"
    "+ / -: speed limit +/- 25%\n"
    "V: next camera view\n"
    "G: set slope, F: set friction\n"
    "X / Y / Z: set CoM offset\n"
    "N: nominal physics\n"
    "Backspace: reset robot\n"
    "H: hide menus, Esc: quit"
)


class KeyboardTeleop:
    def __init__(self, env: SelfBalancingRobotEnv, max_linear: float, max_angular: float):
        self.env = env
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.speed_fraction = 1.0
        self.camera_view = 0
        self.reset_requested = False
        self.quit_requested = False
        self.editing: int | None = None  # key of the physics value being typed in
        self.edit_text = ""
        self.message = ""
        env.render()  # creates the window
        self.viewer = env.mujoco_renderer.viewer
        self._viewer_key_callback = self.viewer._key_callback
        glfw.set_key_callback(self.viewer.window, self._key_callback)
        glfw.set_char_callback(self.viewer.window, self._char_callback)
        self._apply_camera_view()

    def _key_callback(self, window, key: int, scancode, action: int, mods) -> None:
        # Arrows are polled in `command()`; the viewer would otherwise use them to pause.
        # Tab would switch to the model's fixed cameras, of which there are none.
        if key in ARROW_KEYS or key in DIGIT_KEYS or key == glfw.KEY_TAB:
            return
        if key in PHYSICS_KEYS or key == glfw.KEY_N or self.editing is not None:
            if action == glfw.PRESS:
                self._physics_key(key)
            return
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                self.quit_requested = True
            elif key == glfw.KEY_BACKSPACE:
                self.reset_requested = True
            elif key == glfw.KEY_V:
                self.camera_view = (self.camera_view + 1) % len(CAMERA_VIEWS)
                self._apply_camera_view()
        if key not in (glfw.KEY_ESCAPE, glfw.KEY_V):
            self._viewer_key_callback(window, key, scancode, action, mods)

    def _physics_key(self, key: int) -> None:
        if self.editing is None:
            if key == glfw.KEY_N:
                self.env.set_physics(
                    slope=0.0,
                    friction=self.env.nominal_friction,
                    com_offset=np.zeros(3),
                )
                self.reset_requested = True
                self.message = "nominal physics"
            else:
                self.editing, self.edit_text, self.message = key, "", ""
        elif key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
            self._apply_edit()
        elif key == glfw.KEY_BACKSPACE:
            self.edit_text = self.edit_text[:-1]
        elif key == glfw.KEY_ESCAPE:
            self.editing = None

    def _apply_edit(self) -> None:
        key, self.editing = self.editing, None
        field = PHYSICS_KEYS[key]
        try:
            value = float(self.edit_text)
        except ValueError:
            self.message = f"'{self.edit_text}' is not a number"
            return
        if key == glfw.KEY_G:
            self.env.set_physics(slope=value)
            self.reset_requested = True  # the robot is placed on the new slope
        elif key == glfw.KEY_F:
            if value <= 0:
                self.message = "friction must be positive"
                return
            self.env.set_physics(friction=value)
        else:
            com_offset = self.env.physics["com_offset"]
            com_offset[field.com_axis] = value / 1000.0
            self.env.set_physics(com_offset=com_offset)
        self.message = f"{field.label} = {value:g} {field.unit}"

    def _char_callback(self, window, codepoint: int) -> None:
        # Characters, not key codes, so "+" and "-" work on any keyboard layout and keypad.
        character = chr(codepoint)
        if self.editing is not None:
            if character in "0123456789.-":
                self.edit_text += character
            return
        if character in "+=":
            self.speed_fraction = min(self.speed_fraction + 0.25, 1.0)
        elif character in "-_":
            self.speed_fraction = max(self.speed_fraction - 0.25, 0.25)

    def _apply_camera_view(self) -> None:
        view = CAMERA_VIEWS[self.camera_view]
        camera = self.viewer.cam
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = self.env.base_body_id
        camera.elevation = view.elevation
        camera.distance = view.distance

    def _pressed(self, key: int) -> bool:
        return glfw.get_key(self.viewer.window, key) == glfw.PRESS

    def command(self) -> tuple[float, float]:
        forward = self._pressed(glfw.KEY_UP) - self._pressed(glfw.KEY_DOWN)
        turn = self._pressed(glfw.KEY_LEFT) - self._pressed(glfw.KEY_RIGHT)
        return (
            forward * self.speed_fraction * self.max_linear,
            turn * self.speed_fraction * self.max_angular,
        )

    @property
    def window_open(self) -> bool:
        return not (self.quit_requested or glfw.window_should_close(self.viewer.window))

    def render(self, info: dict) -> None:
        view = CAMERA_VIEWS[self.camera_view]
        if view.azimuth is not None:
            # Turn with the robot; the mouse can still zoom (scroll).
            self.viewer.cam.azimuth = self.env.heading_degrees + view.azimuth
            self.viewer.cam.elevation = view.elevation
        self.viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, HELP, "")
        self._draw_edit_prompt()
        self._draw_physics()
        self._draw_status(info, view.name)
        self.env.render()

    def _draw_edit_prompt(self) -> None:
        if self.editing is not None:
            field = PHYSICS_KEYS[self.editing]
            prompt = f"{field.label} [{field.unit}], {field.training_range}"
            self.viewer.add_overlay(
                mujoco.mjtGridPos.mjGRID_TOP, prompt, f"{self.edit_text}_  (Enter / Esc)"
            )
        elif self.message:
            self.viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOP, self.message, "")

    def _draw_physics(self) -> None:
        physics = self.env.physics
        com_mm = physics["com_offset"] * 1000.0
        self.viewer.add_overlay(
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            "slope\nfriction\nCoM offset x/y/z",
            f"{physics['slope']:.2f} rad ({np.degrees(physics['slope']):.1f} deg)\n"
            f"{physics['friction']:.2f}\n"
            f"{com_mm[0]:.1f} / {com_mm[1]:.1f} / {com_mm[2]:.1f} mm",
        )

    def _draw_status(self, info: dict, view_name: str) -> None:
        self.viewer.add_overlay(
            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            "v cmd / actual\nyaw cmd / actual\npitch\nspeed limit\nstage\ncamera",
            f"{info['v_cmd']:.2f} / {info['forward_velocity']:.2f} m/s\n"
            f"{info['omega_cmd']:.2f} / {info['yaw_rate']:.2f} rad/s\n"
            f"{np.degrees(info['pitch']):.1f} deg\n"
            f"{self.speed_fraction:.0%}\n"
            f"{info['curriculum_stage']}\n"
            f"{view_name}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive a trained policy with the keyboard.")
    parser.add_argument("--model", type=Path, default=DEFAULT_TRAINING_CONFIG.evaluation_model_path)
    parser.add_argument(
        "--stage",
        type=int,
        default=2,
        choices=range(1, 4),
        help="2 = clean simulation, 3 = add noise, latency, slopes and random pushes.",
    )
    parser.add_argument("--render-every", type=int, default=2, help="Render every N control steps.")
    args = parser.parse_args()

    env = SelfBalancingRobotEnv(
        render_mode="human",
        curriculum_stage=args.stage,
        max_episode_steps=np.iinfo(np.int64).max,
    )
    env.set_difficulty_level(env.max_level)  # full stage-3 slopes and pushes
    model = load_model(args.model, device="cpu")
    linear_limit, angular_limit = DEFAULT_CURRICULUM_CONFIG.command_limits
    env.hold_command(0.0, 0.0)
    observation, _ = env.reset()
    teleop = KeyboardTeleop(env, linear_limit, angular_limit)

    step = 0
    while teleop.window_open:
        start = time.perf_counter()
        env.hold_command(*teleop.command())
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, _, info = env.step(action)
        if terminated or teleop.reset_requested:
            if terminated:
                print(f"Robot fell ({info['terminated_reason']}), resetting.")
            teleop.reset_requested = False
            observation, _ = env.reset()
        step += 1
        if step % args.render_every == 0:
            teleop.render(info)
        # Run in real time.
        time.sleep(max(env.dt - (time.perf_counter() - start), 0.0))
    env.close()


if __name__ == "__main__":
    main()
