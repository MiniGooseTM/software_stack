"""Drive a trained policy with the keyboard in a live MuJoCo window.

Hold the arrow keys to command the robot: up/down for forward/backward speed and
left/right for turning. The policy only receives the (v_cmd, omega_cmd) command,
exactly like on the real robot. The simulation is the JAX training environment, stepped
one control period at a time (GPU or CPU) and mirrored into a MuJoCo window.
"""

import argparse
import os
import time
from pathlib import Path

import glfw
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

from self_balancing_robot.actor import Actor, resolve_model
from self_balancing_robot.curriculum import CURRICULUM, FULL_COMMAND
from self_balancing_robot.envs.balance_env import ACTOR_KEY, SelfBalancingRobotEnv

# One environment: CPU JAX has less per-step overhead than a GPU (override with JAX_PLATFORMS).
if "JAX_PLATFORMS" not in os.environ:
    jax.config.update("jax_platforms", "cpu")

ARROW_KEYS = (glfw.KEY_UP, glfw.KEY_DOWN, glfw.KEY_LEFT, glfw.KEY_RIGHT)
# The viewer toggles geom groups on 0-4; on many layouts "=" or "+" is Shift + a digit.
DIGIT_KEYS = tuple(range(glfw.KEY_0, glfw.KEY_9 + 1))
# Camera views cycled with V: (name, azimuth relative to the robot heading or None for
# mouse control, elevation in degrees, distance in m). All of them follow the robot.
CAMERA_VIEWS = (
    ("chase", 0.0, -25.0, 1.0),
    ("top-down", 0.0, -89.0, 1.5),
    ("side", 90.0, -5.0, 0.8),
    ("free (mouse)", None, -25.0, 1.0),
)
# Training ranges shown when typing in physics values: those of the last stage.
_LAST_STAGE = CURRICULUM[-1]
_COM_RANGE_MM = _LAST_STAGE.com_offset_max * 1000
# Physics parameters that can be typed in: key -> (label, unit shown, training range).
PHYSICS_KEYS = {
    glfw.KEY_G: (
        "slope",
        "rad",
        f"training range +-{_LAST_STAGE.slope_max:g}",
    ),
    glfw.KEY_F: (
        "friction",
        "coefficient",
        f"training range nominal +-{_LAST_STAGE.friction_fraction:.0%}",
    ),
    glfw.KEY_X: ("CoM x (axle)", "mm", f"training range +-{_COM_RANGE_MM:g}"),
    glfw.KEY_Y: ("CoM y (forward)", "mm", f"training range +-{_COM_RANGE_MM:g}"),
    glfw.KEY_Z: ("CoM z (up)", "mm", f"training range +-{_COM_RANGE_MM:g}"),
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


class TeleopSim:
    """One JAX environment driven step by step, with a held command and pinned physics."""

    def __init__(self, env: SelfBalancingRobotEnv, seed: int = 0):
        self.env = env
        self._reset = jax.jit(env.reset_episode)
        self._drive = jax.jit(self._drive_step)
        self.persistent = {**env.initial_persistent(), "command_held": jnp.array(True)}
        self.command = np.zeros(2, dtype=np.float32)
        self.key = jax.random.PRNGKey(seed)
        self.data = mujoco.MjData(env.mj_model)
        self.reset()

    def _drive_step(self, state, command, action):
        """Set the held command, take one control step, return the values the window shows."""
        state = state.replace(info={**state.info, "command": command})
        state = self.env.step(state, action)
        info = state.info
        readings = {
            "observation": state.obs[ACTOR_KEY],
            "v_cmd": info["command"][0],
            "omega_cmd": info["command"][1],
            "forward_velocity": info["forward_velocity"],
            "yaw_rate": info["yaw_rate"],
            "pitch": info["pitch"],
            "curriculum_stage": info["stage"],
            "terminated": info["terminated"],
            "slope": info["slope"],
            "friction_scale": info["friction_scale"],
            "com_offset": info["com_offset"],
            "forward": self.env.heading_axes(self.env._rotation(state.data.qpos))[0],
            "qpos": state.data.qpos,
            "qvel": state.data.qvel,
            "mocap_quat": state.data.mocap_quat,
            "xfrc_applied": state.data.xfrc_applied,
        }
        return state, readings

    def reset(self) -> None:
        self.key, key = jax.random.split(self.key)
        self.persistent["command"] = jnp.asarray(self.command)
        self.state = self._reset(key, self.persistent)
        self.readings = self._drive_free_readings()

    def _drive_free_readings(self) -> dict:
        info, data = self.state.info, self.state.data
        return jax.device_get(
            {
                "observation": self.state.obs[ACTOR_KEY],
                "slope": info["slope"],
                "friction_scale": info["friction_scale"],
                "com_offset": info["com_offset"],
                "forward": np.array([0.0, 1.0]),
                "qpos": data.qpos,
                "qvel": data.qvel,
                "mocap_quat": data.mocap_quat,
                "xfrc_applied": data.xfrc_applied,
            }
        )

    def hold_command(self, linear_velocity: float, angular_velocity: float) -> None:
        """Use this command, scaled down together so the outer wheel stays within full speed
        (`SelfBalancingRobotEnv.wheel_feasible_command`), instead of random ones."""
        full_speed = FULL_COMMAND[0]
        outer_wheel = abs(linear_velocity) + abs(angular_velocity) * self.env.half_track
        scale = min(1.0, full_speed / outer_wheel) if outer_wheel > 0 else 1.0
        self.command = np.array(
            [linear_velocity * scale, angular_velocity * scale], dtype=np.float32
        )

    def _set_info(self, **values) -> None:
        info = dict(self.state.info)
        for name, value in values.items():
            info[name] = jnp.asarray(value, dtype=info[name].dtype)
        self.state = self.state.replace(info=info)

    def set_physics(
        self,
        slope: float | None = None,
        friction: float | None = None,
        com_offset: np.ndarray | None = None,
    ) -> None:
        """Pin physics values instead of randomizing them.

        ``slope`` is the terrain incline in rad (rising along world +y), ``friction`` the
        floor friction coefficient and ``com_offset`` the CoM shift in m (body frame:
        x = axle, y = forward, z = up). Friction and CoM take effect immediately, the slope
        at the next reset, which places the robot on it.
        """
        if slope is not None:
            self.persistent["override_slope"] = jnp.float32(slope)
        if friction is not None:
            self.persistent["override_friction"] = jnp.float32(friction)
            self._set_info(friction_scale=friction / self.env.nominal_ground_friction)
            self.readings["friction_scale"] = friction / self.env.nominal_ground_friction
        if com_offset is not None:
            self.persistent["override_com"] = jnp.asarray(com_offset, dtype=jnp.float32)
            self._set_info(com_offset=com_offset)
            self.readings["com_offset"] = np.asarray(com_offset)

    @property
    def physics(self) -> dict:
        """Current slope (rad), floor friction coefficient and CoM offset (m)."""
        return {
            "slope": float(self.readings["slope"]),
            "friction": float(self.readings["friction_scale"]) * self.env.nominal_ground_friction,
            "com_offset": np.array(self.readings["com_offset"]),
        }

    @property
    def observation(self) -> np.ndarray:
        return self.readings["observation"]

    def step(self, action: np.ndarray) -> dict:
        self.state, readings = self._drive(
            self.state, jnp.asarray(self.command), jnp.asarray(action, dtype=jnp.float32)
        )
        self.readings = jax.device_get(readings)
        return self.readings

    def sync(self) -> None:
        """Copy the latest state into the MjData shown in the window."""
        for name in ("qpos", "qvel", "mocap_quat", "xfrc_applied"):
            getattr(self.data, name)[:] = self.readings[name]
        mujoco.mj_forward(self.env.mj_model, self.data)

    def heading_degrees(self) -> float:
        forward = self.readings["forward"]
        return float(np.degrees(np.arctan2(forward[1], forward[0])))


class KeyboardTeleop:
    def __init__(self, sim: TeleopSim, max_linear: float, max_angular: float):
        self.sim = sim
        self.env = sim.env
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.speed_fraction = 1.0
        self.camera_view = 0
        self.reset_requested = False
        self.quit_requested = False
        self.editing: int | None = None  # key of the physics value being typed in
        self.edit_text = ""
        self.message = ""
        self.renderer = MujocoRenderer(sim.env.mj_model, sim.data, width=1280, height=720)
        self.renderer.render("human")  # creates the window
        self.viewer = self.renderer.viewer
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
                self.sim.set_physics(
                    slope=0.0,
                    friction=self.env.nominal_ground_friction,
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
        label, unit, _ = PHYSICS_KEYS[self.editing]
        key, self.editing = self.editing, None
        try:
            value = float(self.edit_text)
        except ValueError:
            self.message = f"'{self.edit_text}' is not a number"
            return
        if key == glfw.KEY_G:
            self.sim.set_physics(slope=value)
            self.reset_requested = True  # the robot is placed on the new slope
        elif key == glfw.KEY_F:
            if value <= 0:
                self.message = "friction must be positive"
                return
            self.sim.set_physics(friction=value)
        else:
            com_offset = self.sim.physics["com_offset"]
            com_offset[(glfw.KEY_X, glfw.KEY_Y, glfw.KEY_Z).index(key)] = value / 1000.0
            self.sim.set_physics(com_offset=com_offset)
        self.message = f"{label} = {value:g} {unit}"

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
        _, _, elevation, distance = CAMERA_VIEWS[self.camera_view]
        camera = self.viewer.cam
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = self.env.mj_model.body("base_link").id
        camera.elevation = elevation
        camera.distance = distance

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
        view_name, relative_azimuth, elevation, _ = CAMERA_VIEWS[self.camera_view]
        if relative_azimuth is not None:
            # Turn with the robot; the mouse can still zoom (scroll).
            self.viewer.cam.azimuth = self.sim.heading_degrees() + relative_azimuth
            self.viewer.cam.elevation = elevation
        self.viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, HELP, "")
        physics = self.sim.physics
        com_mm = physics["com_offset"] * 1000.0
        if self.editing is not None:
            label, unit, training_range = PHYSICS_KEYS[self.editing]
            prompt = f"{label} [{unit}], {training_range}"
            self.viewer.add_overlay(
                mujoco.mjtGridPos.mjGRID_TOP, prompt, f"{self.edit_text}_  (Enter / Esc)"
            )
        elif self.message:
            self.viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOP, self.message, "")
        self.viewer.add_overlay(
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            "slope\nfriction\nCoM offset x/y/z",
            f"{physics['slope']:.2f} rad ({np.degrees(physics['slope']):.1f} deg)\n"
            f"{physics['friction']:.2f}\n"
            f"{com_mm[0]:.1f} / {com_mm[1]:.1f} / {com_mm[2]:.1f} mm",
        )
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
        self.sim.sync()
        self.renderer.render("human")


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive a trained policy with the keyboard.")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Training run directory or policy.bin file (default: the newest run).",
    )
    parser.add_argument(
        "--stage",
        default="-1",
        help="Curriculum stage to simulate, by name or index (default -1, the last: noise, "
        'latency, slopes and pushes; "commands 0.5 m/s" for a clean simulation). Stages: '
        + ", ".join(f'{i} "{s.name}"' for i, s in enumerate(CURRICULUM)),
    )
    parser.add_argument("--render-every", type=int, default=2, help="Render every N control steps.")
    args = parser.parse_args()

    try:
        env = SelfBalancingRobotEnv(
            start_stage=args.stage, fixed_difficulty=True, max_episode_steps=2**31 - 1
        )
    except ValueError as error:
        parser.error(str(error))
    actor = Actor.load(resolve_model(args.model))
    linear_limit, angular_limit = FULL_COMMAND
    sim = TeleopSim(env)
    teleop = KeyboardTeleop(sim, linear_limit, angular_limit)

    step = 0
    while teleop.window_open:
        start = time.perf_counter()
        sim.hold_command(*teleop.command())
        info = sim.step(actor.predict(sim.observation))
        if info["terminated"] or teleop.reset_requested:
            if info["terminated"]:
                print("Robot fell, resetting.")
            teleop.reset_requested = False
            sim.reset()
        step += 1
        if step % args.render_every == 0:
            teleop.render(info)
        # Run in real time.
        time.sleep(max(env.dt - (time.perf_counter() - start), 0.0))
    teleop.renderer.close()


if __name__ == "__main__":
    main()
