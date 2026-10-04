"""Quick PID balancing controller for env_v0.xml (goose.xml on flat ground).

Uses only sensors the real robot has (IMU accel + gyro, wheel encoders):
pitch is estimated with a complementary filter, not read from the sim state.

Control structure (cascade):
    outer loop: wheel position + velocity -> pitch setpoint (keeps the robot in place)
    inner loop: pitch PID                 -> wheel command [rad/s]

Sign convention: pitch is rotation about the body x axis (the wheel axle).
A positive wheel command drives towards +y and its reaction torque pitches the
body towards +pitch, so leaning with +pitch needs a *negative* command.

Usage:
    python balance_pid.py              # run in the viewer
    python balance_pid.py --headless   # run 10 s without viewer and print a summary
"""
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

XML_PATH = Path(__file__).parent / "env_v0.xml"

# Initial tilt so the controller has something to do [rad]
INITIAL_PITCH = 0.1

# Inner loop: pitch error [rad] -> wheel command [rad/s]
KP = 80.0
KI = 20.0
KD = 4.0

# Outer loop: wheel velocity [rad/s] and position [rad] -> pitch setpoint [rad]
KP_VEL = 0.02
KP_POS = 0.002

# Complementary filter: trust in gyro integration vs accelerometer angle
ALPHA = 0.98


class PID:
    def __init__(self, kp, ki, kd, i_limit):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_limit = i_limit
        self.integral = 0.0

    def update(self, error, error_rate, dt):
        self.integral = np.clip(self.integral + error * dt, -self.i_limit, self.i_limit)
        # Derivative on the measured rate (gyro) instead of differentiating the error
        return self.kp * error + self.ki * self.integral + self.kd * error_rate


class BalanceController:
    def __init__(self, model, dt):
        self.dt = dt
        # Assume the robot starts roughly upright; the filter converges from there.
        # (Don't initialise from the first accel sample: it's dominated by contact impulses.)
        self.pitch = 0.0
        self.pid = PID(KP, KI, KD, i_limit=0.2)
        self.ctrl_max = model.actuator_ctrlrange[:, 1].min()

    def update(self, accel, gyro, wheel_pos, wheel_vel):
        pitch_acc = np.arctan2(accel[1], accel[2])
        pitch_rate = gyro[0]
        self.pitch = ALPHA * (self.pitch + pitch_rate * self.dt) + (1 - ALPHA) * pitch_acc

        target_pitch = KP_VEL * wheel_vel + KP_POS * wheel_pos
        u = -self.pid.update(self.pitch - target_pitch, pitch_rate, self.dt)
        return np.clip(u, -self.ctrl_max, self.ctrl_max)


def reset(model, data):
    mujoco.mj_resetData(model, data)
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([1.0, 0.0, 0.0]), INITIAL_PITCH)
    data.qpos[3:7] = quat
    mujoco.mj_forward(model, data)


def control_step(model, data, ctrl):
    accel = data.sensor("imu_accel").data
    gyro = data.sensor("imu_gyro").data
    wheel_pos = 0.5 * (data.sensor("left_wheel_pos").data[0] + data.sensor("right_wheel_pos").data[0])
    wheel_vel = 0.5 * (data.sensor("left_wheel_vel").data[0] + data.sensor("right_wheel_vel").data[0])
    u = ctrl.update(accel, gyro, wheel_pos, wheel_vel)
    data.ctrl[:] = u
    mujoco.mj_step(model, data)
    return u


def true_pitch(data):
    """Ground-truth pitch about the body x axis, for checking the estimate."""
    R = data.xmat[1].reshape(3, 3)
    return np.arctan2(R[2, 1], R[2, 2])


def run_headless(model, data, duration=10.0):
    ctrl = BalanceController(model, model.opt.timestep)
    reset(model, data)
    while data.time < duration:
        u = control_step(model, data, ctrl)
        if round(data.time / model.opt.timestep) % 200 == 0:
            print(f"t={data.time:5.2f}  pitch={np.degrees(true_pitch(data)):6.2f} deg  "
                  f"est={np.degrees(ctrl.pitch):6.2f} deg  u={u:6.2f} rad/s  "
                  f"pos=({data.qpos[0]:6.3f}, {data.qpos[1]:6.3f}) m")
    upright = abs(true_pitch(data)) < np.radians(10)
    print("BALANCING" if upright else "FELL OVER")


def run_viewer(model, data):
    import mujoco.viewer

    ctrl = BalanceController(model, model.opt.timestep)
    reset(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.05]
        viewer.cam.distance = 0.8
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -25
        while viewer.is_running():
            step_start = time.time()
            control_step(model, data, ctrl)
            viewer.sync()
            slack = model.opt.timestep - (time.time() - step_start)
            if slack > 0:
                time.sleep(slack)


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    if "--headless" in sys.argv:
        run_headless(model, data)
    else:
        run_viewer(model, data)
