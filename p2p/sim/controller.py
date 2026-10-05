"""From "where I want the gripper to be" to LIBERO's 7-D action.

The action space (robosuite OSC_POSE controller, delta mode, 20 Hz):
    a[0:3] in [-1, 1]  ->  position goal  = current_pos + 0.05 * a[0:3]           (metres)
    a[3:6] in [-1, 1]  ->  rotation goal  = Exp(0.5 * a[3:6]) @ current_R          (axis-angle, WORLD frame)
    a[6]               ->  gripper: -1 open, +1 close
An operational-space controller (stiffness kp = 150) then computes joint torques that pull the
gripper toward that goal during the 50 ms control step. It does NOT get there in one step,
so a delta action behaves like a velocity command, and a policy is a feedback loop.

So to follow a target pose we use a simple proportional law:
    a_pos = clip(target - current, max step) / 0.05
    a_rot = clip(log(R_target @ R_current^T), max step) / 0.5
Two nuances that matter later:
  * Step limits (max_lin_step, max_rot_step) set the speed. They also keep actions away from
    the +-1 saturation, where an imitation policy can no longer tell "fast" from "very fast".
  * The rotation error is R_target @ R_current^T (pre-multiplied), because robosuite applies the
    delta as Exp(delta) @ R_current. Getting this order wrong produces rotations about the wrong axes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POS_SCALE = 0.05   # metres per unit action  (robosuite osc_pose.json: output_max)
ROT_SCALE = 0.5    # radians per unit action
GRIPPER_OPEN, GRIPPER_CLOSE = -1.0, 1.0

# Top-down gripper orientation in the LIBERO world: approach axis (gripper z) points down,
# fingers close along gripper y = world -y. This is (almost exactly) LIBERO's reset pose.
R_DOWN = np.diag([1.0, -1.0, -1.0])


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def mat2axisangle(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis * angle (the matrix logarithm), robust near 0 and pi."""
    cos = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-6:
        return np.zeros(3)
    if np.pi - angle < 1e-4:  # near 180 deg: axis from the symmetric part
        B = (R + np.eye(3)) / 2
        axis = np.sqrt(np.clip(np.diag(B), 0, None))
        k = int(np.argmax(axis))
        axis = B[:, k] / np.sqrt(B[k, k])
        return axis / np.linalg.norm(axis) * angle
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(angle))
    return axis * angle


def grasp_yaw_for_radial(direction_xy: np.ndarray) -> float:
    """Gripper yaw (about world z, applied to R_DOWN) so the fingers close along `direction_xy`.

    With R_DOWN the closing axis is world -y. After rotating by yaw psi it is
    (sin psi, -cos psi). Matching a direction at angle theta gives psi = theta + pi/2, and since
    a closing axis has no sign (left/right finger are symmetric) we wrap to [-pi/2, pi/2): the
    smallest wrist rotation from the reset pose.
    """
    theta = np.arctan2(direction_xy[1], direction_xy[0])
    psi = theta + np.pi / 2
    return float((psi + np.pi / 2) % np.pi - np.pi / 2)


@dataclass
class PoseController:
    max_lin_step: float = 0.025   # m of goal lead per step (the OSC lags, so the real speed is lower)
    max_rot_step: float = 0.15    # rad of goal lead per step

    def __call__(self, eef_pos, eef_R, target_pos, target_R, gripper: float) -> np.ndarray:
        dp = np.asarray(target_pos) - eef_pos
        n = np.linalg.norm(dp)
        if n > self.max_lin_step:
            dp *= self.max_lin_step / n
        dr = mat2axisangle(np.asarray(target_R) @ eef_R.T)
        n = np.linalg.norm(dr)
        if n > self.max_rot_step:
            dr *= self.max_rot_step / n
        a = np.concatenate([dp / POS_SCALE, dr / ROT_SCALE, [gripper]])
        return np.clip(a, -1.0, 1.0)
