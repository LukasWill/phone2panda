"""Fast tests that need no simulator:  python -m pytest -q tests/test_geometry.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2p.sim.controller import R_DOWN, PoseController, grasp_yaw_for_radial, mat2axisangle, rot_z  # noqa: E402
from p2p.sim.env import quat2axisangle, quat2mat  # noqa: E402
from p2p.stats import wilson_ci  # noqa: E402


def _expm(aa):
    """Rodrigues: axis-angle -> rotation matrix."""
    th = np.linalg.norm(aa)
    if th < 1e-12:
        return np.eye(3)
    k = aa / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def test_axisangle_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        aa = rng.normal(size=3)
        aa *= rng.uniform(0, np.pi - 1e-3) / np.linalg.norm(aa)
        assert np.allclose(mat2axisangle(_expm(aa)), aa, atol=1e-6)
    # exactly 180 deg (R_DOWN is a 180 deg rotation about x)
    assert np.allclose(np.abs(mat2axisangle(R_DOWN)), [np.pi, 0, 0], atol=1e-6)


def test_quat_conventions_match():
    """quat2axisangle (LeRobot's state formula) must agree with quat2mat + matrix log."""
    rng = np.random.default_rng(1)
    for _ in range(100):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        if q[3] < 0:
            q = -q  # same rotation; keeps the angle in [0, pi]
        assert np.allclose(_expm(quat2axisangle(q)), quat2mat(q), atol=1e-6)


def test_grasp_yaw_makes_fingers_close_radially():
    for theta in np.linspace(-np.pi, np.pi, 25):
        d = np.array([np.cos(theta), np.sin(theta)])
        psi = grasp_yaw_for_radial(d)
        closing_axis = (rot_z(psi) @ R_DOWN)[:, 1]   # gripper y-axis in the world
        assert abs(abs(closing_axis[:2] @ d) - 1) < 1e-9
        assert -np.pi / 2 <= psi < np.pi / 2


def test_controller_points_at_target_and_respects_limits():
    c = PoseController(max_lin_step=0.02, max_rot_step=0.1)
    a = c(np.zeros(3), R_DOWN, np.array([1.0, 0, 0]), R_DOWN, 1.0)
    assert np.allclose(a[:3], [0.02 / 0.05, 0, 0]) and np.allclose(a[3:6], 0) and a[6] == 1.0
    a = c(np.zeros(3), R_DOWN, np.zeros(3), rot_z(0.5) @ R_DOWN, -1.0)
    assert np.allclose(a[3:6], [0, 0, 0.1 / 0.5])  # pure yaw about world z, clipped


def test_wilson():
    lo, hi = wilson_ci(45, 50)
    assert 0.77 < lo < 0.80 and 0.95 < hi < 0.97
    assert wilson_ci(50, 50)[1] == 1.0 and wilson_ci(0, 50)[0] == 0.0
