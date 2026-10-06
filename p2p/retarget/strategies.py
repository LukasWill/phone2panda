"""Three ways to turn one human demo (table frame) into robot waypoints for one LIBERO scene.

The question behind them: what should transfer from a phone video, the HAND or the INTENT?

  absolute         The hand path, mapped with ONE fixed table->robot transform for all demos
                   (how you would use a calibrated camera). It ignores where the objects are in the
                   new scene, so it only works if the scene happens to match your table.
  object_relative  The hand path, re-anchored on the objects: the approach is expressed relative to
                   the bowl, the carry is blended from the bowl frame to the plate frame, the release
                   is relative to the plate. The path SHAPE (approach direction, lift profile) is kept.
  intent           Only two facts from the demo: where on the rim you grasped (an angle) and where on
                   the plate you put the bowl (an offset). The robot's own scripted motion does the rest.

Shared embodiment choices (applied identically, so the comparison is fair):
  * frames: you sit where the robot base is. Table frame (x right, y away from you) -> LIBERO
    (x away from the robot, y left): x_sim = y_table, y_sim = -x_table.
  * sizes: offsets measured on your objects are scaled to the sim objects (your plate is 1.45x
    wider than LIBERO's, so a 3 cm off-centre placement becomes ~2 cm).
  * fingers: you pinch the top ~1 cm of the rim; the Panda's pads need ~2.5 cm of overlap, so every
    hand-derived height is shifted down by PINCH_TO_GRIP_Z. This is a gripper property, not a scene one.
"""
from __future__ import annotations

import numpy as np

from p2p.config import HUMAN_BOWL_HEIGHT, HUMAN_BOWL_RIM_DIAMETER, HUMAN_PLATE_DIAMETER
from p2p.sim.controller import GRIPPER_CLOSE, GRIPPER_OPEN, grasp_yaw_for_radial
from p2p.sim.env import BOWL_HEIGHT, BOWL_RADIUS, PLATE_HEIGHT, PLATE_RADIUS, Scene
from p2p.sim.scripted import Waypoint, plan_bowl_to_plate

R_TS = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # table vector -> sim vector
TABLE_Z_SIM = 0.898                    # LIBERO table top
PINCH_TO_GRIP_Z = -0.015               # human pinch ~1 cm below rim -> robot grip site ~2.5 cm below rim
BOWL_SCALE = (BOWL_RADIUS - 0.006) / (HUMAN_BOWL_RIM_DIAMETER / 2)   # rim-wall radius, sim / human
PLATE_SCALE = PLATE_RADIUS / (HUMAN_PLATE_DIAMETER / 2)
STRATEGIES = ("absolute", "object_relative", "intent")


def to_sim(v_table: np.ndarray) -> np.ndarray:
    return np.asarray(v_table) @ R_TS.T


def _yaw_from_closing_dir(d_table: np.ndarray) -> float:
    d = to_sim(d_table)
    return grasp_yaw_for_radial(d[:2])


def _keyframes(demo: dict, k0: int, k1: int, dt: float = 0.12) -> list[int]:
    """Indices between k0 and k1 (inclusive) roughly every dt seconds, only where the hand was seen."""
    t, ok = demo["t"], demo["hand_ok"]
    ks, last = [], -np.inf
    for k in range(k0, k1 + 1):
        if ok[k] and t[k] - last >= dt:
            ks.append(k)
            last = t[k]
    return ks


def _path_waypoints(points, yaw, gripper, tol=0.03, timeout=12, name="path"):
    return [Waypoint(np.asarray(p), yaw, gripper, tol=tol, timeout=timeout, name=name) for p in points]


def _grasp_and_place_tail(grasp, place, yaw, lift_points, release_points) -> list[Waypoint]:
    """The two moments that must be exact are gated tightly: closing at the grasp, opening at the place."""
    wps = [Waypoint(grasp, yaw, GRIPPER_OPEN, tol=0.006, timeout=60, name="descend"),
           Waypoint(grasp, yaw, GRIPPER_CLOSE, tol=0.01, hold=12, name="close")]
    wps += _path_waypoints(lift_points, yaw, GRIPPER_CLOSE, name="carry")
    wps += [Waypoint(place, yaw, GRIPPER_CLOSE, tol=0.008, timeout=60, name="lower"),
            Waypoint(place, yaw, GRIPPER_OPEN, tol=0.01, hold=8, name="release")]
    wps += _path_waypoints(release_points, yaw, GRIPPER_OPEN, tol=0.03, name="retreat")
    return wps


def human_bowl_rim_top(demo) -> float:
    return demo["bowl_start"][2] + HUMAN_BOWL_HEIGHT


# ------------------------------------------------------------------------------------------------
def plan_intent(demo: dict, s: Scene) -> list[Waypoint]:
    radial = to_sim(demo["grasp_point"] - demo["bowl_start"])
    place_off = PLATE_SCALE * to_sim(np.r_[demo["bowl_end"][:2] - demo["plate_center"][:2], 0.0])
    return plan_bowl_to_plate(s, rim_angle=float(np.arctan2(radial[1], radial[0])), place_offset=place_off)


def plan_object_relative(demo: dict, s: Scene) -> list[Waypoint]:
    P, ev = demo["pinch"], demo["events"]
    _, k_grasp, k_lift, k_down, k_release, k_end = ev
    b_h, b_s = demo["bowl_start"], s.bowl_pos
    rim_h, rim_s = human_bowl_rim_top(demo), s.bowl_pos[2] + BOWL_HEIGHT
    yaw = _yaw_from_closing_dir(demo["closing_dir"][k_grasp]) if np.isfinite(demo["closing_dir"][k_grasp, 0]) \
        else grasp_yaw_for_radial(to_sim(demo["grasp_point"] - b_h)[:2])

    def in_bowl_frame(p):        # human point -> sim point, anchored on the bowl (heights from the rim)
        d = to_sim(p - b_h)
        return np.array([b_s[0] + d[0], b_s[1] + d[1], rim_s + (p[2] - rim_h) + PINCH_TO_GRIP_Z])

    # grasp: the measured rim point, radius scaled to the sim bowl
    g_rel = to_sim(demo["grasp_point"] - b_h)
    g_rel[:2] *= BOWL_SCALE
    grasp = np.array([b_s[0] + g_rel[0], b_s[1] + g_rel[1], rim_s + (demo["grasp_point"][2] - rim_h) + PINCH_TO_GRIP_Z])

    # approach: the last part of your reach (within 15 cm of the bowl), bent so it ends exactly at `grasp`
    ks = [k for k in _keyframes(demo, 0, k_grasp) if np.linalg.norm(P[k, :2] - b_h[:2]) < 0.15]
    approach = [in_bowl_frame(P[k]) for k in ks]
    if approach:
        err = grasp - in_bowl_frame(P[k_grasp]) if np.isfinite(P[k_grasp, 0]) else grasp - approach[-1]
        n = len(approach)
        approach = [a + err * (j + 1) / n for j, a in enumerate(approach)]
        approach = [np.maximum(a, [-np.inf, -np.inf, rim_s + 0.01]) for a in approach]   # stay above the rim until the final descent
    pre = grasp + np.array([0, 0, 0.08])
    wps = [Waypoint(pre, yaw, GRIPPER_OPEN, tol=0.02, name="pre-grasp")] if not approach else \
          [Waypoint(approach[0] + np.array([0, 0, 0.05]), yaw, GRIPPER_OPEN, tol=0.02, name="pre-approach")]
    wps += _path_waypoints(approach, yaw, GRIPPER_OPEN, name="approach")

    # carry: endpoints re-anchored on the bowl (start) and the plate (end); your path's deviation from
    # its own straight line (mainly the lift) is added back on top
    place_off = PLATE_SCALE * to_sim(np.r_[demo["bowl_end"][:2] - demo["plate_center"][:2], 0.0])
    held = grasp - b_s
    place = s.plate_pos + np.array([place_off[0], place_off[1], PLATE_HEIGHT + 0.01]) + held
    kc = [k for k in _keyframes(demo, k_grasp, k_down) if k not in (k_grasp, k_down)]
    p0, p1 = P[k_grasp], P[k_down] if np.isfinite(P[k_down, 0]) else P[kc[-1]] if kc else P[k_grasp]
    carry = []
    for k in kc:
        u = (demo["t"][k] - demo["t"][k_grasp]) / max(demo["t"][k_down] - demo["t"][k_grasp], 1e-6)
        shape = to_sim(P[k] - (p0 + u * (p1 - p0)))
        carry.append(grasp + u * (place - grasp) + shape)
    # release + retreat: relative to where you let go
    kr = [k for k in _keyframes(demo, k_release, min(k_end, k_release + int(1.0 / np.median(np.diff(demo["t"])))))
          if k != k_release]
    ref = P[k_release] if np.isfinite(P[k_release, 0]) else P[k_down]
    retreat = [place + to_sim(P[k] - ref) for k in kr][:4] or [place + np.array([0, 0, 0.1])]
    return wps + _grasp_and_place_tail(grasp, place, yaw, carry, retreat)


def absolute_alignment(demos: list[dict], sim_bowl_mean: np.ndarray) -> np.ndarray:
    """ONE translation for all demos: map your average bowl start onto the sim's average bowl start.
    (Rotation is the fixed seat->robot rotation; the table top maps to the sim table top.)"""
    mean_h = np.mean([d["bowl_start"] for d in demos], 0)
    t = sim_bowl_mean - to_sim(mean_h)
    t[2] = TABLE_Z_SIM
    return t


def plan_absolute(demo: dict, s: Scene, offset: np.ndarray) -> list[Waypoint]:
    P, ev = demo["pinch"], demo["events"]
    _, k_grasp, k_lift, k_down, k_release, k_end = ev
    m = lambda p: to_sim(p) + offset + np.array([0, 0, PINCH_TO_GRIP_Z])
    yaw = _yaw_from_closing_dir(demo["closing_dir"][k_grasp]) if np.isfinite(demo["closing_dir"][k_grasp, 0]) \
        else grasp_yaw_for_radial(to_sim(demo["grasp_point"] - demo["bowl_start"])[:2])
    grasp = m(demo["grasp_point"])
    place = m(P[k_down]) if np.isfinite(P[k_down, 0]) else m(demo["bowl_end"] + demo["grasp_point"] - demo["bowl_start"])
    approach = [m(P[k]) for k in _keyframes(demo, 0, k_grasp) if np.linalg.norm(P[k, :2] - demo["bowl_start"][:2]) < 0.15]
    carry = [m(P[k]) for k in _keyframes(demo, k_grasp, k_down) if k not in (k_grasp, k_down)]
    retreat = [m(P[k]) for k in _keyframes(demo, k_release, k_end) if k != k_release][:4] or [place + [0, 0, 0.1]]
    first = approach[0] if approach else grasp
    wps = [Waypoint(np.maximum(first + [0, 0, 0.05], [-np.inf, -np.inf, grasp[2] + 0.08]), yaw, GRIPPER_OPEN, tol=0.02,
                    name="pre-approach")]
    wps += _path_waypoints(approach, yaw, GRIPPER_OPEN, name="approach")
    return wps + _grasp_and_place_tail(grasp, place, yaw, carry, retreat)


def make_plan(strategy: str, demo: dict, s: Scene, abs_offset: np.ndarray | None = None) -> list[Waypoint]:
    if strategy == "intent":
        return plan_intent(demo, s)
    if strategy == "object_relative":
        return plan_object_relative(demo, s)
    if strategy == "absolute":
        return plan_absolute(demo, s, abs_offset)
    raise ValueError(strategy)
