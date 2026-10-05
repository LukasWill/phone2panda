"""Hand-coded pick-and-place from privileged object poses.

This is the Day-1 sanity check and, later, the backbone of every retargeting strategy:
a trajectory is a list of Waypoints (gripper pose + gripper command), and WaypointFollower
turns it into actions with the PoseController. Retargeting only changes WHERE the waypoints
come from (your hand, your hand relative to the objects, or just your grasp/place intent).

Rim grasp geometry (top view):
                 plate
                   o            the fingers straddle the bowl wall, so the closing axis
        bowl     /              must be RADIAL: pointing from the bowl centre to the
       (  c --g  )              grasp point g. `rim_angle` picks where on the rim we grasp.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .controller import GRIPPER_CLOSE, GRIPPER_OPEN, R_DOWN, PoseController, grasp_yaw_for_radial, rot_z
from .env import BOWL_HEIGHT, BOWL_RADIUS, PLATE_HEIGHT, Scene


@dataclass
class Waypoint:
    pos: np.ndarray
    yaw: float
    gripper: float
    tol: float = 0.01          # metres: when the gripper is this close, the waypoint counts as reached
    hold: int = 0              # extra steps to stay once reached (e.g. let the fingers close)
    timeout: int = 80          # give up waiting and move on after this many steps
    name: str = ""


@dataclass
class WaypointFollower:
    waypoints: list[Waypoint]
    controller: PoseController = field(default_factory=PoseController)
    i: int = 0
    _steps_here: int = 0
    _held: int = 0

    @property
    def done(self) -> bool:
        return self.i >= len(self.waypoints)

    @property
    def current(self) -> str:
        return self.waypoints[min(self.i, len(self.waypoints) - 1)].name

    def act(self, s: Scene) -> np.ndarray:
        wp = self.waypoints[min(self.i, len(self.waypoints) - 1)]
        a = self.controller(s.eef_pos, s.eef_R, wp.pos, rot_z(wp.yaw) @ R_DOWN, wp.gripper)
        if self.done:
            return a
        self._steps_here += 1
        reached = np.linalg.norm(s.eef_pos - wp.pos) < wp.tol
        if reached or self._steps_here > wp.timeout:
            self._held += 1
            if self._held > wp.hold:
                self.i += 1
                self._steps_here = self._held = 0
        return a


def plan_bowl_to_plate(s: Scene, rim_angle: float = np.pi / 2, grasp_depth: float = 0.025,
                       lift: float = 0.12, place_clearance: float = 0.01,
                       grasp_offset: np.ndarray | None = None,
                       place_offset: np.ndarray | None = None) -> list[Waypoint]:
    """Waypoints for: pre-grasp above the rim -> descend -> close -> lift -> carry -> lower -> open -> retreat.

    rim_angle: where on the rim to grasp, angle (rad, world frame) of the bowl-centre -> grasp
        direction. pi/2 = the robot's left side of the bowl, which needs no wrist rotation.
    grasp_offset / place_offset: extra xyz shifts (metres) used later to perturb plans for data
        generation and to inject the human's grasp / place choice.
    """
    radial = np.array([np.cos(rim_angle), np.sin(rim_angle), 0.0])
    yaw = grasp_yaw_for_radial(radial[:2])
    rim_top = s.bowl_pos[2] + BOWL_HEIGHT
    grasp = s.bowl_pos + radial * (BOWL_RADIUS - 0.006)
    grasp[2] = rim_top - grasp_depth
    if grasp_offset is not None:
        grasp = grasp + grasp_offset
    held = grasp - s.bowl_pos                      # gripper position relative to the bowl, kept while carrying
    place_bowl = s.plate_pos + np.array([0, 0, PLATE_HEIGHT + place_clearance])
    if place_offset is not None:
        place_bowl = place_bowl + place_offset
    place = place_bowl + held
    up = np.array([0, 0, lift])
    return [
        Waypoint(grasp + np.array([0, 0, 0.08]), yaw, GRIPPER_OPEN, tol=0.01, name="pre-grasp"),
        Waypoint(grasp, yaw, GRIPPER_OPEN, tol=0.006, name="descend"),
        Waypoint(grasp, yaw, GRIPPER_CLOSE, tol=0.01, hold=12, name="close"),
        Waypoint(grasp + up, yaw, GRIPPER_CLOSE, tol=0.015, name="lift"),
        Waypoint(place + np.array([0, 0, up[2] * 0.6]), yaw, GRIPPER_CLOSE, tol=0.012, name="carry"),
        Waypoint(place, yaw, GRIPPER_CLOSE, tol=0.008, name="lower"),
        Waypoint(place, yaw, GRIPPER_OPEN, tol=0.01, hold=8, name="release"),
        Waypoint(place + np.array([0, 0, 0.10]), yaw, GRIPPER_OPEN, tol=0.02, name="retreat"),
    ]


def run_episode(env, follower: WaypointFollower, record=None) -> dict:
    """Roll a follower out until success or the step limit. `record(env, action)` is called
    before every step (used to log datasets)."""
    success, done = False, False
    while not done:
        a = follower.act(env.scene())
        if record is not None:
            record(env, a)
        _, success, done = env.step(a)
    return {"success": success, "steps": env.t, "last_waypoint": follower.current}
