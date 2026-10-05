"""ArUco tag detection and the TABLE frame.

Frames used throughout the project (all right-handed, metres):
  cam   : OpenCV camera frame - x right, y down, z forward along the optical axis.
  table : defined by the big TABLE tag lying flat on the table. Origin at the tag centre,
          x/y along the tag's edges, z pointing UP out of the table (towards the camera).
  sim   : LIBERO world frame (see p2p/retarget). We only ever map table -> sim with a planar
          rigid transform plus a height offset, because both are z-up with the table at z = const.

solvePnP gives the pose of a tag in the camera frame, T_cam_tag (maps tag points into the
camera). We invert it once to get T_table_cam, and every 3D point measured in the camera frame
(hands, objects) is then expressed in the table frame: p_table = T_table_cam @ p_cam.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DICT = cv2.aruco.DICT_4X4_50

# id -> (role, black-square side in metres). The side length is what PnP needs to recover
# metric scale, so if your printed square measures e.g. 149 mm, put 0.149 here.
TAG_SPECS: dict[int, tuple[str, float]] = {
    0: ("table", 0.150),
    1: ("bowl", 0.040),
    2: ("plate", 0.050),
    3: ("bowl", 0.030),
}


@dataclass
class TagPose:
    tag_id: int
    role: str
    corners: np.ndarray        # (4, 2) pixel corners: TL, TR, BR, BL in the tag's own orientation
    T_cam_tag: np.ndarray      # 4x4
    reproj_px: float           # mean reprojection error of the 4 corners
    side_px: float             # apparent size in the image

    @property
    def center_cam(self) -> np.ndarray:
        return self.T_cam_tag[:3, 3]


def _make_detector() -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    # Sub-pixel corner refinement noticeably improves PnP for small tags.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), params)


_DETECTOR = None


def _tag_object_points(side: float) -> np.ndarray:
    """Tag corners in the tag frame, in the order ArUco returns them and IPPE_SQUARE expects."""
    h = side / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.ravel(t)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return to_T(R.T, -R.T @ t)


def transform(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.atleast_2d(p)
    return p @ T[:3, :3].T + T[:3, 3]


def detect_tags(bgr: np.ndarray, K: np.ndarray, dist: np.ndarray | None = None) -> dict[int, TagPose]:
    """Detect known tags and estimate each one's 6-DoF pose with PnP.

    IPPE_SQUARE is the PnP solver made for exactly this case (4 coplanar points of a square).
    A single small square has a known pose ambiguity (two mirror-like solutions); OpenCV returns
    the one with lower reprojection error, which is reliable for the big table tag and good enough
    for the *position* (not orientation) of the small object tags - and position is all we use.
    """
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = _make_detector()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    corners, ids, _ = _DETECTOR.detectMarkers(gray)
    out: dict[int, TagPose] = {}
    if ids is None:
        return out
    dist = np.zeros(5) if dist is None else dist
    for c, i in zip(corners, ids.ravel()):
        i = int(i)
        if i not in TAG_SPECS:
            continue
        role, side = TAG_SPECS[i]
        img_pts = c.reshape(4, 2).astype(np.float64)
        obj_pts = _tag_object_points(side)
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        err = float(np.linalg.norm(proj.reshape(4, 2) - img_pts, axis=1).mean())
        side_px = float(np.mean(np.linalg.norm(img_pts - np.roll(img_pts, 1, axis=0), axis=1)))
        pose = TagPose(i, role, img_pts, to_T(cv2.Rodrigues(rvec)[0], tvec), err, side_px)
        # keep the biggest instance if the same id is seen twice (e.g. a spare left in view)
        if i not in out or side_px > out[i].side_px:
            out[i] = pose
    return out


def table_frame(tags: dict[int, TagPose]) -> np.ndarray | None:
    """T_table_cam (4x4) from the TABLE tag, or None if it is not visible."""
    t = tags.get(0)
    return None if t is None else inv_T(t.T_cam_tag)


def camera_geometry(T_table_cam: np.ndarray) -> dict[str, float]:
    """Where the phone is relative to the table: height and how steeply it looks down."""
    T_cam_table = inv_T(T_table_cam)
    cam_center = T_table_cam[:3, 3]                          # camera origin in table frame
    optical_axis = T_table_cam[:3, :3] @ np.array([0, 0, 1.0])  # cam z-axis in table frame
    tilt_from_vertical = np.degrees(np.arccos(np.clip(-optical_axis[2], -1, 1)))
    return {
        "height_above_table_m": float(cam_center[2]),
        "tilt_from_vertical_deg": float(tilt_from_vertical),   # 0 = looking straight down
        "dist_to_table_tag_m": float(np.linalg.norm(T_cam_table[:3, 3])),
    }
