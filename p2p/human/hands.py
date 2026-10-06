"""Hand landmarks from MediaPipe, lifted to metric 3D with the iPhone's LiDAR.

MediaPipe gives, per detected hand,
  - 21 image landmarks (x, y normalised to the image), and
  - 21 "world" landmarks: metric 3D (metres) relative to the hand's own centre. The shape is
    learned from data, so the hand SIZE is approximate, and the absolute DEPTH is unknown.
LiDAR gives absolute depth, but only at 256x192 and only for surfaces the sensor sees.

Seen from above (egocentric), the fingertips are often hidden under the back of the hand, so a
LiDAR read at a fingertip pixel returns the back of a finger instead. We therefore combine them:
  1. palm depth  = robust median LiDAR depth at the 5 palm landmarks (wrist + 4 knuckles), which
                   stay visible from above;
  2. depth of any landmark j = palm depth + (z_j - z_palm) from the world landmarks
                   (their axes are aligned with the camera, so z differences are depth differences);
  3. 3D point    = back-project landmark j's PIXEL with that depth (the 2D landmark is the most
                   reliable quantity MediaPipe outputs).
The pinch point (the robot's "grip site" analogue) is the midpoint of thumb tip and index tip;
the aperture (open/closed) is their 3D distance, taken from the world landmarks alone.

Two MediaPipe APIs are supported: the current Tasks API (needs a .task model file, downloaded
once) and the older `solutions` API (models bundled in mediapipe <= 0.10.x).
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from p2p.human.stray import backproject

WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 0, 4, 5, 8, 9, 13, 17
PALM = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
             "float16/latest/hand_landmarker.task")


@dataclass
class Hand2D:
    uv: np.ndarray        # (21, 2) landmark pixels in the FULL-resolution frame
    world: np.ndarray     # (21, 3) metric landmarks, camera-aligned axes, origin near the hand centre
    label: str            # MediaPipe's "Left"/"Right". It assumes a MIRRORED (selfie) image, so for the
                          # phone's rear camera the label is swapped: your right hand is reported "Left".
    score: float

    @property
    def is_right_hand(self) -> bool:
        return self.label == "Left"


class HandDetector:
    """Video-mode hand detector. Call with frames in time order (it tracks between frames)."""

    def __init__(self, backend: str = "auto", model_path: str | None = None, max_hands: int = 2,
                 process_width: int = 960, min_conf: float = 0.5):
        import mediapipe as mp

        self.mp = mp
        self.process_width = process_width
        if backend == "auto":
            backend = "legacy" if hasattr(mp, "solutions") and model_path is None else "tasks"
        self.backend = backend
        if backend == "legacy":
            self._hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=max_hands,
                                                   model_complexity=1, min_detection_confidence=min_conf,
                                                   min_tracking_confidence=min_conf)
        else:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions

            path = model_path or self._ensure_model()
            opts = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(path)), running_mode=vision.RunningMode.VIDEO,
                num_hands=max_hands, min_hand_detection_confidence=min_conf,
                min_hand_presence_confidence=min_conf, min_tracking_confidence=min_conf)
            self._hands = vision.HandLandmarker.create_from_options(opts)

    @staticmethod
    def _ensure_model() -> Path:
        path = Path(os.environ.get("P2P_CACHE", Path.home() / ".cache" / "phone2panda")) / "hand_landmarker.task"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, path)
        return path

    def __call__(self, bgr: np.ndarray, t_ms: int) -> list[Hand2D]:
        H, W = bgr.shape[:2]
        scale = self.process_width / W
        small = cv2.resize(bgr, (self.process_width, int(round(H * scale))), interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(small[:, :, ::-1])
        out = []
        if self.backend == "legacy":
            res = self._hands.process(rgb)
            if not res.multi_hand_landmarks:
                return out
            for lm, wl, hd in zip(res.multi_hand_landmarks, res.multi_hand_world_landmarks, res.multi_handedness):
                c = hd.classification[0]
                out.append(Hand2D(np.array([[p.x * W, p.y * H] for p in lm.landmark]),
                                  np.array([[p.x, p.y, p.z] for p in wl.landmark]), c.label, float(c.score)))
        else:
            res = self._hands.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), int(t_ms))
            for lm, wl, hd in zip(res.hand_landmarks, res.hand_world_landmarks, res.handedness):
                out.append(Hand2D(np.array([[p.x * W, p.y * H] for p in lm]),
                                  np.array([[p.x, p.y, p.z] for p in wl]), hd[0].category_name, float(hd[0].score)))
        return out

    def close(self):
        self._hands.close()


def pick_hand(hands: list[Hand2D], prev_uv: np.ndarray | None) -> Hand2D | None:
    """The demonstrating hand: prefer the right hand, then continuity with the previous frame."""
    if not hands:
        return None
    if prev_uv is not None:
        return min(hands, key=lambda h: np.linalg.norm(h.uv[WRIST] - prev_uv) - 200 * h.is_right_hand)
    return max(hands, key=lambda h: (h.is_right_hand, h.score))


@dataclass
class Hand3D:
    points: np.ndarray    # (21, 3) camera frame
    pinch: np.ndarray     # (3,) midpoint of thumb tip and index tip, camera frame
    palm: np.ndarray      # (3,) mean of the palm landmarks, camera frame
    aperture: float       # thumb-index tip distance (m)
    palm_depth_px: int    # how many palm landmarks had valid LiDAR (quality flag)


def lift_hand(hand: Hand2D, depth_at_px, K: np.ndarray) -> Hand3D | None:
    """depth_at_px(uv (N,2)) -> z-depth (N,) with NaN where invalid (e.g. StrayRecording.depth_at_rgb_pixel)."""
    z_palm_pts = depth_at_px(hand.uv[PALM])
    ok = np.isfinite(z_palm_pts)
    if ok.sum() < 2:
        return None
    w = hand.world
    # Each palm landmark gives one estimate of the palm-centre depth; take their median.
    z_palm = float(np.median(z_palm_pts[ok] - (w[PALM][ok, 2] - w[PALM, 2].mean())))
    z = z_palm + (w[:, 2] - w[PALM, 2].mean())
    pts = backproject(hand.uv, z, K)
    uv_pinch = (hand.uv[THUMB_TIP] + hand.uv[INDEX_TIP]) / 2
    z_pinch = (z[THUMB_TIP] + z[INDEX_TIP]) / 2
    return Hand3D(points=pts, pinch=backproject(uv_pinch[None], [z_pinch], K)[0], palm=pts[PALM].mean(0),
                  aperture=float(np.linalg.norm(w[THUMB_TIP] - w[INDEX_TIP])), palm_depth_px=int(ok.sum()))
