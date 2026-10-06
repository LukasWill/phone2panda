"""Loader for recordings made with the Stray Scanner iOS app (iPhone Pro, LiDAR).

A recording folder looks like:
    rgb.mp4              full-res colour video, 1920x1440 on current iPhones
    depth/000000.png     LiDAR depth, uint16 millimetres, 256x192 (older app versions: .npy)
    confidence/000000.png  ARKit depth confidence per pixel: 0 = low, 1 = medium, 2 = high
    camera_matrix.csv    3x3 pinhole intrinsics K for the *RGB* resolution
    odometry.csv         ARKit camera poses (unused: our phone sits on a tripod and the
                         table tag gives a better, shared frame across recordings)

The one subtle point: K is given for the RGB image, but depth has a lower resolution.
A pinhole camera resized by factors (sx, sy) has K' = diag(sx, sy, 1) @ K, so we scale
fx, cx by sx and fy, cy by sy before using K with depth pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def scale_intrinsics(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    S = np.diag([sx, sy, 1.0])
    return S @ K


@dataclass
class StrayRecording:
    root: Path
    K_rgb: np.ndarray = field(init=False)
    rgb_size: tuple[int, int] = field(init=False)      # (W, H)
    depth_size: tuple[int, int] = field(init=False)    # (W, H)
    fps: float = field(init=False)
    n_rgb: int = field(init=False)
    depth_paths: list[Path] = field(init=False)
    conf_paths: list[Path] = field(init=False)

    def __post_init__(self):
        self.root = Path(self.root)
        if not (self.root / "rgb.mp4").exists():
            raise FileNotFoundError(f"{self.root} has no rgb.mp4 - is this a Stray Scanner export?")
        self.K_rgb = np.loadtxt(self.root / "camera_matrix.csv", delimiter=",").reshape(3, 3)
        cap = cv2.VideoCapture(str(self.root / "rgb.mp4"))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        self.n_rgb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.rgb_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        ok, _ = cap.read()
        cap.release()
        if not ok:
            # The app writes HEVC (H.265). Most OpenCV builds decode it; if yours does not:
            raise RuntimeError(
                f"OpenCV cannot decode {self.root / 'rgb.mp4'} (HEVC). Convert once with\n"
                f"  ffmpeg -i rgb.mp4 -c:v libx264 -crf 12 rgb_h264.mp4 && mv rgb_h264.mp4 rgb.mp4")
        self.depth_paths = sorted(p for p in (self.root / "depth").iterdir() if p.suffix in (".png", ".npy"))
        conf_dir = self.root / "confidence"
        self.conf_paths = sorted(conf_dir.glob("*.png")) if conf_dir.exists() else []
        d0 = self.depth(0)
        self.depth_size = (d0.shape[1], d0.shape[0])

    # ---- frames -------------------------------------------------------------------------
    def rgb_frames(self, step: int = 1, start: int = 0) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (frame_index, BGR image). Decodes sequentially, which is much faster than seeking."""
        cap = cv2.VideoCapture(str(self.root / "rgb.mp4"))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i >= start and (i - start) % step == 0:
                yield i, frame
            i += 1
        cap.release()

    def rgb(self, index: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(self.root / "rgb.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise IndexError(index)
        return frame

    def depth(self, index: int) -> np.ndarray:
        """Depth in metres, float32, shape (H_d, W_d). 0 means 'no measurement'."""
        p = self.depth_paths[min(index, len(self.depth_paths) - 1)]
        d = np.load(p) if p.suffix == ".npy" else cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        return d.astype(np.float32) / 1000.0

    def confidence(self, index: int) -> np.ndarray | None:
        if not self.conf_paths:
            return None
        return cv2.imread(str(self.conf_paths[min(index, len(self.conf_paths) - 1)]), cv2.IMREAD_UNCHANGED)

    # ---- camera motion (ARKit visual-inertial odometry) -------------------------------------
    def odometry(self) -> dict[str, np.ndarray] | None:
        """Per-frame ARKit camera pose as stored by the app: position (N,3) in metres and
        quaternion (N,4, x y z w), in ARKit's session frame (gravity-aligned, origin = where the
        recording started).

        This is what makes a HANDHELD phone usable: every frame has its own camera pose, so hand
        points can be moved into one fixed frame even though the camera moves. Distances between
        camera positions do not depend on axis conventions, which lets us validate ARKit against
        the table tag without having to trust ARKit's camera-axis convention (see check_recording).
        """
        path = self.root / "odometry.csv"
        if not path.exists():
            return None
        import csv

        with open(path) as f:
            rows = list(csv.reader(f))
        header = [h.strip() for h in rows[0]]
        data = np.array([[float(v) for v in r] for r in rows[1:] if r], dtype=np.float64)
        col = {h: i for i, h in enumerate(header)}
        return {
            "frame": data[:, col["frame"]].astype(int) if "frame" in col else np.arange(len(data)),
            "position": data[:, [col["x"], col["y"], col["z"]]],
            "quat_xyzw": data[:, [col["qx"], col["qy"], col["qz"], col["qw"]]],
        }

    # ---- geometry -----------------------------------------------------------------------
    @property
    def K_depth(self) -> np.ndarray:
        sx = self.depth_size[0] / self.rgb_size[0]
        sy = self.depth_size[1] / self.rgb_size[1]
        return scale_intrinsics(self.K_rgb, sx, sy)

    def depth_at_rgb_pixel(self, index: int, uv_rgb: np.ndarray, win: int = 2, min_conf: int = 1) -> np.ndarray:
        """Robust depth (metres) at RGB pixel(s) uv_rgb (N,2): median over a (2*win+1)^2 depth patch,
        ignoring zero / low-confidence pixels. Returns NaN where nothing valid is found.

        Why a median patch: LiDAR depth is low-res and noisy at object edges (fingers!), and a
        single pixel can land on the background behind a thin finger.
        """
        uv_rgb = np.atleast_2d(uv_rgb).astype(np.float64)
        d = self.depth(index)
        conf = self.confidence(index)
        sx = self.depth_size[0] / self.rgb_size[0]
        sy = self.depth_size[1] / self.rgb_size[1]
        out = np.full(len(uv_rgb), np.nan)
        H, W = d.shape
        for k, (u, v) in enumerate(uv_rgb):
            # pixel centres: continuous coord u maps to (u + 0.5) * s - 0.5 in the smaller image
            ud, vd = (u + 0.5) * sx - 0.5, (v + 0.5) * sy - 0.5
            x0, x1 = int(round(ud)) - win, int(round(ud)) + win + 1
            y0, y1 = int(round(vd)) - win, int(round(vd)) + win + 1
            if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
                continue
            patch = d[max(y0, 0):y1, max(x0, 0):x1]
            ok = patch > 0
            if conf is not None:
                ok &= conf[max(y0, 0):y1, max(x0, 0):x1] >= min_conf
            if ok.any():
                out[k] = float(np.median(patch[ok]))
        return out


def backproject(uv: np.ndarray, z: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixel (u, v) with z-depth z -> 3D point in the camera frame (x right, y down, z forward).
    This is the pinhole model inverted:  x = (u - cx) * z / fx,  y = (v - cy) * z / fy.
    """
    uv = np.atleast_2d(uv)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    x = (uv[:, 0] - K[0, 2]) * z / K[0, 0]
    y = (uv[:, 1] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1)
