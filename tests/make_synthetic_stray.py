"""Write a fake Stray Scanner recording with known geometry, to test the human-data pipeline.

The scene: table tag at the table-frame origin, plate tag lying on the table, bowl tag that
is lifted 10 cm and carried 15 cm in +x halfway through the clip. Ground truth is saved to
gt.npz so tests can compare recovered positions against it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2p.human.markers import DICT, TAG_SPECS  # noqa: E402

W, H, WD, HD = 1920, 1440, 256, 192
K = np.array([[1450.0, 0, 960], [0, 1450.0, 720], [0, 0, 1]])


def look_at(eye, target, up=(0, 0, 1)):
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)  # R_world_cam (columns = camera axes in world)


def tag_image(tag_id, px=600):
    d = cv2.aruco.getPredefinedDictionary(DICT)
    m = cv2.aruco.generateImageMarker(d, tag_id, px)
    return cv2.copyMakeBorder(m, px // 6, px // 6, px // 6, px // 6, cv2.BORDER_CONSTANT, value=255)


def project(p_world, R, eye, Kmat):
    pc = (np.atleast_2d(p_world) - eye) @ R
    uv = (pc @ Kmat.T)
    return uv[:, :2] / uv[:, 2:3], pc[:, 2]


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def mat2quat_xyzw(R):
    w = np.sqrt(max(0.0, 1 + np.trace(R))) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([x, y, z, w])


def main(out: Path, n_frames=90, fps=30, moving: bool = False):
    """Egocentric layout in the table frame (x right, y away from you): table tag at the origin,
    plate 21 cm towards you, bowl 38 cm towards you, eye ~0.5 m above the table behind the bowl.
    moving=True adds handheld-style camera drift (up to ~6 cm, a few degrees) and writes an
    ARKit-like odometry.csv in an arbitrary world frame."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(exist_ok=True)
    (out / "confidence").mkdir(exist_ok=True)
    np.savetxt(out / "camera_matrix.csv", K, delimiter=",")
    Kd = np.diag([WD / W, HD / H, 1]) @ K

    plate_c = np.array([0.0, -0.21, 0.015])
    bowl_start = np.array([0.02, -0.38, 0.010])

    def bowl_c(f):
        s = np.clip((f - 30) / 30, 0, 1)  # carried between frames 30 and 60
        lift = 0.10 * np.sin(np.pi * s)
        p = bowl_start + s * (plate_c - bowl_start)
        p[2] = bowl_start[2] + s * (plate_c[2] + 0.01 - bowl_start[2]) + lift
        return p

    def camera(f):
        a = f / n_frames * 2 * np.pi
        drift = np.array([0.04 * np.sin(a), 0.03 * np.sin(2 * a), 0.03 * np.cos(a) - 0.03]) if moving else np.zeros(3)
        eye = np.array([0.05, -0.75, 0.50]) + drift
        target = np.array([0.0, -0.25, 0.0]) + (np.array([0.02 * np.sin(a), 0, 0]) if moving else 0)
        return eye, look_at(eye, target)

    # an arbitrary "ARKit" world frame: odometry is stored there, not in the table frame
    R_w, t_w = rot_z(0.7) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]]), np.array([0.3, -1.2, 0.4])
    vw = cv2.VideoWriter(str(out / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    gt_bowl, gt_eye, odo = [], [], []
    uu, vv = np.meshgrid(np.arange(WD), np.arange(HD))
    rays_cam = np.stack([(uu - Kd[0, 2]) / Kd[0, 0], (vv - Kd[1, 2]) / Kd[1, 1], np.ones_like(uu, float)], -1)
    for f in range(n_frames):
        eye, R = camera(f)
        rays_w = rays_cam @ R.T
        img = np.full((H, W, 3), 170, np.uint8)
        tags = [(0, np.array([0.0, 0.0, 0.0])), (2, plate_c), (1, bowl_c(f))]
        gt_bowl.append(bowl_c(f))
        gt_eye.append(eye)
        odo.append([f / fps, f, *(R_w @ eye + t_w), *mat2quat_xyzw(R_w @ R)])
        depth = (-eye[2] / rays_w[..., 2]).astype(np.float64)  # ray param t equals z-depth since ray z_cam = 1
        for tid, c in tags:
            s = TAG_SPECS[tid][1] * 8 / 6  # include the white border drawn around the marker
            corners_w = c + np.array([[-s / 2, s / 2, 0], [s / 2, s / 2, 0], [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]])
            uv, _ = project(corners_w, R, eye, K)
            src = tag_image(tid)
            M = cv2.getPerspectiveTransform(np.float32([[0, 0], [src.shape[1], 0], [src.shape[1], src.shape[0]], [0, src.shape[0]]]),
                                            uv.astype(np.float32))
            warped = cv2.warpPerspective(cv2.cvtColor(src, cv2.COLOR_GRAY2BGR), M, (W, H), borderValue=(0, 0, 0))
            mask = cv2.warpPerspective(np.full(src.shape, 255, np.uint8), M, (W, H)) > 0
            img[mask] = warped[mask]
            uvd, _ = project(corners_w, R, eye, Kd)
            m = np.zeros((HD, WD), np.uint8)
            cv2.fillConvexPoly(m, np.round(uvd).astype(np.int32), 1)
            t_plane = (c[2] - eye[2]) / rays_w[..., 2]
            depth[m > 0] = t_plane[m > 0]
        vw.write(img)
        cv2.imwrite(str(out / "depth" / f"{f:06d}.png"), np.round(depth * 1000).astype(np.uint16))
        cv2.imwrite(str(out / "confidence" / f"{f:06d}.png"), np.full((HD, WD), 2, np.uint8))
    vw.release()
    with open(out / "odometry.csv", "w") as fh:
        fh.write("timestamp, frame, x, y, z, qx, qy, qz, qw\n")
        for r in odo:
            fh.write(", ".join(f"{v:.6f}" if i != 1 else str(int(v)) for i, v in enumerate(r)) + "\n")
    np.savez(out / "gt.npz", eye=np.array(gt_eye), plate=plate_c, bowl=np.array(gt_bowl))
    print("wrote", out)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/synthetic_stray/demo_01"), moving="--moving" in sys.argv)
