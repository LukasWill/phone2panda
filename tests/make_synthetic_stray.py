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


def main(out: Path, n_frames=90, fps=30):
    out.mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(exist_ok=True)
    (out / "confidence").mkdir(exist_ok=True)
    eye = np.array([0.05, -0.45, 0.60])
    R = look_at(eye, [0.0, 0.18, 0.0])
    np.savetxt(out / "camera_matrix.csv", K, delimiter=",")
    Kd = np.diag([WD / W, HD / H, 1]) @ K

    plate_c, plate_h = np.array([0.15, 0.25, 0.015]), 0.015
    bowl_start = np.array([-0.10, 0.25, 0.010])

    def bowl_c(f):
        s = np.clip((f - 30) / 30, 0, 1)  # carried between frames 30 and 60
        lift = 0.10 * np.sin(np.pi * s)
        return bowl_start + np.array([0.15 * s, 0, lift])

    vw = cv2.VideoWriter(str(out / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    gt_bowl = []
    # depth of the table plane (z=0) for every depth pixel: intersect the pixel ray with z = 0
    uu, vv = np.meshgrid(np.arange(WD), np.arange(HD))
    rays_cam = np.stack([(uu - Kd[0, 2]) / Kd[0, 0], (vv - Kd[1, 2]) / Kd[1, 1], np.ones_like(uu, float)], -1)
    rays_w = rays_cam @ R.T
    for f in range(n_frames):
        img = np.full((H, W, 3), 170, np.uint8)
        depth_w_plane = {}
        tags = [(0, np.array([0.0, 0.0, 0.0])), (2, plate_c), (1, bowl_c(f))]
        gt_bowl.append(bowl_c(f))
        depth = (-eye[2] / rays_w[..., 2]).astype(np.float64)  # t along ray; z-depth = t since ray z_cam = 1
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
            # depth inside this tag's footprint = depth of the plane z = c_z
            uvd, _ = project(corners_w, R, eye, Kd)
            m = np.zeros((HD, WD), np.uint8)
            cv2.fillConvexPoly(m, np.round(uvd).astype(np.int32), 1)
            t_plane = (c[2] - eye[2]) / rays_w[..., 2]
            depth[m > 0] = t_plane[m > 0]
        vw.write(img)
        cv2.imwrite(str(out / "depth" / f"{f:06d}.png"), np.round(depth * 1000).astype(np.uint16))
        cv2.imwrite(str(out / "confidence" / f"{f:06d}.png"), np.full((HD, WD), 2, np.uint8))
    vw.release()
    np.savez(out / "gt.npz", eye=eye, R_world_cam=R, plate=plate_c, bowl=np.array(gt_bowl))
    print("wrote", out)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/synthetic_stray/demo_01"))
