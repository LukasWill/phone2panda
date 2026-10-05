"""Recover known object positions from a synthetic Stray Scanner recording.
    python -m pytest -q tests/test_human_markers.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2p.human.markers import detect_tags, table_frame, transform  # noqa: E402
from p2p.human.stray import StrayRecording, backproject  # noqa: E402
from tests.make_synthetic_stray import main as make_recording  # noqa: E402


def test_tags_and_depth_recover_scene(tmp_path):
    root = tmp_path / "demo"
    make_recording(root, n_frames=12, fps=30)
    gt = np.load(root / "gt.npz")
    rec = StrayRecording(root)
    frame0 = rec.rgb(0)
    tags = detect_tags(frame0, rec.K_rgb)
    T = table_frame(tags)
    assert T is not None
    # camera centre in the table frame vs ground truth (which uses the same frame)
    assert np.linalg.norm(T[:3, 3] - gt["eye"]) < 0.01
    plate = transform(T, tags[2].center_cam)[0]
    bowl = transform(T, tags[1].center_cam)[0]
    assert np.linalg.norm(plate[:2] - gt["plate"][:2]) < 0.012
    assert np.linalg.norm(bowl[:2] - gt["bowl"][0][:2]) < 0.012
    # LiDAR path: back-project the plate tag centre with measured depth -> same point
    uv = (rec.K_rgb @ tags[2].center_cam)[:2] / tags[2].center_cam[2]
    z = rec.depth_at_rgb_pixel(0, uv[None])
    p_cam = backproject(uv[None], z, rec.K_rgb)[0]
    assert np.linalg.norm(transform(T, p_cam)[0] - plate) < 0.015
