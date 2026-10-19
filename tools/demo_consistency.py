"""How consistent are the demonstrations in a LeRobot dataset? (official teleop vs ours)

    python tools/demo_consistency.py /content/hub/lerobot/official /content/hub/lerobot/ours_all

Reads only the numeric frame table (data/*.parquet: 8-D state, 7-D action), no videos. Per episode:
  * length (steps at 20 Hz),
  * the moment the gripper is first commanded to close (action[6] > 0) = the grasp,
  * the wrist yaw at that moment: rotation of the gripper about the vertical, relative to the straight-down
    reset pose (state[3:6] is the gripper orientation as an axis-angle),
  * the gripper height at that moment, and the time from the start to the grasp.
An imitation policy has to reproduce the grasp from images. If the demonstrations grasp with very different
wrist angles, the policy sees conflicting targets for similar-looking scenes and may average between them,
which for a rim pinch means a grasp that matches none of the demos.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2p.sim.controller import R_DOWN  # noqa: E402


def axisangle_to_R(v: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def wrist_yaw_deg(state: np.ndarray) -> float:
    """Yaw psi with R = rot_z(psi) @ R_DOWN; wrapped to [-90, 90) because the two fingers are symmetric."""
    M = axisangle_to_R(state[3:6]) @ R_DOWN.T
    psi = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    return float((psi + 90) % 180 - 90)


def episodes(root: Path):
    import pandas as pd

    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    df = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state", "action"])
                   for f in files)
    for e, g in df.sort_values(["episode_index", "frame_index"]).groupby("episode_index"):
        yield int(e), np.stack(g["observation.state"].to_numpy()), np.stack(g["action"].to_numpy())


def summarize(root: Path) -> dict:
    rows = []
    for e, S, A in episodes(root):
        close = np.flatnonzero(A[:, 6] > 0)
        if len(close) == 0:
            continue
        k = int(close[0])
        rows.append({"episode": e, "length": len(A), "grasp_step": k, "yaw_deg": wrist_yaw_deg(S[k]),
                     "grasp_z": float(S[k, 2])})
    L, G = np.array([r["length"] for r in rows]), np.array([r["grasp_step"] for r in rows])
    Y, Z = np.array([r["yaw_deg"] for r in rows]), np.array([r["grasp_z"] for r in rows])
    out = {"dataset": root.name, "episodes": len(rows),
           "length_median": float(np.median(L)), "steps_to_grasp_median": float(np.median(G)),
           "grasp_yaw_deg_p10_p50_p90": np.percentile(Y, [10, 50, 90]).round(1).tolist(),
           "grasp_yaw_deg_std": round(float(Y.std()), 1),
           "grasp_height_cm_std": round(float(100 * Z.std()), 2)}
    return out


if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for d in sys.argv[1:]:
        print(json.dumps(summarize(Path(d))))
