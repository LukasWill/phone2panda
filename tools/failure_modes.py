"""What actually happened to the bowl in failed generated episodes (complements `last_waypoint`).

    python tools/failure_modes.py /content/generated [--all]

`last_waypoint` only says how far the waypoint follower got before the 300-step limit. Waypoints also
advance on a timeout, so a rollout whose grasp missed still walks through carry/lower/release/retreat
with an empty gripper and ends at "retreat". The bowl's own trajectory (low_dim[8:15]) tells them apart:

  never lifted        bowl never rose > 2 cm             -> grasp missed, or the arm never got there
  left off plate      lifted, but ends beyond the plate's radius (6.85 cm) from its centre
  on plate, > 3 cm    on the plate but more than 3 cm off-centre. LIBERO's On(bowl, plate)
                      (libero/envs/object_states/base_object_states.py, check_ontop) needs contact,
                      bowl above plate, AND body origins < 3 cm apart in xy: on the plate is not enough
  within 3 cm, not ok centred, yet not counted (no contact yet, or tilted on the rim)
plus the bowl's final tilt (a tipped bowl is a placement-quality failure).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2p.sim.env import PLATE_RADIUS, quat2mat  # noqa: E402

LIBERO_ON_XY_TOL = 0.03      # LIBERO's success needs the bowl within 3 cm (xy) of the plate's centre


def classify(z) -> dict:
    meta = json.loads(str(z["meta"]))
    low = z["low_dim"]
    bowl, quat, plate = low[:, 8:11], low[:, 11:15], low[0, 15:18]
    lift = float(bowl[:, 2].max() - bowl[0, 2])
    end_dist = float(np.linalg.norm(bowl[-1, :2] - plate[:2]))
    tilt = float(np.degrees(np.arccos(np.clip(quat2mat(quat[-1])[2, 2], -1, 1))))   # bowl axis vs vertical
    if lift < 0.02:
        kind = "never lifted"
    elif end_dist > PLATE_RADIUS:
        kind = "left off plate"
    elif end_dist >= LIBERO_ON_XY_TOL:
        kind = "on plate, > 3 cm"
    else:
        kind = "within 3 cm, not ok"
    return {"demo": meta["demo"], "seed": meta["seed"], "success": bool(z["success"]), "last": meta["last_waypoint"],
            "kind": kind, "lift_cm": round(100 * lift, 1), "end_dist_cm": round(100 * end_dist, 1),
            "tilt_deg": round(tilt, 1)}


def main(gen_dir: str, include_success: bool = False):
    rows = []
    for f in sorted(Path(gen_dir).glob("ep_*.npz")):
        z = np.load(f)
        if include_success or not bool(z["success"]):
            rows.append(classify(z))
    fails = [r for r in rows if not r["success"]]
    print(f"{len(fails)} failed episodes\n")
    print("what happened to the bowl:")
    for k, n in Counter(r["kind"] for r in fails).most_common():
        sub = [r for r in fails if r["kind"] == k]
        print(f"  {k:19s} {n:4d}   median end distance {np.median([r['end_dist_cm'] for r in sub]):5.1f} cm, "
              f"tilted > 30 deg: {sum(r['tilt_deg'] > 30 for r in sub)}")
    print("\nby last waypoint reached (rows) x what happened (columns):")
    kinds = ["never lifted", "left off plate", "on plate, > 3 cm", "within 3 cm, not ok"]
    print(f"  {'':10s}" + "".join(f"{k:>21s}" for k in kinds))
    for w, _ in Counter(r["last"] for r in fails).most_common():
        print(f"  {w:10s}" + "".join(f"{sum(r['last'] == w and r['kind'] == k for r in fails):21d}" for k in kinds))
    print("\nper demo (failures by kind):")
    for d in sorted({r["demo"] for r in fails}):
        c = Counter(r["kind"] for r in fails if r["demo"] == d)
        print(f"  {d:22s} " + ", ".join(f"{k}: {c[k]}" for k in kinds))
    if include_success:
        ok = [r for r in rows if r["success"]]
        print(f"\nsuccesses: {len(ok)}, median end distance {np.median([r['end_dist_cm'] for r in ok]):.1f} cm")
    return rows


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], "--all" in sys.argv)
