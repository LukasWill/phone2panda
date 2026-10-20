"""What exactly happens in an episode: the same features for policy rollouts and for the scripted teacher.

Per step we record the gripper position, the gripper command and the bowl position. From that:
  closes          how many times the gripper command switched from open to closed
  close_r_cm      at the FIRST close: horizontal distance of the gripper from the bowl's centre axis (the rim
                  is at 5.6 cm; the teacher closes about 5 cm out, on the wall)
  close_dz_cm     at the first close: gripper height relative to the rim top (negative = below the rim)
  lifted          the bowl rose more than 2 cm at some point
  released        the gripper opened again after the bowl had been lifted
  frozen_end      steps at the end of the episode during which the gripper moved less than 0.5 mm per step
                  (a policy stuck in a fixed point: the image stops changing, so the prediction does too)
"""
from __future__ import annotations

import numpy as np

from p2p.sim.env import BOWL_HEIGHT


def episode_features(eef: np.ndarray, grip: np.ndarray, bowl: np.ndarray) -> dict:
    """eef (T, 3), grip (T,) commanded gripper (> 0 = close), bowl (T, 3)."""
    closed = grip > 0
    switches = np.flatnonzero(closed[1:] & ~closed[:-1]) + 1
    first = int(np.flatnonzero(closed)[0]) if closed.any() else None
    lift = bowl[:, 2] - bowl[0, 2]
    lifted_at = np.flatnonzero(lift > 0.02)
    released = bool(len(lifted_at) and (~closed[lifted_at[0]:]).any())
    speed = np.r_[np.linalg.norm(np.diff(eef, axis=0), axis=1), np.inf]
    still = speed[:-1] < 0.0005
    frozen = 0
    for s in still[::-1]:
        if not s:
            break
        frozen += 1
    out = {"closes": int(len(switches) + (1 if closed[0] else 0)), "lifted": bool(len(lifted_at)),
           "released": released, "frozen_end": int(frozen)}
    if first is not None:
        d = eef[first] - bowl[first]
        out.update(close_step=first, close_r_cm=round(100 * float(np.linalg.norm(d[:2])), 1),
                   close_dz_cm=round(100 * float(eef[first, 2] - (bowl[first, 2] + BOWL_HEIGHT)), 1))
    return out


def summarize(rows: list[dict]) -> dict:
    """Medians / counts over episodes, split by success."""
    out = {}
    for name, sel in (("success", [r for r in rows if r["success"]]), ("failure", [r for r in rows if not r["success"]])):
        if not sel:
            continue
        g = lambda k: [r[k] for r in sel if r.get(k) is not None]
        out[name] = {
            "n": len(sel),
            "never_closed": sum("close_step" not in r for r in sel),
            "closed_more_than_once": sum(r["closes"] > 1 for r in sel),
            "close_r_cm_median": float(np.median(g("close_r_cm"))) if g("close_r_cm") else None,
            "close_dz_cm_median": float(np.median(g("close_dz_cm"))) if g("close_dz_cm") else None,
            "close_dz_cm_p10_p90": np.percentile(g("close_dz_cm"), [10, 90]).round(1).tolist() if g("close_dz_cm") else None,
            "lifted": sum(r["lifted"] for r in sel),
            "lifted_and_released": sum(r["lifted"] and r["released"] for r in sel),
            "frozen_last_50_steps_or_more": sum(r["frozen_end"] >= 50 for r in sel),
        }
    return out
