"""Table 1: replay every human demo, with every retargeting strategy, in LIBERO's 50 eval scenes.

    python -m p2p.retarget.replay data/processed --out results/table1.csv [--inits 50] [--workers 2]

Physics only (no rendering), so it runs on any CPU. Each row of the CSV is one rollout.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np

from p2p.retarget.strategies import STRATEGIES, absolute_alignment, make_plan

_ENV = None


def _init_worker():
    global _ENV
    from p2p.sim.env import TaskEnv

    _ENV = TaskEnv(render=False, hard_reset=False)


def _run(job):
    from p2p.sim.scripted import WaypointFollower, run_episode

    demo, strategy, init_id, abs_offset = job
    _ENV.reset(init_state_id=init_id)
    s0 = _ENV.scene()
    try:
        plan = make_plan(strategy, demo, s0, abs_offset)
        r = run_episode(_ENV, WaypointFollower(plan))
    except Exception as e:  # a plan that cannot be built counts as a failure, with the reason kept
        r = {"success": False, "steps": 0, "last_waypoint": f"error: {type(e).__name__}"}
    s1 = _ENV.scene()
    return {"demo": demo["name"], "strategy": strategy, "init": init_id, "success": int(r["success"]),
            "steps": r["steps"], "last_waypoint": r["last_waypoint"],
            "bowl_to_plate_cm": round(100 * float(np.linalg.norm(s1.bowl_pos[:2] - s1.plate_pos[:2])), 1)}


def sim_bowl_mean(n: int = 50) -> np.ndarray:
    from p2p.sim.env import TaskEnv

    env = TaskEnv(render=False)
    pts = []
    for i in range(n):
        env.reset(init_state_id=i)
        pts.append(env.scene().bowl_pos)
    env.close()
    return np.mean(pts, 0)


def replay(demos: list[dict], strategies=STRATEGIES, inits=range(50), workers: int = 2) -> list[dict]:
    offset = absolute_alignment(demos, sim_bowl_mean())
    jobs = [(d, s, i, offset) for d in demos for s in strategies for i in inits]
    # 'spawn', not the default 'fork': forking a process that already initialised MuJoCo / numba
    # threads can deadlock the children. Spawned workers start clean and build their own env.
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        return list(pool.imap_unordered(_run, jobs, chunksize=2))


def summarize(rows: list[dict]) -> str:
    from p2p.stats import fmt_rate

    lines = []
    for s in STRATEGIES:
        r = [x for x in rows if x["strategy"] == s]
        if r:
            per_demo = {}
            for x in r:
                per_demo.setdefault(x["demo"], []).append(x["success"])
            demo_rates = [np.mean(v) for v in per_demo.values()]
            lines.append(f"{s:16s} {fmt_rate(sum(x['success'] for x in r), len(r))}   "
                         f"demos with >=50% success: {sum(v >= 0.5 for v in demo_rates)}/{len(demo_rates)}")
    return "\n".join(lines)


def main():
    from p2p.human.extract import load

    ap = argparse.ArgumentParser()
    ap.add_argument("processed_dir")
    ap.add_argument("--out", default="results/table1.csv")
    ap.add_argument("--inits", type=int, default=50)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--exclude", default="", help="comma-separated demo names to skip")
    a = ap.parse_args()
    skip = set(filter(None, a.exclude.split(",")))
    demos = [load(p) for p in sorted(Path(a.processed_dir).glob("*.npz")) if p.stem not in skip]
    t0 = time.time()
    rows = replay(demos, inits=range(a.inits), workers=a.workers)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x: (x["strategy"], x["demo"], x["init"])))
    print(summarize(rows))
    print(f"{len(rows)} rollouts in {time.time() - t0:.0f}s -> {out}")
    (out.with_suffix(".json")).write_text(json.dumps({"n_demos": len(demos), "inits": a.inits}))


if __name__ == "__main__":
    main()
