"""Expected generation yield per training demo, on any CPU (physics only, no rendering).

    python tools/expected_yield.py K results/expected_yield_k20.json

Reruns the first K seeds of notebook 02 for every training demo: same seeds, so the same scenes,
object shifts and grasp/place perturbations, and the same retargeting strategy (object_relative).

It also checks reproducibility: the first seed is run again in a fresh worker, and once more after
that worker has run another episode.
Finding (13 Oct):
  * A given seed places the objects identically.
  * A soft reset (hard_reset=False) keeps MuJoCo's solver warm-start and the controller's goal, so the
    robot starts a few mm differently, and a contact-rich rollout can end differently. Same seed:
    139 vs 142 steps, both successes.
This adds randomness, not a trend. A hard reset is exact but rebuilds the model (~1.3 s per episode).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_ENV = None


def _init():
    global _ENV
    from p2p.sim.env import TaskEnv

    _ENV = TaskEnv(render=False, hard_reset=False)


def _run(job):
    """Same as p2p.data.generate._run, minus the cameras."""
    from p2p.data.generate import perturb_demo
    from p2p.retarget.strategies import make_plan
    from p2p.sim.env import BOWL, PLATE
    from p2p.sim.scripted import WaypointFollower

    demo, seed, cfg = job
    rng = np.random.default_rng(seed)
    shift = {BOWL: rng.uniform(-cfg["obj_shift"], cfg["obj_shift"], 2),
             PLATE: rng.uniform(-cfg["obj_shift"], cfg["obj_shift"], 2)}
    _ENV.reset(seed=seed, object_shift=shift)
    d, pert = perturb_demo(demo, rng, cfg["grasp_sigma"], cfg["place_sigma"])
    f = WaypointFollower(make_plan("object_relative", d, _ENV.scene()))
    done, success = False, False
    while not done:
        _, success, done = _ENV.step(f.act(_ENV.scene()))
    return {"demo": demo["name"], "seed": seed, "success": bool(success), "steps": _ENV.t, "last": f.current,
            "bowl": _ENV.scene().bowl_pos.round(5).tolist(), **pert}


def main(K: int, out: Path):
    from p2p.human.extract import load

    hold = set(json.loads((ROOT / "data" / "splits.json").read_text())["holdout"])
    demos = [d for d in (load(p) for p in sorted((ROOT / "data" / "processed").glob("*.npz"))) if d["name"] not in hold]
    cfg = dict(grasp_sigma=0.01, place_sigma=0.015, obj_shift=0.03)          # notebook 02's defaults
    jobs = [(d, 1000 + 1000 * j + k, cfg) for j, d in enumerate(demos) for k in range(K)]   # its seeds
    t0 = time.time()
    with mp.get_context("spawn").Pool(2, initializer=_init) as pool:
        rows = list(pool.imap(_run, jobs, chunksize=1))
    with mp.get_context("spawn").Pool(1, initializer=_init) as pool:          # seed 0 fresh, then after another
        again = pool.map(_run, [jobs[0], jobs[1], jobs[0]], chunksize=1)
    out.write_text(json.dumps({"rows": rows, "again": again}))
    for d in demos:
        r = [x["success"] for x in rows if x["demo"] == d["name"]]
        print(f"{d['name']:22s} {sum(r)}/{len(r)}")
    print("total", sum(x["success"] for x in rows), "/", len(rows), f"{time.time() - t0:.0f}s")
    print("same seed: first run", rows[0]["steps"], rows[0]["bowl"], "| fresh worker", again[0]["steps"], again[0]["bowl"],
          "| after another episode", again[2]["steps"], again[2]["bowl"])


if __name__ == "__main__":
    main(int(sys.argv[1]), Path(sys.argv[2]))
