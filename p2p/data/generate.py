"""Sim-verified data generation: many robot demos from few human demos.

    python -m p2p.data.generate data/processed data/generated --per-demo 40 --workers 2 [--strategy object_relative]

For each human demo we roll out its retargeted plan in many FRESH scenes (never LIBERO's 50 eval
scenes), each with a small random perturbation of the human's grasp point and place point. The
simulator is the judge: a rollout is a success only if LIBERO says the bowl is on the plate.

Why keep the failures? Imitation learning (ACT) trains on successes only, but
  * the world model must see what failure looks like to predict it, and
  * offline RL can learn from the good *parts* of failed attempts (a clean approach, a missed grasp).

Each episode is saved as one .npz:
  image (T,128,128,3), image2 (T,128,128,3)   LeRobot orientation (rotated 180 deg)
  state (T,8), action (T,7), low_dim (T,18), success (bool), plus where it came from.
Rendering needs a GPU runtime (MUJOCO_GL=egl).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

_ENV = None


def _init_worker():
    global _ENV
    from p2p.sim.env import TaskEnv

    _ENV = TaskEnv(render=True, camera_size=256, hard_reset=False)


def perturb_demo(demo: dict, rng: np.random.Generator, grasp_sigma: float, place_sigma: float) -> tuple[dict, dict]:
    """Shift where the human grasped (along the rim and radially) and where they placed the bowl."""
    d = dict(demo)
    rel = demo["grasp_point"] - demo["bowl_start"]
    ang = rng.normal(0, grasp_sigma / 0.05)              # sigma in metres along a 5 cm-radius rim
    c, s = np.cos(ang), np.sin(ang)
    rel = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1], rel[2]])
    rel[:2] *= 1 + rng.normal(0, 0.05)
    d["grasp_point"] = demo["bowl_start"] + rel
    dp = rng.normal(0, place_sigma, 2)
    d["bowl_end"] = demo["bowl_end"] + np.r_[dp, 0.0]
    return d, {"grasp_rot_deg": float(np.degrees(ang)), "place_shift_cm": (100 * dp).round(2).tolist()}


def _run(job):
    from p2p.retarget.strategies import make_plan
    from p2p.sim.env import BOWL, PLATE
    from p2p.sim.scripted import WaypointFollower

    demo, strategy, seed, cfg, out_dir = job
    rng = np.random.default_rng(seed)
    shift = {BOWL: rng.uniform(-cfg["obj_shift"], cfg["obj_shift"], 2),
             PLATE: rng.uniform(-cfg["obj_shift"], cfg["obj_shift"], 2)}
    _ENV.reset(seed=seed, object_shift=shift)
    d, pert = perturb_demo(demo, rng, cfg["grasp_sigma"], cfg["place_sigma"])
    follower = WaypointFollower(make_plan(strategy, d, _ENV.scene(), cfg.get("abs_offset")))
    buf = {k: [] for k in ("image", "image2", "state", "action", "low_dim")}
    success, done = False, False
    while not done:
        a = follower.act(_ENV.scene())
        f = _ENV.frame(size=cfg["image_size"])
        buf["image"].append(f["observation.images.image"])
        buf["image2"].append(f["observation.images.image2"])
        buf["state"].append(f["observation.state"])
        buf["action"].append(a.astype(np.float32))
        buf["low_dim"].append(_ENV.low_dim_state())
        _, success, done = _ENV.step(a)
    meta = {"demo": demo["name"], "strategy": strategy, "seed": seed, "success": bool(success),
            "steps": _ENV.t, "last_waypoint": follower.current, **pert}
    np.savez_compressed(Path(out_dir) / f"ep_{seed:06d}.npz", **{k: np.stack(v) for k, v in buf.items()},
                        success=success, meta=json.dumps(meta))
    return meta


def generate(demos: list[dict], out_dir: Path, per_demo: int, strategy: str, workers: int, cfg: dict,
             seed0: int = 1000) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(d, strategy, seed0 + 1000 * j + k, cfg, str(out_dir)) for j, d in enumerate(demos) for k in range(per_demo)]
    jobs = [jb for jb in jobs if not (out_dir / f"ep_{jb[2]:06d}.npz").exists()]   # resumable
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        metas = []
        for m in pool.imap_unordered(_run, jobs):
            metas.append(m)
            if len(metas) % 20 == 0:
                print(f"{len(metas)}/{len(jobs)} episodes, success so far {np.mean([x['success'] for x in metas]):.2f}",
                      flush=True)
    return metas


def main():
    from p2p.human.extract import load

    ap = argparse.ArgumentParser()
    ap.add_argument("processed_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--per-demo", type=int, default=40)
    ap.add_argument("--strategy", default="object_relative")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--holdout", default="demo_21,demo_22,demo_23,demo_24,demo_25")
    ap.add_argument("--demos", default="", help="only these demo names (comma-separated)")
    ap.add_argument("--grasp-sigma", type=float, default=0.01, help="m along the rim")
    ap.add_argument("--place-sigma", type=float, default=0.015, help="m")
    ap.add_argument("--obj-shift", type=float, default=0.03, help="max extra bowl/plate shift, m")
    ap.add_argument("--image-size", type=int, default=128)
    a = ap.parse_args()
    hold = set(filter(None, a.holdout.split(",")))
    only = set(filter(None, a.demos.split(",")))
    demos = [load(p) for p in sorted(Path(a.processed_dir).glob("*.npz"))]
    demos = [d for d in demos if d["name"] not in hold and (not only or d["name"] in only)]
    cfg = dict(grasp_sigma=a.grasp_sigma, place_sigma=a.place_sigma, obj_shift=a.obj_shift, image_size=a.image_size)
    t0 = time.time()
    metas = generate(demos, Path(a.out_dir), a.per_demo, a.strategy, a.workers, cfg)
    print(f"done: {len(metas)} episodes, {sum(m['success'] for m in metas)} successes, {time.time() - t0:.0f}s")
    with open(Path(a.out_dir) / "generation_log.jsonl", "a") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")


if __name__ == "__main__":
    main()
