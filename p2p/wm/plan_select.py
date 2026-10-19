"""The world model as a plan critic: choose which retargeted demo to execute, BEFORE acting.

    python -m p2p.wm.plan_select WM.pt OUT_DIR [--inits 50] [--per-demo 1] [--workers 2]

Setting. A robot faces a new scene (one of LIBERO's 50 evaluation scenes; the world model never saw them).
It has N candidate plans: the object-relative retargeting of each of my 13 demos, each with a random
perturbation of its grasp and place point (exactly as in data generation), so some candidates will work in
this scene and some will not. It may look at the scene (one camera frame + the object poses) but may not try
anything. Which plan should it execute?

1. Plan -> actions, without the simulator. The waypoint follower turns a plan into actions by feedback on
   the gripper pose. Here that pose comes from a NOMINAL model of the arm instead of the simulator: a linear
   response of the gripper motion to the commanded deltas (ArmModel below, fitted by tools/fit_arm_model.py
   on free-motion rollouts; it reproduces the simulator's waypoint timings to within a few steps). It knows
   nothing about contacts, the bowl or the cabinet: the robot knows its own arm, not the outcome. The
   sequence is cut where the gripper has opened at the place point (the end of the "release" waypoint).
2. The world model imagines that action sequence from the real first frame and reads where the bowl ends.
3. Execute the candidate whose imagined bowl ends closest to the plate centre.

Scoring (simulator used only here): every candidate is also executed in the simulator, which gives
  * world-model choice: success of the chosen candidate,
  * random choice: mean success of all candidates (what picking blindly gives on average),
  * best fixed demo: always execute the demo with the highest generation yield (65ee), chosen in advance,
  * oracle: at least one candidate succeeds (the ceiling of any selection rule),
  * diagnostic: the same world model fed the actions that were really executed, which removes the error of
    the nominal arm model and isolates the world model's own error.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

PERTURB = {"grasp_sigma": 0.01, "place_sigma": 0.015}                                 # as in data generation
CUT_AFTER = "release"   # cut the action sequence when the follower moves past this waypoint
_ENV = None


def expmap(w: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


@dataclass
class ArmModel:
    """dp_t = a1 dp_{t-1} + a2 dp_{t-2} + a3 dp_{t-3} + b0 u_t + b1 u_{t-1} + b2 u_{t-2} for the gripper
    position (u = 0.05 * action[:3]) and, with its own coefficients, for its rotation (axis-angle increments,
    u = 0.5 * action[3:6]). Coefficients [a1, a2, a3, b0, b1, b2] from tools/fit_arm_model.py."""
    pos: np.ndarray = field(default_factory=lambda: np.array([1.510, -0.702, 0.173, 0.101, -0.060, -0.036]))
    rot: np.ndarray = field(default_factory=lambda: np.array([1.451, -0.640, 0.107, 0.111, -0.091, -0.001]))


def nominal_actions(plan, s0, arm: ArmModel | None = None, max_steps: int = 300, cut: bool = True,
                    events: dict | None = None) -> np.ndarray:
    """Run the waypoint follower against the nominal arm model (no simulator). Returns (T, 7) actions.
    events (optional, filled in): waypoint name -> step at which the follower moved past it."""
    from p2p.sim.controller import POS_SCALE, ROT_SCALE
    from p2p.sim.scripted import WaypointFollower

    arm = arm or ArmModel()
    na, nb = 3, 2
    f = WaypointFollower(plan)
    p, R = np.array(s0.eef_pos, float), np.array(s0.eef_R, float)
    dp, dw = [np.zeros(3)] * na, [np.zeros(3)] * na            # past motions, newest first
    up, uw = [np.zeros(3)] * (nb + 1), [np.zeros(3)] * (nb + 1)  # past commands, newest first
    out = []
    for t in range(max_steps):
        i0 = f.i
        a = f.act(replace(s0, eef_pos=p, eef_R=R))
        out.append(a)
        if events is not None and f.i != i0:
            events.setdefault(plan[i0].name, t)
        if f.done or (cut and CUT_AFTER in _passed(f)):          # the gripper has opened at the place point
            break
        up, uw = [POS_SCALE * a[:3]] + up[:-1], [ROT_SCALE * a[3:6]] + uw[:-1]
        step_p = sum(arm.pos[i] * dp[i] for i in range(na)) + sum(arm.pos[na + j] * up[j] for j in range(nb + 1))
        step_w = sum(arm.rot[i] * dw[i] for i in range(na)) + sum(arm.rot[na + j] * uw[j] for j in range(nb + 1))
        dp, dw = [step_p] + dp[:-1], [step_w] + dw[:-1]
        p, R = p + step_p, expmap(step_w) @ R
    return np.array(out, np.float32)


def _passed(f) -> list[str]:
    """Names of the waypoints the follower has completed."""
    return [wp.name for wp in f.waypoints[: f.i]]


def candidates(demos: list[dict], init_id: int, per_demo: int, perturb: bool = True) -> list[dict]:
    from p2p.data.generate import perturb_demo

    out = []
    for j, d in enumerate(demos):
        for k in range(per_demo):
            rng = np.random.default_rng(1_000_000 + 10_000 * init_id + 100 * j + k)   # reproducible per scene
            dd, pert = perturb_demo(d, rng, **PERTURB) if perturb else (d, {})
            out.append({"demo": d["name"], "k": k, "data": dd, **pert})
    return out


# ---- ground truth in the simulator (physics only) --------------------------------------------------
def _init_worker():
    global _ENV
    from p2p.sim.env import TaskEnv

    _ENV = TaskEnv(render=False, hard_reset=False)


def _ground_truth(job):
    from p2p.retarget.strategies import make_plan
    from p2p.sim.scripted import WaypointFollower

    init_id, ci, demo = job
    _ENV.reset(init_state_id=init_id)
    f = WaypointFollower(make_plan("object_relative", demo, _ENV.scene()))
    acts, cut, done, success = [], None, False, False
    while not done:
        a = f.act(_ENV.scene())
        acts.append(a)
        if cut is None and CUT_AFTER in _passed(f):
            cut = len(acts)
        _, success, done = _ENV.step(a)
    s = _ENV.scene()
    return {"init": init_id, "cand": ci, "success": bool(success), "steps": _ENV.t,
            "bowl_to_plate_cm": round(100 * float(np.linalg.norm(s.bowl_pos[:2] - s.plate_pos[:2])), 1),
            "executed": np.array(acts[: cut or len(acts)], np.float32)}


# ---- the world model's verdict ------------------------------------------------------------------------
def first_frame(env, init_id: int) -> tuple[np.ndarray, object]:
    """The camera frame the world model starts from, prepared exactly like its training data
    (LeRobot orientation, 256 px render -> 128 px -> 64 px with area averaging)."""
    import cv2

    env.reset(init_state_id=init_id)
    img = env.frame(size=128)["observation.images.image"]
    return cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA), env.scene()


def wm_end_distance(model, frame64: np.ndarray, action_seqs: list[np.ndarray], plate_xy: np.ndarray,
                    device: str) -> np.ndarray:
    """Imagined bowl-to-plate distance (m) at the end of each action sequence, all from the same frame."""
    import torch

    pairs = [a[: (len(a) // 2) * 2].reshape(-1, 14) for a in action_seqs]   # 20 Hz -> 10 Hz action pairs
    n = np.array([len(p) for p in pairs])
    A = np.zeros((len(pairs), max(n.max(), 1), 14), np.float32)
    for i, p in enumerate(pairs):
        A[i, : len(p)] = p                                                  # zero padding after the end
    with torch.no_grad():
        x = torch.from_numpy(frame64).to(device).permute(2, 0, 1).float().div(255).unsqueeze(0)
        z0 = model.encode(x).expand(len(pairs), -1)
        z = model.imagine(z0, torch.from_numpy(A).to(device))               # causal: padding never leaks back
        last = z[torch.arange(len(pairs)), torch.as_tensor(np.maximum(n - 1, 0))]
        bowl = model.bowl(last).cpu().numpy()
    d = np.linalg.norm(bowl[:, :2] - plate_xy[None], axis=1)
    d[n == 0] = np.inf
    return d


def run(ckpt: Path, out_dir: Path, inits=range(50), per_demo: int = 1, workers: int = 2,
        best_fixed: str = "65ee6ec786", perturb: bool = True) -> dict:
    import torch

    from p2p.human.extract import load
    from p2p.retarget.strategies import make_plan
    from p2p.sim.env import TaskEnv, prefetch_assets
    from p2p.stats import fmt_rate
    from p2p.wm.evaluate import auroc
    from p2p.wm.train import load as load_wm

    root = Path(__file__).resolve().parents[2]
    split = json.loads((root / "data" / "splits.json").read_text())
    demos = [load(root / "data" / "processed" / f"{n}.npz") for n in split["main"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_wm(ckpt, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1+2: candidates, nominal actions and the world model's verdict, scene by scene (one rendering env)
    env = TaskEnv(render=True, camera_size=256, hard_reset=False)
    cands, rows = {}, []
    for i in inits:
        frame, s0 = first_frame(env, i)
        cands[i] = candidates(demos, i, per_demo, perturb)
        nominal = [nominal_actions(make_plan("object_relative", c["data"], s0), s0) for c in cands[i]]
        d_nom = wm_end_distance(model, frame, nominal, s0.plate_pos[:2], device)
        for ci, c in enumerate(cands[i]):
            rows.append({"init": i, "cand": ci, "demo": c["demo"], "held_out": c["demo"] in split["holdout"],
                         "k": c["k"], "nominal_steps": len(nominal[ci]), "wm_dist_nominal_cm": 100 * d_nom[ci],
                         "_nominal": nominal[ci], "_frame": frame, "_plate": s0.plate_pos[:2]})
    env.close()
    print(f"world model verdicts for {len(rows)} candidates in {time.time() - t0:.0f}s", flush=True)

    # ground truth: every candidate in the simulator (only to score the choice)
    prefetch_assets()
    jobs = [(r["init"], r["cand"], cands[r["init"]][r["cand"]]["data"]) for r in rows]
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        gt = {(g["init"], g["cand"]): g for g in pool.imap_unordered(_ground_truth, jobs, chunksize=2)}
    print(f"ground truth done in {time.time() - t0:.0f}s", flush=True)

    # diagnostic: the world model fed the actions that were really executed
    for i in inits:
        rr = [r for r in rows if r["init"] == i]
        ex = [gt[(i, r["cand"])]["executed"] for r in rr]
        d_ex = wm_end_distance(model, rr[0]["_frame"], ex, rr[0]["_plate"], device)
        for r, d, e in zip(rr, d_ex, ex):
            g = gt[(i, r["cand"])]
            m = min(len(e), len(r["_nominal"]))
            r.update(wm_dist_executed_cm=100 * d, success=g["success"], steps=g["steps"],
                     bowl_to_plate_cm=g["bowl_to_plate_cm"],
                     nominal_vs_executed_action_gap=float(np.abs(r["_nominal"][:m] - e[:m]).mean()) if m else np.nan)

    # scoring
    def choose(rr, key):
        return min(rr, key=lambda r: r[key])

    per_scene, res = [], {}
    for i in inits:
        rr = [r for r in rows if r["init"] == i]
        per_scene.append({"init": i, "wm": choose(rr, "wm_dist_nominal_cm")["success"],
                          "wm_executed": choose(rr, "wm_dist_executed_cm")["success"],
                          "random": float(np.mean([r["success"] for r in rr])),
                          "best_fixed": float(np.mean([r["success"] for r in rr if r["demo"] == best_fixed])),
                          "oracle": any(r["success"] for r in rr)})
    n = len(per_scene)
    succ = np.array([r["success"] for r in rows], bool)
    mixed = [p for p in per_scene if 0 < p["random"] < 1]     # scenes where the choice matters
    res = {"scenes": n, "candidates_per_scene": len(rows) // n, "best_fixed_demo": best_fixed,
           "world_model_choice": fmt_rate(sum(p["wm"] for p in per_scene), n),
           "random_choice_expected": f"{100 * np.mean([p['random'] for p in per_scene]):.0f}%",
           "best_fixed_demo_rate": f"{100 * np.mean([p['best_fixed'] for p in per_scene]):.0f}%",
           "oracle": fmt_rate(sum(p["oracle"] for p in per_scene), n),
           "diagnostic_wm_with_executed_actions": fmt_rate(sum(p["wm_executed"] for p in per_scene), n),
           "scenes_where_choice_matters": len(mixed),
           "in_those_scenes": {"world_model": fmt_rate(sum(p["wm"] for p in mixed), len(mixed)) if mixed else "-",
                               "random_expected": f"{100 * np.mean([p['random'] for p in mixed]):.0f}%" if mixed else "-"},
           "within_scene_auroc_nominal": float(np.mean(
               [auroc(-np.array([r["wm_dist_nominal_cm"] for r in rows if r["init"] == p["init"]]),
                      np.array([r["success"] for r in rows if r["init"] == p["init"]], bool)) for p in mixed]))
               if mixed else float("nan"),
           "candidate_auroc_nominal": auroc(-np.array([r["wm_dist_nominal_cm"] for r in rows]), succ),
           "candidate_auroc_executed": auroc(-np.array([r["wm_dist_executed_cm"] for r in rows]), succ),
           "candidate_success_rate": f"{100 * succ.mean():.0f}%",
           "median_nominal_vs_executed_action_gap": float(np.nanmedian([r["nominal_vs_executed_action_gap"] for r in rows])),
           "seconds": round(time.time() - t0)}
    keys = ["init", "cand", "demo", "held_out", "k", "success", "steps", "bowl_to_plate_cm", "nominal_steps",
            "wm_dist_nominal_cm", "wm_dist_executed_cm", "nominal_vs_executed_action_gap"]
    with open(out_dir / "plan_select_candidates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([round(r[k], 3) if isinstance(r[k], float) else r[k] for k in keys])
    (out_dir / "plan_select_summary.json").write_text(json.dumps({**res, "per_scene": per_scene}, indent=1))
    print(json.dumps(res, indent=1), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out_dir")
    ap.add_argument("--inits", type=int, default=50)
    ap.add_argument("--per-demo", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--best-fixed", default="65ee6ec786")
    ap.add_argument("--no-perturb", action="store_true")
    a = ap.parse_args()
    run(Path(a.ckpt), Path(a.out_dir), range(a.inits), a.per_demo, a.workers, a.best_fixed, not a.no_perturb)


if __name__ == "__main__":
    main()
