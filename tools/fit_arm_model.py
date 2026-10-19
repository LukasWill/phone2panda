"""Fit the nominal arm model used by p2p/wm/plan_select.py (how the gripper moves for a given action).

    python tools/fit_arm_model.py            # ~2 min on CPU, physics only

The waypoint follower computes each action from the CURRENT gripper pose, so turning a plan into an action
sequence needs a model of how the arm responds. LIBERO's controller (OSC_POSE) does not simply jump by the
commanded delta (0.05 m per unit action); the arm lags and overshoots a little. A linear response with three
past motions and three past commands (an ARX model, fitted by least squares, the same coefficients for
x, y, z, and separately for rotation as axis-angle increments) captures this:

    dp_t = a1 dp_{t-1} + a2 dp_{t-2} + a3 dp_{t-3} + b0 u_t + b1 u_{t-1} + b2 u_{t-2},   u = 0.05 * action[:3]

Fitted only on the part of each rollout BEFORE the gripper closes (free motion): later, the held bowl or
contacts with the plate slow the arm, which is exactly what the arm model should NOT know about.
Check: following the same plans with this model reproduces the simulator's waypoint timings to within a few
steps (printed at the end); a one-lag model was ~35 steps (25%) too slow.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2p.sim.controller import POS_SCALE, ROT_SCALE, mat2axisangle  # noqa: E402

NA, NB = 3, 2


def rollouts(demos, inits, env):
    from p2p.retarget.strategies import make_plan
    from p2p.sim.scripted import WaypointFollower
    from p2p.wm.plan_select import candidates

    out = []
    for init in inits:
        for c in candidates(demos, init, 1)[init % 13: init % 13 + 2]:
            env.reset(init_state_id=init)
            plan = make_plan("object_relative", c["data"], env.scene())
            f, P, Rs, A, ev = WaypointFollower(plan), [], [], [], {}
            for t in range(300):
                s, i0 = env.scene(), f.i
                a = f.act(s)
                if f.i != i0:
                    ev.setdefault(plan[i0].name, t)
                P.append(s.eef_pos), Rs.append(s.eef_R), A.append(a)
                if f.done:
                    break
                env.step(a)
            out.append({"P": np.array(P), "R": np.array(Rs), "A": np.array(A), "events": ev, "plan": plan,
                        "init": init, "data": c["data"]})
    return out


def increments(r, kind):
    k = int(np.flatnonzero(r["A"][:, 6] > 0)[0])                   # free motion: before the first close
    P, Rs, A = r["P"][:k], r["R"][:k], r["A"][:k]
    if kind == "pos":
        return np.diff(P, axis=0), POS_SCALE * A[:-1, :3]
    return np.array([mat2axisangle(Rs[t + 1] @ Rs[t].T) for t in range(len(Rs) - 1)]), ROT_SCALE * A[:-1, 3:6]


def fit(runs, kind):
    X, Y = [], []
    for r in runs:
        d, u = increments(r, kind)
        for t in range(max(NA, NB), len(d)):
            for ax in range(3):
                X.append([d[t - i, ax] for i in range(1, NA + 1)] + [u[t - j, ax] for j in range(NB + 1)])
                Y.append(d[t, ax])
    X, Y = np.array(X), np.array(Y)
    c = np.linalg.lstsq(X, Y, rcond=None)[0]
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
    print(f"{kind}: a = {np.round(c[:NA], 3).tolist()}, b = {np.round(c[NA:], 3).tolist()}, "
          f"one-step residual {rms(Y - X @ c):.1e} vs signal {rms(Y):.1e}")
    return c


def main():
    from p2p.human.extract import load
    from p2p.sim.env import TaskEnv
    from p2p.wm.plan_select import ArmModel, nominal_actions

    root = Path(__file__).resolve().parents[1]
    split = json.loads((root / "data" / "splits.json").read_text())
    demos = [load(root / "data" / "processed" / f"{n}.npz") for n in split["main"]]
    env = TaskEnv(render=False, hard_reset=False)
    train = rollouts(demos, [3, 7, 11, 19, 23, 31], env)
    arm = ArmModel(pos=fit(train, "pos"), rot=fit(train, "rot"))
    # closed-loop check on other scenes: when does the follower pass each waypoint, model vs simulator?
    test = rollouts(demos, [0, 5, 9], env)
    keys, err = ["descend", "close", "lower", "release"], []
    for r in test:
        env.reset(init_state_id=r["init"])
        ev = {}
        nominal_actions(r["plan"], env.scene(), arm=arm, events=ev, cut=False)
        err += [ev[k] - r["events"][k] for k in keys if k in ev and k in r["events"]]
    print(f"waypoint timing, model minus simulator (steps): mean {np.mean(err):+.1f}, "
          f"median |error| {np.median(np.abs(err)):.0f}, over {len(test)} held-out rollouts")


if __name__ == "__main__":
    main()
