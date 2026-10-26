"""High-resolution videos for the README's top: a robot demo generated from one of my recordings, and the
learned policy solving evaluation scenes it never saw.

    python tools/render_hero.py OUT_DIR --ckpt runs/act_ours_all_vertical_s0/final --inits 0 1 2 3 4 5 \
        --demo 92e0b86bc6_lefthand --size 512

The policy sees exactly what it saw in evaluation (cameras rendered at 256 px, resized to 128 px); the video
comes from an extra render of the same camera at --size px, which does not touch the policy's input.
Writes policy_init{i:02d}_{ok|fail}_{camera}.mp4, teacher_{demo}_init{i:02d}_{ok|fail}_{camera}.mp4 (cameras:
agentview = the policy's view, frontview, sideview) and summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def hi_res(env, size: int, camera: str = "agentview") -> np.ndarray:
    """The same camera as the policy's main image, rendered at size x size, upright for viewing."""
    img = env._env.env.sim.render(width=size, height=size, camera_name=camera)
    return np.ascontiguousarray(img[::-1])            # OpenGL images come bottom-up


def save(frames, path: Path, fps: int = 20):
    import imageio

    imageio.mimsave(path, frames, fps=fps, quality=8, macro_block_size=8)


CAMERAS = ("agentview", "frontview", "sideview")


def grab(env, size):
    return {c: hi_res(env, size, c) for c in CAMERAS}


def save_all(frames, stem: str, out: Path) -> list[str]:
    names = []
    for c in CAMERAS:
        save([f[c] for f in frames], out / f"{stem}_{c}.mp4")
        names.append(f"{stem}_{c}.mp4")
    return names


def teacher(env, demo, init: int, size: int, out: Path) -> dict:
    from p2p.retarget.strategies import make_plan
    from p2p.sim.scripted import WaypointFollower

    env.reset(init_state_id=init)
    f = WaypointFollower(make_plan("object_relative_vertical", demo, env.scene()))
    frames, done, ok = [], False, False
    while not done:
        frames.append(grab(env, size))
        _, ok, done = env.step(f.act(env.scene()))
    frames.append(grab(env, size))
    names = save_all(frames, f"teacher_{demo['name'][:10]}_init{init:02d}_{'ok' if ok else 'fail'}", out)
    return {"init": init, "success": bool(ok), "steps": env.t, "videos": names}


def policy_run(env, policy, pre, post, init: int, size: int, out: Path) -> dict:
    import torch

    from p2p.policy.act import to_batch

    env.reset(init_state_id=init)
    policy.reset()
    frames, done, ok = [], False, False
    with torch.no_grad():
        while not done:
            frames.append(grab(env, size))
            a = post(policy.select_action(pre(to_batch(env.frame(size=128))))).squeeze(0).float().cpu().numpy()
            _, ok, done = env.step(a)
    frames.append(grab(env, size))
    names = save_all(frames, f"policy_init{init:02d}_{'ok' if ok else 'fail'}", out)
    return {"init": init, "success": bool(ok), "steps": env.t, "videos": names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--inits", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--demo", default="92e0b86bc6_lefthand")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    import torch

    from p2p.human.extract import load
    from p2p.sim.env import TaskEnv

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    env = TaskEnv(render=True, camera_size=256, hard_reset=False)
    summary = {"teacher": [], "policy": []}
    demo = load(root / "data" / "processed" / f"{a.demo}.npz")
    for i in a.inits:
        summary["teacher"].append(teacher(env, demo, i, a.size, out)); print("teacher", summary["teacher"][-1], flush=True)
    if a.ckpt:
        from p2p.policy.act import load as load_policy

        device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        policy, pre, post = load_policy(a.ckpt, device)
        for i in a.inits:
            summary["policy"].append(policy_run(env, policy, pre, post, i, a.size, out)); print("policy", summary["policy"][-1], flush=True)
    env.close()
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
