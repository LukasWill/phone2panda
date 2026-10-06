"""Write episodes as LeRobot datasets, so ACT (and any LeRobot policy) can train on them.

Two sources go through the SAME writer, so both datasets have identical image size, orientation,
state definition and fps, and the only difference between "ours" and "official" is the data:
  * our generated episodes (p2p/data/generate.py), and
  * the official LIBERO teleop demos of this task, taken from `lerobot/libero` and resized to 128 px.

    python -m p2p.data.lerobot_io ours     data/generated  data/lerobot/ours_all          [--limit N]
    python -m p2p.data.lerobot_io official -               data/lerobot/official
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

from p2p.sim.env import TASK_LANGUAGE

FPS = 20   # one frame per control step (LIBERO runs its controller at 20 Hz)


def features(size: int = 128) -> dict:
    img = {"dtype": "video", "shape": (size, size, 3), "names": ["height", "width", "channels"]}
    return {
        "observation.images.image": img,
        "observation.images.image2": dict(img),
        "observation.state": {"dtype": "float32", "shape": (8,),
                              "names": ["eef_x", "eef_y", "eef_z", "eef_ax", "eef_ay", "eef_az", "grip_l", "grip_r"]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]},
    }


def write_dataset(episodes: Iterable[dict], repo_id: str, root: Path, size: int = 128) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset.create(repo_id, FPS, features(size), root=root, robot_type="panda", use_videos=True)
    n = 0
    for ep in episodes:
        for t in range(len(ep["action"])):
            ds.add_frame({"observation.images.image": ep["image"][t], "observation.images.image2": ep["image2"][t],
                          "observation.state": ep["state"][t].astype(np.float32),
                          "action": ep["action"][t].astype(np.float32), "task": TASK_LANGUAGE})
        ds.save_episode()
        n += 1
    ds.finalize()
    return n


def generated_episodes(gen_dir: Path, success_only: bool = True, demos: set[str] | None = None,
                       limit: int | None = None, seeds: set[int] | None = None) -> Iterator[dict]:
    """Our episodes, in a fixed order (sorted by seed) so subsets like 'first 50' are reproducible."""
    n = 0
    for f in sorted(Path(gen_dir).glob("ep_*.npz")):
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        if success_only and not meta["success"]:
            continue
        if demos is not None and meta["demo"] not in demos:
            continue
        if seeds is not None and meta["seed"] not in seeds:
            continue
        yield {k: d[k] for k in ("image", "image2", "state", "action")}
        n += 1
        if limit is not None and n >= limit:
            return


def balanced_seeds(gen_dir: Path, n: int, success_only: bool = True) -> set[int]:
    """n episodes spread evenly over the human demos (round-robin), for a matched-count comparison
    with the official demos: 'ours-50' should not be 50 copies of the easiest demo."""
    per_demo: dict[str, list[int]] = {}
    for f in sorted(Path(gen_dir).glob("ep_*.npz")):
        meta = json.loads(str(np.load(f)["meta"]))
        if meta["success"] or not success_only:
            per_demo.setdefault(meta["demo"], []).append(meta["seed"])
    picked, queues = [], [list(v) for v in per_demo.values()]
    while len(picked) < n and any(queues):
        for q in queues:
            if q and len(picked) < n:
                picked.append(q.pop(0))
    return set(picked)


def official_episodes(size: int = 128, limit: int | None = None) -> Iterator[dict]:
    """The official teleop demos for our task from lerobot/libero (256 px, already in LeRobot orientation)."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata("lerobot/libero")
    eps = [int(r["episode_index"]) for r in meta.episodes if TASK_LANGUAGE in (r.get("tasks") or [])]
    eps = eps[:limit] if limit else eps
    ds = LeRobotDataset("lerobot/libero", episodes=eps)
    by_ep: dict[int, dict[str, list]] = {}
    for k in range(len(ds)):
        item = ds[k]
        e = int(item["episode_index"])
        buf = by_ep.setdefault(e, {"image": [], "image2": [], "state": [], "action": []})
        for key, out in (("observation.images.image", "image"), ("observation.images.image2", "image2")):
            img = (item[key].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            buf[out].append(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))
        buf["state"].append(item["observation.state"].numpy())
        buf["action"].append(item["action"].numpy())
    for e in sorted(by_ep):
        yield {k: np.stack(v) for k, v in by_ep[e].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["ours", "official"])
    ap.add_argument("gen_dir")
    ap.add_argument("out_root")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="include failed episodes (ours)")
    ap.add_argument("--balanced", type=int, default=None, help="N successes spread evenly over human demos")
    a = ap.parse_args()
    if a.source == "official":
        eps = official_episodes(limit=a.limit)
    else:
        seeds = balanced_seeds(Path(a.gen_dir), a.balanced) if a.balanced else None
        eps = generated_episodes(Path(a.gen_dir), success_only=not a.all, limit=a.limit, seeds=seeds)
    n = write_dataset(eps, f"local/{Path(a.out_root).name}", Path(a.out_root))
    print(f"wrote {n} episodes to {a.out_root}")


if __name__ == "__main__":
    main()
