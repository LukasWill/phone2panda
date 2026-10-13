"""World-model data: generated LIBERO episodes -> one compact cache of 10 Hz, 64 px clips.

    python -m p2p.wm.data CACHE.npz  DIR[:LABEL] [DIR[:LABEL] ...]

Each DIR holds ep_*.npz files written by p2p.data.generate (successes AND failures). Per episode we keep
  frames   (N+1, 64, 64, 3) uint8   agentview camera every 2nd control step (20 Hz -> 10 Hz)
  actions  (N, 14)          float32 the two 7-D actions between consecutive kept frames
  bowl     (N+1, 3)         float32 bowl position (world, metres), from the simulator
  plate    (3,)             float32 plate position at the start (it does not move in practice)
Episode-level labels: success, demo, strategy, seed, and a split:
  test  = episodes from the HELD-OUT human demos (data/splits.json): the world model never sees them
  val   = ~10% of the other episodes (chosen by seed), for model selection and the success threshold
  train = the rest
Frames and actions of all episodes are concatenated; `ptr[i]` is where episode i's frames start, and
actions are stored padded to the same length (last row zero) so one index addresses both.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from p2p.wm.model import IMG

FRAME_SKIP = 2      # 20 Hz control -> 10 Hz world model


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[1] == size:
        return img
    import cv2

    return np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in img])


def load_episode(path: Path, size: int = IMG) -> dict:
    """Images and low_dim are recorded BEFORE each action, so frame i + action i -> frame i+1.
    The observation after the very last action is not stored; the last stored frame is at most one
    control step (50 ms) before the end."""
    z = np.load(path)
    T = len(z["action"])
    n = (T - 1) // FRAME_SKIP                       # number of 10 Hz steps
    idx = np.arange(n + 1) * FRAME_SKIP
    act = z["action"][: n * FRAME_SKIP].reshape(n, FRAME_SKIP * 7)
    meta = json.loads(str(z["meta"]))
    return {"frames": _resize(z["image"][idx], size), "actions": act.astype(np.float32),
            "bowl": z["low_dim"][idx, 8:11].astype(np.float32), "plate": z["low_dim"][0, 15:18].astype(np.float32),
            "success": bool(z["success"]), "demo": meta["demo"], "strategy": meta["strategy"],
            "seed": int(meta["seed"])}


def build_cache(sources: list[tuple[Path, str]], out: Path, holdout: set[str]) -> dict:
    """sources: (directory, label). The label (e.g. 'gen', 'holdout_absolute') is kept per episode."""
    eps = []
    for d, label in sources:
        files = sorted(Path(d).glob("ep_*.npz"))
        print(f"{d}: {len(files)} episodes", flush=True)
        for f in files:
            e = load_episode(f)
            e["source"] = label
            eps.append(e)
    lens = np.array([len(e["frames"]) for e in eps])
    ptr = np.r_[0, np.cumsum(lens)[:-1]]
    actions = np.zeros((lens.sum(), eps[0]["actions"].shape[1]), np.float32)
    for e, p in zip(eps, ptr):
        actions[p: p + len(e["actions"])] = e["actions"]
    split = np.array(["test" if e["demo"] in holdout else ("val" if e["seed"] % 10 == 0 else "train") for e in eps])
    cache = dict(frames=np.concatenate([e["frames"] for e in eps]), actions=actions,
                 bowl=np.concatenate([e["bowl"] for e in eps]), plate=np.stack([e["plate"] for e in eps]),
                 ptr=ptr, length=lens, success=np.array([e["success"] for e in eps]), split=split,
                 demo=np.array([e["demo"] for e in eps]), strategy=np.array([e["strategy"] for e in eps]),
                 source=np.array([e["source"] for e in eps]), seed=np.array([e["seed"] for e in eps]))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **cache)
    for s in ("train", "val", "test"):
        m = split == s
        print(f"{s:5s}: {m.sum():4d} episodes, {cache['success'][m].mean() if m.any() else float('nan'):.2f} success, "
              f"{lens[m].sum()} frames", flush=True)
    return cache


def load_cache(path: Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


class WindowSampler:
    """Random training windows of T world-model steps (T+1 frames) from one split."""

    def __init__(self, cache: dict, split: str, seed: int = 0):
        self.c = cache
        self.eps = np.flatnonzero(cache["split"] == split)
        self.rng = np.random.default_rng(seed)

    def sample(self, batch: int, T: int):
        L = self.c["length"][self.eps]
        ok = self.eps[L >= T + 1]
        ep = self.rng.choice(ok, batch)
        start = self.c["ptr"][ep] + self.rng.integers(0, self.c["length"][ep] - T)    # window fits in the episode
        rows = start[:, None] + np.arange(T + 1)[None]
        frames = self.c["frames"][rows]                       # (B, T+1, 64, 64, 3) uint8
        return frames, self.c["actions"][rows[:, :-1]], self.c["bowl"][rows]


def to_torch(frames, actions, bowl, device):
    import torch

    f = torch.from_numpy(frames).to(device).permute(0, 1, 4, 2, 3).float() / 255.0
    return f, torch.from_numpy(actions).to(device), torch.from_numpy(bowl).to(device)


def main():
    out = Path(sys.argv[1])
    sources = []
    for a in sys.argv[2:]:
        d, _, label = a.partition(":")
        sources.append((Path(d), label or Path(d).name))
    sp = Path(__file__).resolve().parents[2] / "data" / "splits.json"
    holdout = set(json.loads(sp.read_text())["holdout"])
    print("test split = episodes of held-out demos:", sorted(holdout))
    build_cache(sources, out, holdout)


if __name__ == "__main__":
    main()
