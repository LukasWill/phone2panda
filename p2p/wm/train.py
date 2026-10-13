"""Train the world model on the cached clips.

    python -m p2p.wm.train CACHE.npz OUT_DIR [--steps 20000] [--batch 32] [--horizon 32]

Curriculum on the open-loop horizon: training starts with short windows (8 steps = 0.8 s) and grows
linearly to --horizon over the first half of training. Long open-loop rollouts of an untrained model
produce garbage gradients; short ones teach the one-step physics first.
Checkpoint selection: the lowest validation loss at the full horizon (val = held-out seeds of the
training demos; the test demos are not touched here).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from p2p.wm.data import WindowSampler, load_cache, to_torch
from p2p.wm.model import WorldModel, losses


def horizon_at(step: int, steps: int, h_max: int, h_min: int = 8) -> int:
    return int(min(h_max, h_min + (h_max - h_min) * step / max(1, steps // 2)))


@torch.no_grad()
def val_loss(model, sampler, device, T, n_batches=8, batch=32) -> float:
    model.eval()
    out = []
    for _ in range(n_batches):
        out.append(float(losses(model, *to_torch(*sampler.sample(batch, T), device))["total"]))
    model.train()
    return float(np.mean(out))


def train(cache_path: Path, out_dir: Path, steps: int = 20000, batch: int = 32, horizon: int = 32,
          lr: float = 3e-4, seed: int = 0, log_every: int = 250, val_every: int = 1000) -> Path:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = load_cache(cache_path)
    tr, va = WindowSampler(cache, "train", seed), WindowSampler(cache, "val", seed + 1)
    model = WorldModel().to(device)
    b = cache["bowl"][np.isin(np.repeat(np.arange(len(cache["ptr"])), cache["length"]),
                              np.flatnonzero(cache["split"] == "train"))]
    model.bowl_mean.copy_(torch.tensor(b.mean(0)))
    model.bowl_std.copy_(torch.tensor(b.std(0) + 1e-3))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    out_dir.mkdir(parents=True, exist_ok=True)
    best, log, t0 = np.inf, [], time.time()
    for step in range(steps):
        T = horizon_at(step, steps, horizon)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            L = losses(model, *to_torch(*tr.sample(batch, T), device))
        opt.zero_grad(set_to_none=True)
        scaler.scale(L["total"]).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % log_every == 0:
            row = {"step": step, "T": T, **{k: round(float(v.detach()), 5) for k, v in L.items()}}
            log.append(row)
            print(row, f"{time.time() - t0:.0f}s", flush=True)
        if (step + 1) % val_every == 0 and T == horizon:
            v = val_loss(model, va, device, horizon)
            log.append({"step": step, "val_total": v})
            print(f"  val {v:.5f}", flush=True)
            if v < best:
                best = v
                torch.save({"model": model.state_dict(), "step": step, "val": v, "horizon": horizon}, out_dir / "wm.pt")
    if not (out_dir / "wm.pt").exists():          # very short runs never reach the full horizon
        torch.save({"model": model.state_dict(), "step": steps, "val": None, "horizon": horizon}, out_dir / "wm.pt")
    (out_dir / "train_log.json").write_text(json.dumps(log))
    return out_dir / "wm.pt"


def load(ckpt: Path, device: str | None = None) -> WorldModel:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = WorldModel().to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("out_dir")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    print("saved", train(Path(a.cache), Path(a.out_dir), a.steps, a.batch, a.horizon, a.lr))


if __name__ == "__main__":
    main()
