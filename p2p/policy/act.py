"""Train and evaluate ACT (LeRobot's implementation) with a short training loop of our own.

    python -m p2p.policy.act train data/lerobot/ours_all  runs/act_ours_all  --steps 20000
    python -m p2p.policy.act train data/lerobot/ours_mix  runs/act_awr       --weights runs/iql/weights.npz
    python -m p2p.policy.act eval  runs/act_ours_all/final --episodes 50 --videos 3

Why our own loop (≈60 lines) instead of `lerobot-train`? One reason: advantage-weighted
imitation needs a PER-SAMPLE weight on the loss, and ACT's `forward` only returns the batch mean.
`weighted_act_loss` below is ACT's loss written out per sample:
    L = sum_i w_i * [ L1(a_i, a_hat_i) + kl_weight * KL(q(z|a_i, s_i) || N(0, I)) ] / sum_i w_i
With all w_i = 1 it is exactly LeRobot's ACT loss. Why a weighted L1 is "offline RL": the L1 loss
is the negative log-likelihood of a Laplace policy, so weighting it by exp(beta * A(s, a)) is
advantage-weighted regression (AWR), the policy-extraction step of IQL.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def build(dataset_root: str, chunk: int = 50, n_action_steps: int = 25, device: str = "cuda",
          pretrained_backbone: bool = True):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    repo_id = f"local/{Path(dataset_root).name}"
    meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    cfg = ACTConfig(chunk_size=chunk, n_action_steps=n_action_steps, device=device,
                    pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1" if pretrained_backbone else None)
    policy = make_policy(cfg, ds_meta=meta)
    pre, post = make_pre_post_processors(cfg, dataset_stats=meta.stats)
    ds = LeRobotDataset(repo_id, root=dataset_root, delta_timestamps=resolve_delta_timestamps(cfg, meta))
    return policy, pre, post, ds


def weighted_act_loss(policy, batch: dict, weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    from lerobot.utils.constants import ACTION, OBS_IMAGES

    b = dict(batch)
    b[OBS_IMAGES] = [b[k] for k in policy.config.image_features]
    actions_hat, (mu, log_sigma_x2) = policy.model(b)
    err = F.l1_loss(b[ACTION], actions_hat, reduction="none")                 # (B, chunk, action_dim)
    valid = (~b["action_is_pad"]).unsqueeze(-1).float()                        # padded steps past episode end
    l1 = (err * valid).sum((1, 2)) / (valid.sum((1, 2)) * err.shape[-1]).clamp_min(1)
    per_sample = l1
    logs = {"l1": l1.mean().item()}
    if policy.config.use_vae and log_sigma_x2 is not None:
        kld = (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1)
        per_sample = per_sample + policy.config.kl_weight * kld
        logs["kld"] = kld.mean().item()
    w = torch.ones_like(per_sample) if weights is None else weights
    return (per_sample * w).sum() / w.sum().clamp_min(1e-6), logs


def train(dataset_root: str, out_dir: str, steps: int = 20000, batch_size: int = 32, lr: float = 2e-5,
          weights_file: str | None = None, device: str = "cuda", num_workers: int = 2, seed: int = 0,
          save_every: int = 5000, log_every: int = 200, pretrained_backbone: bool = True, amp: bool = True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy, pre, post, ds = build(dataset_root, device=device, pretrained_backbone=pretrained_backbone)
    w_lookup = None
    if weights_file:   # per-frame weights indexed by the dataset's global frame "index"
        w = np.load(weights_file)
        w_lookup = torch.as_tensor(w["weights"], dtype=torch.float32, device=device)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                         drop_last=True, pin_memory=device == "cuda", persistent_workers=num_workers > 0)
    backbone = [p for n, p in policy.named_parameters() if "backbone" in n and p.requires_grad]
    rest = [p for n, p in policy.named_parameters() if "backbone" not in n and p.requires_grad]
    opt = torch.optim.AdamW([{"params": rest, "lr": lr}, {"params": backbone, "lr": lr}], weight_decay=1e-4)
    use_amp = amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy.train()
    step, t0, hist = 0, time.time(), []
    while step < steps:
        for batch in loader:
            batch = pre(batch)
            weights = w_lookup[batch["index"].long().to(device)] if w_lookup is not None else None
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss, logs = weighted_act_loss(policy, batch, weights)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % log_every == 0 or step == 1:
                logs.update(step=step, loss=loss.item(), sec=round(time.time() - t0))
                hist.append(logs)
                print(json.dumps(logs), flush=True)
            if step % save_every == 0 or step == steps:
                save(policy, pre, post, out / ("final" if step == steps else f"step_{step:06d}"))
            if step >= steps:
                break
    (out / "train_log.json").write_text(json.dumps(hist))
    return out / "final"


def save(policy, pre, post, path: Path):
    policy.save_pretrained(path)
    pre.save_pretrained(path)
    post.save_pretrained(path)


def load(ckpt: str, device: str = "cuda"):
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(ckpt)
    policy.config.device = device
    policy.to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=ckpt,
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


def to_batch(frame: dict) -> dict:
    """Env frame (HWC uint8 images, float state) -> the dict a LeRobot policy expects (batch of 1)."""
    out = {}
    for k, v in frame.items():
        if k.startswith("observation.images."):
            out[k] = torch.from_numpy(np.ascontiguousarray(v)).permute(2, 0, 1).float().div(255).unsqueeze(0)
        else:
            out[k] = torch.from_numpy(np.asarray(v, dtype=np.float32)).unsqueeze(0)
    return out


@torch.no_grad()
def evaluate(ckpt: str, init_ids=range(50), device: str = "cuda", image_size: int = 128,
             video_dir: str | None = None, n_videos: int = 3) -> dict:
    from p2p.sim.env import TaskEnv
    from p2p.stats import fmt_rate

    policy, pre, post = load(ckpt, device)
    env = TaskEnv(render=True, camera_size=256)
    results = []
    for n, i in enumerate(init_ids):
        env.reset(init_state_id=i)
        policy.reset()
        frames, done, success = [], False, False
        while not done:
            f = env.frame(size=image_size)
            if video_dir and n < n_videos:
                frames.append(np.hstack([f["observation.images.image"], f["observation.images.image2"]]))
            action = post(policy.select_action(pre(to_batch(f))))
            _, success, done = env.step(action.squeeze(0).float().cpu().numpy())
        results.append({"init": int(i), "success": bool(success), "steps": env.t})
        if frames:
            import imageio

            Path(video_dir).mkdir(parents=True, exist_ok=True)
            imageio.mimsave(Path(video_dir) / f"init{i:02d}_{'ok' if success else 'fail'}.mp4", frames, fps=20)
    env.close()
    k = sum(r["success"] for r in results)
    summary = {"checkpoint": str(ckpt), "success": k, "n": len(results), "rate": fmt_rate(k, len(results)),
               "episodes": results}
    print(summary["rate"])
    return summary


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("dataset_root")
    t.add_argument("out_dir")
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--weights", default=None)
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--no-pretrained", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("ckpt")
    e.add_argument("--episodes", type=int, default=50)
    e.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    e.add_argument("--videos", type=int, default=3)
    e.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.dataset_root, a.out_dir, a.steps, a.batch_size, a.lr, a.weights, a.device, a.workers, a.seed,
              pretrained_backbone=not a.no_pretrained)
    else:
        out = Path(a.out or Path(a.ckpt).parent / "eval")
        s = evaluate(a.ckpt, range(a.episodes), a.device, video_dir=str(out / "videos"), n_videos=a.videos)
        out.mkdir(parents=True, exist_ok=True)
        (out / "eval.json").write_text(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
