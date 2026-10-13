"""Evaluate the world model on episodes of the HELD-OUT human demos (never seen in training).

    python -m p2p.wm.evaluate WM.pt CACHE.npz OUT_DIR

Three questions, each against a baseline that needs no model:
  1. Can it predict the camera image h steps ahead from one real frame + actions?
     PSNR / SSIM vs h, against "copy the start frame" (= nothing moves). The autoencoder's own
     reconstruction (decode(encode(real))) is the ceiling: the dynamics cannot beat its own decoder.
  2. Does it know where the bowl goes? Bowl-position error vs h, against "the bowl stays where it was"
     (the true start position). Reported on all windows and on windows where the bowl really moves
     (> 2 cm), since a static bowl is trivially predicted.
  3. Can it tell success from failure before it happens? Imagine the last H steps of an episode from
     the real frame at N-H and score success by the imagined bowl's distance to the plate at the end.
     AUROC vs H; H = 0 reads the real last frame (perception only), H = "full" imagines the whole
     episode from its first frame (= judging a plan before executing it).
     Then per retargeting strategy: true success rate vs the rate the world model predicts.
Plus a side-by-side video: real vs imagined-from-the-first-frame.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from p2p.stats import fmt_rate
from p2p.wm.data import load_cache
from p2p.wm.train import load


# ---- metrics ---------------------------------------------------------------------------------------
def psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:                 # (B, 3, H, W) in [0, 1] -> (B,)
    return 10 * torch.log10(1.0 / F.mse_loss(a, b, reduction="none").flatten(1).mean(1).clamp_min(1e-10))


def ssim(a: torch.Tensor, b: torch.Tensor, win: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Standard SSIM (Wang et al. 2004) with a Gaussian window, averaged over pixels and channels."""
    g = torch.exp(-(torch.arange(win, device=a.device) - win // 2) ** 2 / (2 * sigma ** 2))
    g = (g / g.sum()).float()
    k = (g[:, None] * g[None]).expand(3, 1, win, win).contiguous()
    mu = lambda x: F.conv2d(x, k, groups=3)                                   # noqa: E731
    ma, mb = mu(a), mu(b)
    va, vb, cov = mu(a * a) - ma ** 2, mu(b * b) - mb ** 2, mu(a * b) - ma * mb
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))
    return s.flatten(1).mean(1)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """P(score of a random success > score of a random failure), ties count half (Mann-Whitney U)."""
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


# ---- rollouts --------------------------------------------------------------------------------------
def _frames(cache, rows, device):
    return torch.from_numpy(cache["frames"][rows]).to(device).permute(0, 3, 1, 2).float() / 255.0


@torch.no_grad()
def imagine_from(model, cache, ep: np.ndarray, start: np.ndarray, H: int, device, decode: bool = True):
    """Batch of (episode, start step) -> imagined latents for steps start+1..start+H, their images and bowl."""
    p = cache["ptr"][ep] + start
    z0 = model.encode(_frames(cache, p, device))
    act = torch.from_numpy(cache["actions"][p[:, None] + np.arange(H)[None]]).to(device)
    z = model.imagine(z0, act)                                                     # (B, H, Z)
    img = model.decode(z.flatten(0, 1)).view(len(ep), H, 3, 64, 64) if decode else None
    return z, img, model.bowl(z)


@torch.no_grad()
def horizon_metrics(model, cache, eps, H=40, stride=10, device="cpu", chunk=64) -> dict:
    jobs = [(e, s) for e in eps for s in range(0, cache["length"][e] - H, stride)]
    acc = {k: [[] for _ in range(H)] for k in ("psnr", "psnr_copy", "psnr_ae", "ssim", "ssim_copy",
                                               "bowl_err", "bowl_err_static", "moving")}
    for i in range(0, len(jobs), chunk):
        ep, st = map(np.array, zip(*jobs[i: i + chunk]))
        _, img, bowl = imagine_from(model, cache, ep, st, H, device)
        p = cache["ptr"][ep] + st
        start_img = _frames(cache, p, device)
        b0 = cache["bowl"][p]
        for h in range(H):
            real = _frames(cache, p + h + 1, device)
            ae = model.decode(model.encode(real))
            acc["psnr"][h] += psnr(img[:, h], real).tolist()
            acc["psnr_copy"][h] += psnr(start_img, real).tolist()
            acc["psnr_ae"][h] += psnr(ae, real).tolist()
            acc["ssim"][h] += ssim(img[:, h], real).tolist()
            acc["ssim_copy"][h] += ssim(start_img, real).tolist()
            bt = cache["bowl"][p + h + 1]
            acc["bowl_err"][h] += np.linalg.norm(bowl[:, h].cpu().numpy() - bt, axis=1).tolist()
            acc["bowl_err_static"][h] += np.linalg.norm(b0 - bt, axis=1).tolist()
            acc["moving"][h] += (np.linalg.norm(b0 - bt, axis=1) > 0.02).tolist()
    out = {"n_windows": len(jobs), "horizon_steps": list(range(1, H + 1))}
    for k in ("psnr", "psnr_copy", "psnr_ae", "ssim", "ssim_copy", "bowl_err", "bowl_err_static"):
        out[k] = [float(np.mean(v)) for v in acc[k]]
    for k in ("bowl_err", "bowl_err_static"):         # only windows where the bowl has really moved
        out[k + "_moving"] = [float(np.mean(np.array(v)[np.array(m, bool)])) if any(m) else float("nan")
                              for v, m in zip(acc[k], acc["moving"])]
    return out


@torch.no_grad()
def end_distance(model, cache, eps, H, device, chunk=64) -> np.ndarray:
    """Imagined bowl-to-plate distance (xy) at each episode's last frame, imagining its last H steps.
    H=0: read from the real last frame. H=None: imagine from the first frame (the whole episode).
    Episodes shorter than H are imagined from their first frame."""
    eps = np.asarray(eps)
    N = cache["length"][eps] - 1
    h = N if H is None else np.minimum(H, N)
    b = np.zeros((len(eps), 3), np.float32)
    for hv in np.unique(h):                       # batch the episodes that need the same number of steps
        idx = np.flatnonzero(h == hv)
        for i in range(0, len(idx), chunk):
            j = idx[i: i + chunk]
            if hv == 0:
                b[j] = model.bowl(model.encode(_frames(cache, cache["ptr"][eps[j]] + N[j], device))).cpu().numpy()
            else:
                _, _, bb = imagine_from(model, cache, eps[j], N[j] - hv, int(hv), device, decode=False)
                b[j] = bb[:, -1].cpu().numpy()
    return np.linalg.norm(b[:, :2] - cache["plate"][eps, :2], axis=1)


def best_threshold(dist: np.ndarray, success: np.ndarray) -> float:
    """Distance threshold (success if below) maximising balanced accuracy, chosen on VALIDATION episodes."""
    if success.all() or not success.any():
        return float(np.median(dist))              # one class only: no meaningful threshold
    cands = np.unique(dist)
    bal = [0.5 * ((dist[success] < t).mean() + (dist[~success] >= t).mean()) for t in cands]
    return float(cands[int(np.argmax(bal))])


# ---- video -------------------------------------------------------------------------------------------
def upright(x: np.ndarray) -> np.ndarray:
    """LeRobot stores LIBERO images rotated 180 deg from robosuite's (upside-down) render, which leaves
    them mirrored left-right; flip back for viewing only."""
    return x[..., :, ::-1, :]


@torch.no_grad()
def hero_video(model, cache, ep: int, out: Path, device, scale: int = 3):
    import cv2
    import imageio

    n = int(cache["length"][ep] - 1)
    _, img, bowl = imagine_from(model, cache, np.array([ep]), np.array([0]), n, device)
    pred = np.concatenate([cache["frames"][cache["ptr"][ep]][None],
                           (img[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)])
    real = cache["frames"][cache["ptr"][ep]: cache["ptr"][ep] + n + 1]
    tiles = []
    for t in range(n + 1):
        row = [cv2.resize(upright(x[t]), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) for x in (real, pred)]
        for im, label in zip(row, ("real", "imagined")):
            cv2.putText(im, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(row[1], f"{t / 10:.1f}s", (5, 64 * scale - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(np.hstack(row))
    imageio.mimsave(out.with_suffix(".mp4"), tiles, fps=10)
    strip = np.vstack([np.hstack([upright(x[t]) for t in np.linspace(0, n, 8).astype(int)]) for x in (real, pred)])
    imageio.imwrite(out.with_name(out.stem + "_filmstrip.png"), strip)


# ---- main ----------------------------------------------------------------------------------------------
def evaluate(ckpt: Path, cache_path: Path, out_dir: Path, H: int = 40):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(ckpt, device)
    c = load_cache(cache_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    test, val = np.flatnonzero(c["split"] == "test"), np.flatnonzero(c["split"] == "val")
    succ = c["success"].astype(bool)
    res = {"n_test": int(len(test)), "n_val": int(len(val))}

    # 1 + 2: prediction quality vs horizon
    hm = horizon_metrics(model, c, test, H=H, device=device)
    res["horizon"] = hm
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    t = np.array(hm["horizon_steps"]) / 10
    ax[0].plot(t, hm["psnr"], label="world model"); ax[0].plot(t, hm["psnr_copy"], "--", label="copy start frame")
    ax[0].plot(t, hm["psnr_ae"], ":", label="autoencoder ceiling"); ax[0].set(xlabel="seconds ahead", ylabel="PSNR (dB)")
    ax[1].plot(t, 100 * np.array(hm["bowl_err_moving"]), label="world model")
    ax[1].plot(t, 100 * np.array(hm["bowl_err_static_moving"]), "--", label="bowl stays put")
    ax[1].set(xlabel="seconds ahead", ylabel="bowl error (cm), windows where it moves")
    for a in ax:
        a.legend(); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "wm_horizon.png", dpi=150); plt.close(fig)

    # 3: success from imagination
    Hs = [0, 5, 10, 20, 40, 80, None]
    res["auroc"], res["threshold"], dist_test = {}, {}, {}
    true_end = np.linalg.norm(c["bowl"][c["ptr"][test] + c["length"][test] - 1, :2] - c["plate"][test, :2], axis=1)
    res["auroc_true_bowl"] = auroc(-true_end, succ[test])        # is "distance to plate" a valid success proxy?
    for h in Hs:
        key = "full" if h is None else str(h)
        dist_test[key] = end_distance(model, c, test, h, device)
        res["auroc"][key] = auroc(-dist_test[key], succ[test])
        res["threshold"][key] = best_threshold(end_distance(model, c, val, h, device), succ[val])
        print(f"H={key:>4s}: AUROC {res['auroc'][key]:.3f}", flush=True)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(range(len(Hs)), [res["auroc"]["full" if h is None else str(h)] for h in Hs], "o-")
    ax.axhline(0.5, ls="--", c="gray"); ax.axhline(res["auroc_true_bowl"], ls=":", c="green", label="true bowl position")
    ax.set_xticks(range(len(Hs)), [f"{h / 10:g}s" if h is not None else "whole\nepisode" for h in Hs])
    ax.set(xlabel="imagined before the end", ylabel="success AUROC (held-out demos)", ylim=(0.4, 1.02))
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out_dir / "wm_auroc.png", dpi=150); plt.close(fig)

    # per strategy: true vs predicted success (judging a whole plan from its first frame, and from 2 s out)
    rows = []
    for src in sorted(set(c["source"][test])):
        m = c["source"][test] == src
        r = {"source": str(src), "strategy": str(c["strategy"][test][m][0]), "n": int(m.sum()),
             "true_success": fmt_rate(int(succ[test][m].sum()), int(m.sum()))}
        for key in ("full", "20"):
            pred = dist_test[key][m] < res["threshold"][key]
            r[f"wm_predicted_{key}"] = fmt_rate(int(pred.sum()), int(m.sum()))
            r[f"agreement_{key}"] = round(float((pred == succ[test][m]).mean()), 2)
        rows.append(r)
    with open(out_dir / "wm_strategies.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    res["strategies"] = rows
    for r in rows:
        print(r, flush=True)

    # hero video: one success and one failure from the held-out demos (object-relative if present)
    for label, want in (("success", True), ("failure", False)):
        cand = [e for e in test if succ[e] == want and c["strategy"][e] == "object_relative"] or \
               [e for e in test if succ[e] == want]
        if cand:
            hero_video(model, c, int(cand[0]), out_dir / f"wm_hero_{label}", device)
    (out_dir / "wm_metrics.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("cache")
    ap.add_argument("out_dir")
    ap.add_argument("--horizon", type=int, default=40, help="steps (10 Hz) for the image / bowl curves")
    a = ap.parse_args()
    evaluate(Path(a.ckpt), Path(a.cache), Path(a.out_dir), a.horizon)


if __name__ == "__main__":
    main()
