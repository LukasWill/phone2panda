"""Offline RL on the generated episodes (successes AND failures): IQL critic -> per-frame weights for ACT.

    python -m p2p.rl.iql GEN_DIR OUT_DIR [--steps 50000]

Why: the ACT baseline ("filtered BC") throws every failed rollout away, and treats every frame of a
successful one as equally good. Advantage-weighted regression keeps all data but weights each frame by
how much better its action is than what is typical from that state:
    w(s, a) = exp(beta * A(s, a)),   A = Q(s, a) - V(s)
and ACT's loss becomes sum_i w_i * loss_i (p2p/policy/act.py, --weights). This is IQL's policy
extraction step (Kostrikov et al., 2021, "Offline RL with Implicit Q-Learning") with ACT as the policy.

IQL in three lines (no action is ever queried outside the dataset, hence no over-estimation):
    V(s)    <- expectile_tau of Q(s, a) over the dataset's actions in s      (tau = 0.7: an "optimistic mean")
    Q(s, a) <- r + gamma * (1 - done) * V(s')                                 (plain TD on dataset transitions)
    A(s, a)  = Q(s, a) - V(s)

Design choices:
  * privileged critic, visual actor: the critic sees the simulator's low-dim state (robot + bowl + plate,
    18-D), the policy still sees only images. The critic is used only offline to weight the data, so it
    may use information the robot will not have at test time.
  * reward: 1 on the step where LIBERO reports success (which ends the episode), else 0, gamma = 0.99.
    A failed episode ends at the 300-step limit; we treat that as terminal with value 0 (the evaluation
    uses the same limit, so a failure at step 300 is final).
  * beta acts on STANDARDISED advantages, so it is scale-free: if A is ~Gaussian, exp(beta * z) keeps an
    effective sample size of exp(-beta^2) of the data (beta = 1 -> ~37%; beta = 3 -> almost nothing).
    Weights are clipped at 20 and normalised to mean 1, so the effective learning rate is unchanged.
The frame order is exactly ours_mix's (p2p/data/lerobot_io.py, --all): episode files sorted by name.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_transitions(gen_dir: Path) -> dict:
    S, A, R, D, S2, ep_id, succ = [], [], [], [], [], [], []
    for i, f in enumerate(sorted(Path(gen_dir).glob("ep_*.npz"))):          # same order as ours_mix
        z = np.load(f)
        s, a, ok = z["low_dim"].astype(np.float32), z["action"].astype(np.float32), bool(z["success"])
        T = len(a)
        r = np.zeros(T, np.float32)
        r[-1] = float(ok)
        d = np.zeros(T, np.float32)
        d[-1] = 1.0                                   # success: true terminal; failure: the time limit (see above)
        s2 = np.concatenate([s[1:], s[-1:]])          # s' of the last step is unused (done = 1)
        S.append(s); A.append(a); R.append(r); D.append(d); S2.append(s2)
        ep_id.append(np.full(T, i)); succ.append(np.full(T, ok))
    cat = lambda x: np.concatenate(x)                 # noqa: E731
    return {"s": cat(S), "a": cat(A), "r": cat(R), "d": cat(D), "s2": cat(S2), "ep": cat(ep_id), "success": cat(succ)}


def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """Asymmetric L2: under-estimates (diff > 0) cost tau, over-estimates cost 1 - tau."""
    return (torch.abs(tau - (diff < 0).float()) * diff ** 2).mean()


def train_iql(data: dict, steps: int = 50000, batch: int = 512, gamma: float = 0.99, tau: float = 0.7,
              lr: float = 3e-4, seed: int = 0, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu, sd = data["s"].mean(0), data["s"].std(0) + 1e-3                     # normalise the state
    t = {k: torch.as_tensor(v, device=device) for k, v in data.items() if k in ("s", "a", "r", "d", "s2")}
    t["s"], t["s2"] = (t["s"] - torch.as_tensor(mu, device=device)) / torch.as_tensor(sd, device=device), \
                      (t["s2"] - torch.as_tensor(mu, device=device)) / torch.as_tensor(sd, device=device)
    ds, da = t["s"].shape[1], t["a"].shape[1]
    q1, q2, v = mlp(ds + da, 1).to(device), mlp(ds + da, 1).to(device), mlp(ds, 1).to(device)
    q1_t, q2_t = mlp(ds + da, 1).to(device), mlp(ds + da, 1).to(device)     # slow target copies of Q
    q1_t.load_state_dict(q1.state_dict()); q2_t.load_state_dict(q2.state_dict())
    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)
    opt_v = torch.optim.Adam(v.parameters(), lr=lr)
    N = len(t["s"])
    for step in range(steps):
        i = torch.as_tensor(rng.integers(0, N, batch), device=device)
        s, a, r, d, s2 = (t[k][i] for k in ("s", "a", "r", "d", "s2"))
        sa = torch.cat([s, a], 1)
        with torch.no_grad():
            q_min = torch.min(q1_t(sa), q2_t(sa)).squeeze(-1)
        loss_v = expectile_loss(q_min - v(s).squeeze(-1), tau)
        opt_v.zero_grad(); loss_v.backward(); opt_v.step()
        with torch.no_grad():
            target = r + gamma * (1 - d) * v(s2).squeeze(-1)
        loss_q = F.mse_loss(q1(sa).squeeze(-1), target) + F.mse_loss(q2(sa).squeeze(-1), target)
        opt_q.zero_grad(); loss_q.backward(); opt_q.step()
        with torch.no_grad():                                                  # Polyak averaging, rate 0.005
            for net, tgt in ((q1, q1_t), (q2, q2_t)):
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(0.995).add_(0.005 * p)
        if step % 5000 == 0:
            print(f"step {step}: V loss {loss_v.item():.5f}  Q loss {loss_q.item():.5f}", flush=True)
    with torch.no_grad():
        sa = torch.cat([t["s"], t["a"]], 1)
        Q = torch.min(q1_t(sa), q2_t(sa)).squeeze(-1).cpu().numpy()
        V = v(t["s"]).squeeze(-1).cpu().numpy()
    return Q, V


def advantage_weights(A: np.ndarray, beta: float = 1.0, clip: float = 20.0) -> np.ndarray:
    w = np.minimum(np.exp(beta * (A - A.mean()) / (A.std() + 1e-8)), clip)   # beta on standardised advantages
    return (w / w.mean()).astype(np.float32)


def diagnostics(data: dict, V: np.ndarray, A: np.ndarray, w: np.ndarray) -> dict:
    """Is the critic informative? The gate for using it at all: V must tell successes from failures."""
    from p2p.wm.evaluate import auroc

    first = np.r_[True, data["ep"][1:] != data["ep"][:-1]]
    mid = np.zeros_like(first)
    for e in np.unique(data["ep"]):
        idx = np.flatnonzero(data["ep"] == e)
        mid[idx[len(idx) // 2]] = True
    succ = data["success"].astype(bool)
    return {"n_frames": int(len(w)), "n_episodes": int(first.sum()), "success_episodes": int(succ[first].sum()),
            "auroc_V_first_frame": auroc(V[first], succ[first]), "auroc_V_mid_episode": auroc(V[mid], succ[mid]),
            "mean_weight_success_frames": float(w[succ].mean()), "mean_weight_failure_frames": float(w[~succ].mean()),
            "weight_p99": float(np.percentile(w, 99)), "effective_sample_fraction": float(w.sum() ** 2 / (w ** 2).sum() / len(w))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.7)
    a = ap.parse_args()
    data = load_transitions(Path(a.gen_dir))
    Q, V = train_iql(data, a.steps, tau=a.tau)
    A = Q - V
    w = advantage_weights(A, a.beta)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "weights.npz", weights=w, advantage=A.astype(np.float32), value=V.astype(np.float32))
    diag = diagnostics(data, V, A, w)
    (out / "iql_diagnostics.json").write_text(json.dumps(diag, indent=1))
    print(json.dumps(diag, indent=1))


if __name__ == "__main__":
    main()
