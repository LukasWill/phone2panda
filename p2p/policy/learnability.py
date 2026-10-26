"""Are the demonstrations learnable? One command, one trained ACT policy.

    python -m p2p.policy.learnability CHECKPOINT [--episodes 50]     # GPU: two evaluations, then the verdict
    python -m p2p.policy.learnability results/act/*/                  # no simulator: verdicts from saved evaluations

The check (README, "A quick check: are the demos learnable?"): evaluate the policy as trained (execute 25 of
each 50 predicted actions, then predict again), then again with temporal ensembling (predict at every step and
execute the average of all overlapping predictions for the current step; ACT's original inference mode). If the
demos are consistent, the overlapping predictions agree and averaging costs little. If the demos teach
conflicting moves, the predictions disagree and their average is a move nobody demonstrated, so success drops.

Verdict, from the share of the normal success rate that survives averaging (rule of thumb read off our 12 runs:
consistent data kept 93-103%, conflicting data 21-76%):
  keeps >= 90%   consistent
  keeps <  80%   conflicting moves
  otherwise      unclear
  normal < 20%   policy too weak to judge
The interval on the drop says whether the drop exceeds evaluation noise at this many episodes; when it does not,
evaluate more episodes before trusting the verdict. The check says THAT something is inconsistent, not WHAT:
recording where the gripper closes (p2p/policy/diagnose.py, in every eval.json) is how we found the cause.

A CHECKPOINT is a local folder written by `p2p.policy.act train` (e.g. runs/ours_all_s0/final) or a Hub id.
Its evaluations are written to --out (default: CHECKPOINT/../learnability) as eval.json, eval_te.json and
learnability.json. A results folder is any folder holding evaluations from `p2p.policy.act eval`, flat
(eval.json, eval_rerun.json, eval_te.json, as in results/act/) or as written by the notebooks
(eval/eval.json, eval_rerun/eval.json, eval_te/eval.json).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from p2p.stats import diff_ci, fmt_rate

TE_COEFF = 0.01                     # the original ACT paper's weighting of the overlapping predictions
KEEP_OK, KEEP_BAD, MIN_RATE = 0.9, 0.8, 0.2


def verdict(k_norm: int, n_norm: int, k_avg: int, n_avg: int) -> dict:
    p_norm, p_avg = k_norm / n_norm, k_avg / n_avg
    lo, hi = diff_ci(k_norm, n_norm, k_avg, n_avg)
    keeps = p_avg / p_norm if p_norm else float("nan")
    if p_norm < MIN_RATE:
        v = "policy too weak to judge"
    elif keeps >= KEEP_OK:
        v = "consistent"
    elif keeps < KEEP_BAD:
        v = "conflicting moves"
    else:
        v = "unclear"
    beyond_noise = lo > 0 or hi < 0
    return {"normal": fmt_rate(k_norm, n_norm), "averaged": fmt_rate(k_avg, n_avg),
            "drop_points": round(100 * (p_norm - p_avg)), "drop_ci": [round(100 * lo), round(100 * hi)],
            "keeps": round(keeps, 2), "verdict": v, "drop_beyond_noise": beyond_noise}


def explain(r: dict) -> str:
    s = (f"normal {r['normal']}, averaged {r['averaged']}: drop {r['drop_points']:+d} points "
         f"[{r['drop_ci'][0]} to {r['drop_ci'][1]}], keeps {100 * r['keeps']:.0f}% -> {r['verdict'].upper()}")
    if r["verdict"] in ("conflicting moves", "unclear") and not r["drop_beyond_noise"]:
        s += " (drop within evaluation noise at this many episodes: evaluate more before trusting it)"
    return s


# ---------------------------------------------------------------- saved evaluations
def _evals(folder: Path) -> dict[str, dict]:
    """name -> summary, for every evaluation in a folder (flat files or one sub-folder per evaluation)."""
    out = {}
    for f in sorted(folder.glob("*.json")) + sorted(folder.glob("*/eval.json")):
        try:
            s = json.loads(f.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(s, dict) and "episodes" in s and "success" in s:
            out[f.parent.name if f.name == "eval.json" and f.parent != folder else f.stem] = s
    return out


def from_results(folder: Path, normal_steps: int = 25) -> list[dict]:
    """Pair every temporal-ensembling evaluation X_te with the normal evaluations named X* in the same folder
    (eval_te -> eval, eval_rerun; eval_60k_te -> eval_60k_a, eval_60k_b) and pool the normal ones."""
    ev = _evals(folder)
    rows = []
    for name, s in ev.items():
        if not str(s.get("execution", "")).startswith("temporal_ensemble"):
            continue
        prefix = re.sub(r"_?te$", "", name)
        norm = [n for n, t in ev.items() if n.startswith(prefix) and n != name
                and t.get("execution", f"n_action_steps={normal_steps}") == f"n_action_steps={normal_steps}"]
        if not norm:
            continue
        k, n = sum(ev[x]["success"] for x in norm), sum(ev[x]["n"] for x in norm)
        rows.append({"run": folder.name, "normal_evals": norm, "averaged_eval": name,
                     **verdict(k, n, s["success"], s["n"])})
    return rows


# ---------------------------------------------------------------- a checkpoint (needs the simulator)
def from_checkpoint(ckpt: str, episodes: int, device: str, out: Path, normal: str | None) -> dict:
    from p2p.policy.act import evaluate

    out.mkdir(parents=True, exist_ok=True)
    if normal:
        s_norm = json.loads(Path(normal).read_text())
        print(f"normal execution: reusing {normal} ({s_norm['rate']})")
    else:
        print("normal execution (as trained) ...")
        s_norm = evaluate(ckpt, range(episodes), device, video_dir=None)
        (out / "eval.json").write_text(json.dumps(s_norm, indent=1))
    print(f"predictions averaged (temporal ensembling, {TE_COEFF}) ...")
    s_avg = evaluate(ckpt, range(episodes), device, video_dir=None, temporal_ensemble=TE_COEFF)
    (out / "eval_te.json").write_text(json.dumps(s_avg, indent=1))
    r = {"checkpoint": str(ckpt), "normal_execution": s_norm["execution"], "averaged_execution": s_avg["execution"],
         **verdict(s_norm["success"], s_norm["n"], s_avg["success"], s_avg["n"])}
    (out / "learnability.json").write_text(json.dumps(r, indent=1))
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("paths", nargs="+", help="checkpoint (folder or Hub id) or results folder(s)")
    ap.add_argument("--episodes", type=int, default=50, help="evaluation scenes per execution mode")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="where a checkpoint's evaluations go")
    ap.add_argument("--normal", default=None, help="reuse an existing normal eval.json of this checkpoint")
    a = ap.parse_args(argv)

    rows = []
    for p in a.paths:
        path = Path(p)
        if path.is_dir() and not (path / "config.json").exists():
            found = from_results(path)
            if not found:
                print(f"{p}: no temporal-ensembling evaluation with a matching normal one, skipped", file=sys.stderr)
            rows += found
        else:
            device = a.device
            if device is None:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            out = Path(a.out) if a.out else (path.parent / "learnability" if path.is_dir() else Path("learnability"))
            r = from_checkpoint(p, a.episodes, device, out, a.normal)
            print(explain(r))
            print(f"-> {out / 'learnability.json'}")
            return
    if len(rows) == 1:
        print(f"{rows[0]['run']}: {explain(rows[0])}")
    elif rows:
        print("| run | normal | averaged | drop (points) [95%] | keeps | verdict |")
        print("|---|---|---|---|---|---|")
        for r in rows:
            flag = "" if r["drop_beyond_noise"] or r["verdict"] == "consistent" else " (within noise)"
            print(f"| {r['run']} | {r['normal']} | {r['averaged']} | {r['drop_points']:+d} "
                  f"[{r['drop_ci'][0]} to {r['drop_ci'][1]}] | {100 * r['keeps']:.0f}% | {r['verdict']}{flag} |")


if __name__ == "__main__":
    main()
