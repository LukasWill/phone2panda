"""README figures from saved results (no simulator): the key result and where the policies close the gripper.

    python tools/readme_figures.py            # writes assets/key_result.png, assets/where_gripper_closes.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p2p.stats import wilson_ci  # noqa: E402

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
COPIED, FIXED, PLANNED, TELEOP = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "font.family": "DejaVu Sans"})


def pooled(*files: str) -> tuple[int, int]:
    k = n = 0
    for f in files:
        e = json.loads((ROOT / f).read_text())["episodes"]
        k += sum(x["success"] for x in e)
        n += len(e)
    return k, n


def key_result(out: Path):
    A = "results/act/"
    rows = [  # label, files, colour
        ("My whole motion copied\n(10 demos, 438 episodes)", [A + "ours_all_s0/eval.json", A + "ours_all_s0/eval_rerun.json"], COPIED),
        ("My approach copied\n(5 demos, 2 training seeds)", [A + "ours_consistent_clean_s0/eval.json", A + "ours_consistent_clean_s0/eval_rerun.json",
                                                              A + "ours_consistent_clean_s1/eval.json", A + "ours_consistent_clean_s1/eval_rerun.json"], COPIED),
        ("Same path, walked at a fixed pace\n(5 demos)", [A + "ours_consistent_timed_s0/eval.json", A + "ours_consistent_timed_s0/eval_rerun.json"], FIXED),
        ("Approach planned straight down\n(5 demos, 2 training seeds)", [A + "ours_consistent_vertical_s0/eval.json", A + "ours_consistent_vertical_s0/eval_rerun.json",
                                                                          A + "ours_consistent_vertical_s1/eval.json", A + "ours_consistent_vertical_s1/eval_rerun.json"], PLANNED),
        ("Approach planned straight down\n(all 10 demos, 438 episodes)", [A + "ours_all_vertical_s0/eval.json", A + "ours_all_vertical_s0/eval_rerun.json"], PLANNED),
    ]
    tk, tn = pooled(A + "official_s0/eval.json", A + "official_s0/eval_rerun.json")
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    for i, (label, files, col) in enumerate(rows):
        k, n = pooled(*files)
        p, (lo, hi) = k / n, wilson_ci(k, n)
        y = len(rows) - 1 - i
        ax.barh(y, 100 * p, height=0.58, color=col, zorder=2)
        ax.plot([100 * lo, 100 * hi], [y, y], color=INK, lw=1.4, zorder=3)
        ax.text(100 * hi + 1.2, y, f"{100 * p:.0f}%", va="center", color=INK, fontsize=10, fontweight="bold")
    tp, (tlo, thi) = tk / tn, wilson_ci(tk, tn)
    ax.axvspan(100 * tlo, 100 * thi, color=TELEOP, alpha=0.18, lw=0, zorder=1)
    ax.axvline(100 * tp, color=TELEOP, lw=1.5, ls="--", zorder=1)
    ax.text(100 * tlo - 1, len(rows) - 1, f"policy trained on\n49 teleoperated\ndemos: {100 * tp:.0f}%",
            ha="right", va="center", color=INK2, fontsize=8.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=9.5, color=INK)
    ax.set_xlim(0, 108); ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax.set_xlabel("policy success on LIBERO's 50 evaluation scenes (bars: pooled evaluations; lines: 95% Wilson intervals)")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("What to take from a human video: the contact, not the approach", loc="left",
                 fontsize=11.5, fontweight="bold", color=INK, pad=10)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURF)


def where_closes(out: Path):
    A = "results/act/"
    groups = [
        ("policy from 49 teleop demos", ["results/behaviour/official_s0.json"], TELEOP),
        ("policy from my copied approach", ["results/behaviour/ours_consistent_clean_s0.json",
                                            A + "ours_consistent_clean_s1/eval.json", A + "ours_consistent_clean_s1/eval_rerun.json"], COPIED),
        ("same path, fixed pace", [A + "ours_consistent_timed_s0/eval.json", A + "ours_consistent_timed_s0/eval_rerun.json"], FIXED),
        ("policy from the planned approach", [A + "ours_consistent_vertical_s0/eval.json", A + "ours_consistent_vertical_s0/eval_rerun.json",
                                              A + "ours_consistent_vertical_s1/eval.json", A + "ours_consistent_vertical_s1/eval_rerun.json"], PLANNED),
    ]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.axvline(5.57, color=MUTED, lw=1.2, zorder=1)
    ax.text(5.7, len(groups) - 0.45, "rim (5.6 cm)", color=INK2, fontsize=9, va="bottom")
    for i, (label, files, col) in enumerate(groups):
        eps = [e for f in files for e in json.loads((ROOT / f).read_text())["episodes"] if "close_r_cm" in e]
        y = len(groups) - 1 - i
        r = np.array([min(e["close_r_cm"], 13.8) for e in eps])
        ok = np.array([e["success"] for e in eps])
        jit = rng.uniform(-0.22, 0.22, len(r))
        ax.scatter(r[ok], y + jit[ok], s=16, color=col, alpha=0.75, lw=0, zorder=3)
        ax.scatter(r[~ok], y + jit[~ok], s=30, color=INK, marker="x", lw=1.2, zorder=4)
        txt = f"{ok.mean():.0%} success"
        if (~ok).sum() >= 8:
            txt += f"\nfailures: median {np.median(r[~ok]):.1f} cm"
        ax.text(13.9, y, txt, va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(groups)), [g[0] for g in groups][::-1], fontsize=9.5, color=INK)
    ax.set_xlim(0, 17.8); ax.set_ylim(-0.6, len(groups) - 0.05)
    ax.set_xticks(range(0, 15, 2))
    ax.set_xlabel("where the gripper first closes: horizontal distance from the bowl's centre line (cm)")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.scatter([], [], s=16, color=INK2, label="episode succeeded"); ax.scatter([], [], s=30, color=INK, marker="x", label="episode failed")
    ax.legend(loc="lower right", bbox_to_anchor=(0.78, -0.02), frameon=False, fontsize=8.5)
    ax.set_title("How the copied approach fails: the gripper closes too far from the bowl", loc="left",
                 fontsize=11.5, fontweight="bold", color=INK, pad=10)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURF)


if __name__ == "__main__":
    key_result(ROOT / "assets" / "key_result.png")
    where_closes(ROOT / "assets" / "where_gripper_closes.png")
    print("ok")
