"""README figures for offline RL and the world model, from saved results (no simulator, no GPU).

    python tools/wm_rl_figures.py

Writes
  assets/offline_rl.png          success of the three ways to use the 600 rollouts, and what the weights changed
  assets/wm_model.png            how the world model imagines: one real frame, then actions only
  assets/wm_prediction.png       image and bowl error vs how far ahead it imagines, with baselines
  assets/wm_auroc.png            success AUROC vs how much of the episode is imagined
  assets/wm_two_uses.png         judging recorded actions vs choosing a plan before acting
  assets/wm_plan_choice.png      per demo: how often its candidates work, how often the world model picked it
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p2p.stats import wilson_ci  # noqa: E402
from tools.policy_failures import kind  # noqa: E402

INK, INK2, MUTED, GRID, SURF, BOX = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb", "#f1f0ec"
BLUE, ORANGE, GREEN, GREY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#d64541"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "font.family": "DejaVu Sans"})
A = ROOT / "results" / "act"


def clean(ax, left=False):
    for sp in ("top", "right") + (("left",) if not left else ()):
        ax.spines[sp].set_visible(False)


def save(fig, name):
    fig.savefig(ROOT / "assets" / name, dpi=170, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("assets/" + name)


# ------------------------------------------------------------------------------------------- offline RL
def offline_rl():
    runs = [("successes only\n(filtered imitation, 438)", "ours_all_s0", GREY),
            ("all 600, unweighted", "ours_mix_s0", BLUE),
            ("all 600, weighted by the critic", "ours_mix_awr_s0", ORANGE)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.3), gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.55})
    modes = {}
    for i, (label, run, col) in enumerate(runs):
        evals = [json.loads((A / run / f).read_text()) for f in ("eval.json", "eval_rerun.json")]
        k, n = sum(e["success"] for e in evals), sum(e["n"] for e in evals)
        lo, hi = wilson_ci(k, n)
        y = len(runs) - 1 - i
        a1.barh(y, 100 * k / n, height=0.55, color=col, alpha=0.9, zorder=2)
        a1.plot([100 * lo, 100 * hi], [y, y], color=INK, lw=1.3, zorder=3)
        for j, e in enumerate(evals):
            a1.scatter(100 * e["success"] / e["n"], y + (0.18 if j else -0.18), s=26, facecolor=SURF, edgecolor=INK,
                       lw=1.1, zorder=4)
        a1.text(100 * hi + 1.5, y, f"{100 * k / n:.0f}%", va="center", fontweight="bold", color=INK)
        eps = [x for e in evals for x in e["episodes"] if "bowl_lift_cm" in x]
        modes[run] = (Counter(kind(x) for x in eps), len(eps))
    a1.set_yticks(range(len(runs)), [r[0] for r in runs][::-1], fontsize=9, color=INK)
    a1.set_xlim(0, 100); a1.set_xticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    a1.scatter([], [], s=26, facecolor=SURF, edgecolor=INK, lw=1.1, label="each of the two evaluations")
    a1.legend(loc="lower right", frameon=False, fontsize=8, bbox_to_anchor=(1.02, -0.3))
    a1.set_xlabel("success, pooled over 2 x 50 episodes (line: 95% interval)", fontsize=8.5)
    a1.grid(axis="x", color=GRID, lw=0.6); a1.set_axisbelow(True); clean(a1); a1.tick_params(axis="y", length=0)
    a1.set_title("Success: no measurable difference", loc="left", fontsize=10.5, fontweight="bold", color=INK)

    cats = [("never lifted the bowl", "never lifted"), ("lifted, left off the plate", "left off plate"),
            ("on the plate, > 3 cm off centre", "on plate, > 3 cm"), ("success", "success")]
    (cu, nu), (cw, nw) = modes["ours_mix_s0"], modes["ours_mix_awr_s0"]
    assert nu == nw == 100
    for i, (label, key) in enumerate(cats):
        y = len(cats) - 1 - i
        u, w = cu[key], cw[key]
        a2.plot([u, w], [y, y], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        a2.scatter(u, y, s=46, color=BLUE, zorder=3)
        a2.scatter(w, y, s=46, color=ORANGE, zorder=3)
        a2.text(max(u, w) + 3, y, f"{u} -> {w}", va="center", fontsize=8.5, color=INK2)
    a2.set_yticks(range(len(cats)), [c[0] for c in cats][::-1], fontsize=9, color=INK)
    a2.set_xlim(0, 80); a2.set_ylim(-0.6, len(cats) - 0.4)
    a2.set_xlabel("episodes out of 100 (2 evaluations x 50 scenes)", fontsize=8.5)
    a2.scatter([], [], s=46, color=BLUE, label="unweighted"); a2.scatter([], [], s=46, color=ORANGE, label="weighted")
    a2.legend(loc="lower right", frameon=False, fontsize=8, ncol=2, bbox_to_anchor=(1.02, -0.3))
    a2.grid(axis="x", color=GRID, lw=0.6); a2.set_axisbelow(True); clean(a2); a2.tick_params(axis="y", length=0)
    a2.set_title("Same 600 rollouts: what the weights changed", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    save(fig, "offline_rl.png")


# ------------------------------------------------------------------------------------------- diagrams
def box(ax, x, y, w, h, text, fc=BOX, ec=MUTED, fs=9, bold=False, color=INK, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.08", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(ax, p, q, color=MUTED, lw=1.2, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def wm_frames():
    """Tiles of assets/wm_hero_success.mp4 (a held-out demo): real | imagined, one tile per 0.1 s."""
    import imageio.v3 as iio
    v = iio.imread(ROOT / "assets" / "wm_hero_success.mp4")
    w = v.shape[2] // 2
    return [f[:, :w] for f in v], [f[:, w:] for f in v]


def wm_model():
    real, imag = wm_frames()
    T = len(imag) - 1
    mid = 30
    fig, ax = plt.subplots(figsize=(10.4, 3.9))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 3.9); ax.axis("off")
    ax.imshow(real[0], extent=(0.1, 1.1, 2.15, 3.15))
    ax.text(0.6, 3.25, "one real frame\n(64 x 64 px)", ha="center", va="bottom", fontsize=8.5, color=INK2)
    box(ax, 1.35, 2.35, 0.95, 0.6, "encoder", fs=8.5)
    arrow(ax, (1.1, 2.65), (1.35, 2.65))
    xs = [2.75, 4.65, 6.55, 8.45]
    names = ["z$_0$", "z$_1$", "z$_t$", "z$_T$"]
    shown = [None, 1, mid, T]
    for j, x in enumerate(xs):
        box(ax, x, 2.35, 0.75, 0.6, names[j], fs=10, bold=True, fc="#e3eefb", ec=BLUE)
        if j == 0:
            arrow(ax, (2.3, 2.65), (x, 2.65))
        if j < 3:
            nx = xs[j + 1]
            box(ax, x + 0.95, 2.4, 0.75, 0.5, "GRU", fs=8.5)
            arrow(ax, (x + 0.75, 2.65), (x + 0.95, 2.65))
            arrow(ax, (x + 1.7, 2.65), (nx, 2.65), ls="--" if j else "-")
            ax.text(x + 1.32, 3.55, "next 2 actions\n(0.1 s)", ha="center", va="center", fontsize=7.5, color=ORANGE)
            arrow(ax, (x + 1.32, 3.35), (x + 1.32, 2.9), color=ORANGE)
        if shown[j] is not None:
            ax.imshow(imag[shown[j]], extent=(x - 0.1, x + 0.85, 0.7, 1.65))
            arrow(ax, (x + 0.375, 2.35), (x + 0.375, 1.67))
            ax.text(x + 0.375, 0.6, "imagined frame\n+ bowl (x, y, z)", ha="center", va="top", fontsize=7.5, color=INK2)
    ax.text(0.1, 1.6, "Only the first frame is real.\nEach later summary z is\npredicted from the previous\none and the next actions,\nso the model imagines the\nwhole episode without\nseeing another image.",
            fontsize=8.3, color=INK, va="top", linespacing=1.35)
    ax.text(9.35, 2.65, "success?\nimagined bowl at\nthe end within\n3.2 cm of the\nplate centre", fontsize=8, color=INK2, va="center")
    save(fig, "wm_model.png")


def wm_two_uses():
    fig, ax = plt.subplots(figsize=(10.4, 4.3))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 4.3); ax.axis("off")
    rows = [(3.0, "A   Judge an action sequence it is given", "the actions the robot\nreally executed",
             None, "on demos it never saw: tells successes\nfrom failures (AUROC 0.93) and ranks\nthe three retargeting strategies correctly", GREEN),
            (1.05, "B   Choose a plan before acting (notebook 07)", "13 candidate plans:\nmy demos, retargeted",
             "arm model turns each\nplan into actions:\nno bowl, cabinet or contacts", "its pick works in 78% of scenes:\nrandom 71%, always the most\nreliable demo 98%", RED)]
    for y, title, src, arm, result, col in rows:
        ax.text(0.05, y + 0.78, title, fontsize=10.5, fontweight="bold", color=INK)
        box(ax, 0.05, y - 0.05, 1.95, 0.7, src, fs=8.3)
        if arm:
            box(ax, 2.35, y - 0.12, 2.15, 0.84, arm, fs=8, fc="#fbeceb", ec=RED)
            arrow(ax, (2.0, y + 0.3), (2.35, y + 0.3))
            arrow(ax, (4.5, y + 0.3), (5.05, y + 0.3))
        else:
            ax.text(3.55, y + 0.42, "first frame + actions", fontsize=7.5, color=INK2, ha="center")
            arrow(ax, (2.0, y + 0.3), (5.05, y + 0.3))
        box(ax, 5.05, y - 0.05, 1.75, 0.7, "world model\nimagines the episode", fs=8.3, fc="#e3eefb", ec=BLUE)
        arrow(ax, (6.8, y + 0.3), (7.1, y + 0.3))
        box(ax, 7.1, y - 0.18, 3.25, 0.96, result, fs=8.2, ec=col, lw=1.6)
    ax.text(0.05, 0.12, "Why B fails: the waypoint follower steers by where the gripper really is. When the arm stalls (against the cabinet,\n"
            "or on an awkward approach), the executed actions change; the planned ones do not, so the world model imagines a clean run.",
            fontsize=8.2, color=INK2, va="bottom", linespacing=1.35)
    save(fig, "wm_two_uses.png")


# ------------------------------------------------------------------------------------------- plots
def wm_prediction():
    h = json.loads((ROOT / "results" / "wm" / "wm_metrics.json").read_text())["horizon"]
    t = np.array(h["horizon_steps"]) / 10
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.4), gridspec_kw={"wspace": 0.28})
    a1.plot(t, h["psnr_ae"], color=GREY, lw=1.4, ls=":", label="ceiling: real frame through encoder and decoder")
    a1.plot(t, h["psnr_copy"], color=ORANGE, lw=1.6, ls="--", label='baseline "nothing moves": the start frame')
    a1.plot(t, h["psnr"], color=BLUE, lw=2.2, label="world model, imagining")
    cross = t[np.argmax(np.array(h["psnr"]) > np.array(h["psnr_copy"]))]
    a1.axvline(cross, color=GRID, lw=1)
    a1.text(cross + 0.05, 37.5, f"better than\n\"nothing moves\"\nfrom {cross:.1f} s on", fontsize=8, color=INK2, va="top")
    a1.set_ylim(22, 40); a1.set_xlim(0, 4.05)
    a1.set_xlabel("seconds imagined ahead"); a1.set_ylabel("image similarity, PSNR (dB)")
    a1.legend(frameon=False, fontsize=7.8, loc="lower left")
    a1.set_title("Image: blurry, but it predicts the motion", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    a2.axhspan(0, 3, color=GREEN, alpha=0.10, lw=0)
    a2.text(0.15, 1.5, "LIBERO's success tolerance: 3 cm", fontsize=8, color="#127a55", va="center")
    a2.plot(t[1:], 100 * np.array(h["bowl_err_static_moving"][1:]), color=ORANGE, lw=1.6, ls="--", label='baseline "the bowl stays put"')
    a2.plot(t[1:], 100 * np.array(h["bowl_err_moving"][1:]), color=BLUE, lw=2.2, label="world model, imagining")
    a2.set_ylim(0, 12); a2.set_xlim(0, 4.05)
    a2.set_xlabel("seconds imagined ahead"); a2.set_ylabel("bowl position error (cm)")
    a2.legend(frameon=False, fontsize=7.8, loc="upper left")
    a2.set_title("Bowl: follows the gripper, but coarse", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    for a in (a1, a2):
        a.grid(color=GRID, lw=0.6); a.set_axisbelow(True); clean(a, left=True)
    save(fig, "wm_prediction.png")


def wm_auroc():
    s = json.loads((ROOT / "results" / "wm" / "within_strategy_auroc.json").read_text())
    keys = ["0", "5", "10", "20", "40", "80", "full"]
    labels = ["0 s", "0.5 s", "1 s", "2 s", "4 s", "8 s", "whole\nepisode"]
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    ax.plot(range(7), [s["all_strategies"]["auroc"][k] for k in keys], "o-", color=GREY, lw=1.6, ms=5,
            label="all three strategies (n=180)")
    ax.plot(range(7), [s["object_relative"]["auroc"][k] for k in keys], "o-", color=BLUE, lw=2, ms=5,
            label="object-relative only (n=60), the harder test")
    ax.axhline(0.5, color=MUTED, lw=1, ls="--"); ax.text(0.05, 0.52, "chance", fontsize=8, color=INK2)
    ax.set_xticks(range(7), labels); ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("how much of the episode is imagined (before its end)"); ax.set_ylabel("success AUROC")
    ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.grid(color=GRID, lw=0.6); clean(ax, left=True)
    save(fig, "wm_auroc.png")


def wm_plan_choice():
    rows = list(csv.DictReader(open(ROOT / "results" / "plan_select" / "plan_select_candidates.csv")))
    by_init = {}
    for r in rows:
        by_init.setdefault(r["init"], []).append(r)
    picks = Counter()
    picks_ok = Counter()
    for cands in by_init.values():
        c = min(cands, key=lambda r: float(r["wm_dist_nominal_cm"]))
        picks[c["demo"]] += 1
        picks_ok[c["demo"]] += c["success"] == "True"
    demos = sorted({r["demo"] for r in rows}, key=lambda d: np.mean([r["success"] == "True" for r in rows if r["demo"] == d]))
    rate = [np.mean([r["success"] == "True" for r in rows if r["demo"] == d]) for d in demos]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True, gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.08})
    y = np.arange(len(demos))
    a1.barh(y, [100 * r for r in rate], color=GREY, alpha=0.55, height=0.6)
    for yi, r in zip(y, rate):
        a1.text(100 * r + 1.5, yi, f"{100 * r:.0f}%", va="center", fontsize=8, color=INK2)
    a1.set_xlim(0, 112); a1.set_xticks([0, 50, 100], ["0%", "50%", "100%"])
    a1.set_yticks(y, [d[:5] if d.startswith("a4558") else d[:4] for d in demos], fontsize=8.5,
                  family="DejaVu Sans Mono")
    a1.set_xlabel("its candidate succeeds in the simulator (50 scenes)", fontsize=8.5)
    a1.set_title("How good each demo's plan is", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    ok = [picks_ok[d] for d in demos]
    bad = [picks[d] - picks_ok[d] for d in demos]
    a2.barh(y, ok, color=GREEN, height=0.6, label="picked, succeeded")
    a2.barh(y, bad, left=ok, color=RED, height=0.6, label="picked, failed")
    for yi, d in zip(y, demos):
        if picks[d]:
            a2.text(picks[d] + 0.3, yi, str(picks[d]), va="center", fontsize=8, color=INK2)
    a2.set_xlim(0, 16)
    a2.set_xlabel("scenes in which the world model picked it (of 50)", fontsize=8.5)
    a2.legend(frameon=False, fontsize=8, loc="lower right")
    a2.set_title("How often the world model chose it", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    for a in (a1, a2):
        a.grid(axis="x", color=GRID, lw=0.6); a.set_axisbelow(True); clean(a); a.tick_params(axis="y", length=0)
    save(fig, "wm_plan_choice.png")


if __name__ == "__main__":
    offline_rl()
    wm_model()
    wm_two_uses()
    wm_prediction()
    wm_auroc()
    wm_plan_choice()
