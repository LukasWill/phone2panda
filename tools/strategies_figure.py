"""assets/retargeting_strategies.png: the three retargeting strategies for one demo in one evaluation scene,
seen from above (physics-only rollouts of the scripted teacher, ~1 min).

    python tools/strategies_figure.py [DEMO] [INIT]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
PATH_OPEN, PATH_HELD = "#52514e", "#6250d6"
BOWL_R, PLATE_R = 5.57, 6.85


def rollouts(demo_name: str, init: int):
    from p2p.human.extract import load
    from p2p.retarget.replay import sim_bowl_mean
    from p2p.retarget.strategies import absolute_alignment, make_plan, to_sim
    from p2p.sim.env import TaskEnv
    from p2p.sim.scripted import WaypointFollower

    demos = [load(p) for p in sorted((ROOT / "data" / "processed").glob("*.npz"))]
    demo = [d for d in demos if d["name"].startswith(demo_name)][0]
    off = absolute_alignment(demos, sim_bowl_mean())
    env = TaskEnv(render=False, hard_reset=False)
    out = {}
    for strat in ("absolute", "object_relative", "intent"):
        env.reset(init_state_id=init)
        s0 = env.scene()
        f = WaypointFollower(make_plan(strat, demo, s0, off))
        E, G, B, done, ok = [], [], [], False, False
        while not done:
            s = env.scene(); a = f.act(s)
            E.append(s.eef_pos); G.append(a[6]); B.append(s.bowl_pos)
            _, ok, done = env.step(a)
        out[strat] = dict(eef=np.array(E), grip=np.array(G), bowl=np.array(B), plate=s0.plate_pos, ok=ok,
                          my_bowl=to_sim(demo["bowl_start"]) + off, my_plate=to_sim(demo["plate_center"]) + off)
    env.close()
    return out


def xy(p):           # top view, robot at the bottom: up = +x (away from the robot), right = -y
    p = np.atleast_2d(p)
    return -100 * p[:, 1], 100 * p[:, 0]


def plot(R, out: Path):
    titles = {"absolute": ("Absolute", "my hand path in fixed coordinates"),
              "object_relative": ("Object-relative", "my path, anchored on the bowl and the plate"),
              "intent": ("Intent", "my grasp side and place point; scripted motion")}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharex=True, sharey=True)
    c = np.r_[xy(R["intent"]["bowl"][0])]
    for ax, (k, r) in zip(axes, R.items()):
        bx, by = xy(r["bowl"][0]); px, py = xy(r["plate"]); ex, ey = xy(r["bowl"][-1])
        ax.add_patch(Circle((px[0], py[0]), PLATE_R, fc="#e6e5e1", ec=MUTED, lw=1, zorder=1))
        ax.text(px[0], py[0] + PLATE_R + 0.8, "plate", ha="center", color=INK2, fontsize=9)
        ax.add_patch(Circle((bx[0], by[0]), BOWL_R, fc="none", ec=INK2, lw=1.2, ls="--", zorder=2))
        ax.text(bx[0] - BOWL_R - 0.8, by[0], "bowl at\nstart", ha="right", va="center", color=INK2, fontsize=9)
        ax.add_patch(Circle((ex[0], ey[0]), BOWL_R, fc="#2b2b2b", alpha=0.18, ec=INK, lw=1, zorder=2))
        if k == "absolute":
            mx, my = xy(r["my_bowl"]); qx, qy = xy(r["my_plate"])
            ax.add_patch(Circle((mx[0], my[0]), BOWL_R, fc="none", ec="#e34948", lw=1.2, ls=":", zorder=2))
            ax.annotate("where my\nbowl was in\nmy recording", xy=(mx[0] + 0.6 * BOWL_R, my[0] + 0.8 * BOWL_R),
                        xytext=(mx[0] + 0.5, my[0] + 11), ha="left", va="bottom", color="#b23a39", fontsize=8.5,
                        arrowprops=dict(arrowstyle="-", color="#b23a39", lw=0.8))
        gx, gy = xy(r["eef"])
        held = r["grip"] > 0
        ax.plot(gx, gy, color=PATH_OPEN, lw=1.3, zorder=3)
        seg = np.where(held, 1, np.nan)
        ax.plot(gx * seg, gy * seg, color=PATH_HELD, lw=3, zorder=4, solid_capstyle="round")
        ax.plot(gx[0], gy[0], "o", ms=5, color=INK, zorder=5)
        verdict = "succeeds" if r["ok"] else "fails"
        ax.text(0, 1.09, f"{titles[k][0]}: {verdict}", transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK, va="bottom")
        ax.text(0, 1.02, titles[k][1], transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
        ax.set_aspect("equal")
        ax.set_xlim(c[0] - 22, c[0] + 22); ax.set_ylim(c[1] - 14, c[1] + 26)
        ax.set_xticks(np.arange(np.floor(c[0] / 5) * 5 - 25, c[0] + 25, 5))
        ax.set_yticks(np.arange(np.floor(c[1] / 5) * 5 - 15, c[1] + 30, 5))
        ax.tick_params(length=0, labelbottom=False, labelleft=False)
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color(GRID)
    axes[0].plot([], [], color=PATH_OPEN, lw=1.3, label="gripper path, open")
    axes[0].plot([], [], color=PATH_HELD, lw=3, label="gripper closed (holding)")
    axes[0].add_patch(Circle((0, 0), 0, fc="#2b2b2b", alpha=0.18, ec=INK, label="bowl at the end"))
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("The same human demo, retargeted three ways into one robot scene (top view, robot at the bottom; grid 5 cm)",
                 x=0.01, ha="left", fontsize=10, color=INK2, y=0.99)
    fig.subplots_adjust(wspace=0.06, top=0.82, bottom=0.1)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURF)


if __name__ == "__main__":
    demo = sys.argv[1] if len(sys.argv) > 1 else "65ee"
    init = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    R = rollouts(demo, init)
    for k, r in R.items():
        print(k, r["ok"], np.round(100 * np.linalg.norm(r["my_bowl"][:2] - r["bowl"][0][:2]), 1))
    plot(R, Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "assets" / "retargeting_strategies.png")
