"""Plot assets/approach_shape_vs_execution.png from tools/approach_figure_collect.py output."""
import pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

T = pickle.load(open(sys.argv[1], 'rb'))
OUT = sys.argv[2]
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
STRATS = [("object_relative", "my approach (copied)", "#2a78d6"),
          ("object_relative_timed", "same path, fixed pace", "#eb6834"),
          ("object_relative_vertical", "planned: straight down", "#1baf7a")]
RIM_R, BOWL_H = 5.57, 5.26
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "font.family": "DejaVu Sans"})

def get(init, demo, strat):
    return [t for t in T if t["init"] == init and t["demo"].startswith(demo) and t["strat"] == strat][0]

def close_idx(t):
    return int(np.flatnonzero(t["grip"] > 0)[0])

fig = plt.figure(figsize=(12, 8.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1], hspace=0.42, wspace=0.28)

# ---- A: side view (shape), one demo, one scene
DEMO, INIT = "92e0", 0
ax = fig.add_subplot(gs[0, :])
# schematic bowl: outer wall from rim top down to the base, mirrored for the far side
for sgn in (1, -1):
    ax.plot([sgn * RIM_R, sgn * 3.4], [0, -BOWL_H], color="#b9b8b2", lw=6, solid_capstyle="round", zorder=1)
ax.plot([-3.4, 3.4], [-BOWL_H, -BOWL_H], color="#b9b8b2", lw=6, solid_capstyle="round", zorder=1)
ax.text(-RIM_R + 0.2, -BOWL_H - 1.2, "bowl (cross-section, schematic)", color=MUTED, fontsize=9)
ax.axvline(0, color=GRID, lw=1, ls="--", zorder=0)
ax.text(0.2, 11.2, "bowl's centre line", color=MUTED, fontsize=9)
for strat, label, col in STRATS:
    t = get(INIT, DEMO, strat)
    c = close_idx(t)
    rel = t["eef"][: c + 1] - t["bowl"][: c + 1]
    gdir = (t["grasp"] - t["bowl"][0])[:2]
    gdir /= np.linalg.norm(gdir)
    x = 100 * rel[:, :2] @ gdir                          # signed distance along the grasp direction
    z = 100 * (t["eef"][: c + 1, 2] - t["rim_z"])
    keep = (z < 11) & (np.abs(x) < 16)
    first = np.flatnonzero(keep)[0]
    ls = "--" if strat == "object_relative_timed" else "-"
    ax.plot(x[first:], z[first:], color=col, lw=2.2, ls=ls, zorder=3 if ls == "-" else 4)
    ax.plot(x[-1], z[-1], "o", ms=9, color=col, mec=SURF, mew=2, zorder=5)
lab = {"object_relative": (8.6, -3.4, "my approach (copied): closes at 7.8 cm"),
       "object_relative_timed": (13.4, 6.0, "same path, fixed pace (dashed):\nends further in, closes at 4.1 cm"),
       "object_relative_vertical": (5.0, 10.2, "planned: straight down, closes at 4.2 cm")}
for strat, label, col in STRATS:
    x, y, txt = lab[strat]
    ax.text(x, y, txt, color=INK, fontsize=9.5, va="center")
    ax.plot([x - 0.5], [y], "s", ms=7, color=col)
ax.text(-15.5, 9.8, "dots: where the gripper closes\nrim (outer edge) at 5.6 cm from the centre line", color=INK2, fontsize=9, va="top")
ax.annotate("comes in sideways,\nabout 1 cm above the rim", xy=(9.5, 1.0), xytext=(15.5, 1.6), color=INK2, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8), va="center")
ax.set_xlim(-16, 24); ax.set_ylim(-7, 12)
ax.set_aspect("equal")
ax.set_xlabel("horizontal distance from the bowl's centre line, towards the grasp point (cm)")
ax.set_ylabel("height above the rim (cm)")
ax.set_title("Path shape: where the gripper goes (one demo, one scene, side view)", loc="left")
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

# ---- C: execution / timing, 5 demos x 3 scenes per panel
axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
for ax, (strat, label, col) in zip(axes, STRATS):
    closes, late = [], 0
    for t in T:
        if t["strat"] != strat:
            continue
        if not (t["grip"] > 0).any():          # still approaching when the recording stopped (8 s)
            late += 1
            d = 100 * np.linalg.norm(t["eef"] - t["grasp"], axis=1)
            ax.plot(np.arange(len(d)) / 20.0, d, color=col, lw=1.2, alpha=0.55, zorder=2)
            continue
        c = close_idx(t)
        d = 100 * np.linalg.norm(t["eef"][: c + 1] - t["grasp"], axis=1)
        s = np.arange(c + 1) / 20.0
        ax.plot(s, d, color=col, lw=1.2, alpha=0.55, zorder=2)
        ax.plot(s[-1], d[-1], "o", ms=8, color=col, mec=SURF, mew=1.5, zorder=3)
        closes.append(s[-1])
    lo, hi = min(closes), max(closes)
    ax.axvspan(lo, hi, color=col, alpha=0.10, lw=0, zorder=0)
    ax.text(0.03, 0.95, label, transform=ax.transAxes, color=INK, fontsize=10, fontweight="bold", va="top")
    when = f"closes at {lo:.1f} s in every run" if hi - lo < 0.1 else f"closes between {lo:.1f} and {hi:.1f} s"
    ax.text(0.03, 0.83, when + (f"\n{late} of 15 still approaching after 4 s" if late else ""),
            transform=ax.transAxes, color=INK2, fontsize=9, va="top")
    ax.set_xlim(0, 4.2); ax.set_ylim(0, 40)
    ax.set_xlabel("time since start (s)")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("gripper's distance to its grasp point (cm)")
fig.text(0.075, 0.47, "Execution: when the gripper gets there (5 demos x 3 scenes per panel; dot = gripper closes)",
         fontsize=11, fontweight="bold", color=INK)
fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=SURF)
print("saved", OUT)
