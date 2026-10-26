"""assets/pipeline.png: the pipeline as one picture (boxes, arrows, a thumbnail per stage).

    python tools/pipeline_figure.py assets/phone/frame_3.jpg SIM_FRAME WM_FRAME   (a robot-camera frame, an imagined frame)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
INK, INK2, MUTED, SURF, BOX = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#f1f0ec"
ACCENT = "#1baf7a"


def img(path):
    return plt.imread(path)


def box(ax, x, y, w, h, title, body, code, accent=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                                fc=BOX, ec=ACCENT if accent else "#d6d5cf", lw=1.6 if accent else 1.0, zorder=1))
    ax.text(x + 0.012, y + h - 0.03, title, fontsize=10.5, fontweight="bold", color=INK, va="top", zorder=2)
    ax.text(x + 0.012, y + h - 0.085, body, fontsize=8.6, color=INK2, va="top", zorder=2, linespacing=1.35)
    ax.text(x + 0.012, y + 0.02, code, fontsize=8, color=MUTED, va="bottom", family="DejaVu Sans Mono", zorder=2)


def thumb(fig, rect, im, label=None):
    a = fig.add_axes(rect); a.imshow(im); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color("#d6d5cf")
    if label:
        a.set_title(label, fontsize=8, color=INK2, pad=2)
    return a


def demo_thumb(fig, rect):
    from p2p.human.extract import load
    d = load(ROOT / "data" / "processed" / "92e0b86bc6_lefthand.npz")
    a = fig.add_axes(rect)
    P = d["pinch"]; ok = np.isfinite(P[:, 0])
    a.plot(100 * P[ok, 0], 100 * P[ok, 1], color="#2a78d6", lw=1.2)
    for c, r, col in ((d["bowl_start"], 5.0, INK2), (d["plate_center"], 9.9, MUTED)):
        a.add_patch(Circle((100 * c[0], 100 * c[1]), r, fill=False, ec=col, lw=1.2, ls="-" if col == INK2 else "--"))
    gp = d["grasp_point"]; a.plot(100 * gp[0], 100 * gp[1], "o", ms=5, color="#eb6834")
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_xlim(100 * min(P[ok, 0].min(), d["plate_center"][0]) - 12, 100 * max(P[ok, 0].max(), d["plate_center"][0]) + 12)
    a.set_ylim(100 * min(P[ok, 1].min(), d["plate_center"][1]) - 12, 100 * max(P[ok, 1].max(), d["plate_center"][1]) + 12)
    for sp in a.spines.values():
        sp.set_color("#d6d5cf")
    a.set_title("hand path, grasp point, bowl, plate", fontsize=8, color=INK2, pad=2)


def retarget_thumb(fig, rect):
    """Side view: bowl (left), plate (right); my carry is kept, my sideways approach is replaced by a planned
    straight-down approach."""
    a = fig.add_axes(rect)
    a.plot([-5.6, -3.4, 3.4, 5.6], [0, -5.3, -5.3, 0], color="#b9b8b2", lw=4, solid_capstyle="round")   # bowl
    a.plot([16, 30], [-5.3, -5.3], color=MUTED, lw=4, solid_capstyle="round")                          # plate
    a.plot([-17, -9, -6.5, -5.0], [1.2, 1.2, 1.0, -2.2], color="#2a78d6", lw=1.6, ls=":")              # my approach
    a.text(-17, -1.0, "my approach\n(not copied)", fontsize=7, color="#2a78d6", va="top")
    a.annotate("", xy=(-5.0, -2.0), xytext=(-5.0, 13), arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2))
    a.text(-6.0, 13.5, "planned:\nstraight down", fontsize=7, color="#14845c", va="bottom", ha="right")
    t = np.linspace(0, 1, 40)
    a.plot(-5.0 + t * 28, -2.2 + 12 * np.sin(np.pi * t) ** 0.8 + t * 0.6, color="#6250d6", lw=2.4)          # my carry
    a.text(12, 13, "carry and place\nfrom my demo", fontsize=7, color="#6250d6", ha="left", va="bottom")
    a.set_xlim(-18, 32); a.set_ylim(-9, 21); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color("#d6d5cf")
    a.set_title("contact from me, approach planned", fontsize=8, color=INK2, pad=2)
    a.text(0.98, 0.03, "side view", transform=a.transAxes, fontsize=6.5, color=MUTED, ha="right")


def main(phone, sim, wm, out):
    fig = plt.figure(figsize=(13, 4.9), facecolor=SURF)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    W, H, Y = 0.17, 0.33, 0.07
    xs = [0.012, 0.212, 0.412, 0.612]
    box(ax, xs[0], Y, W, H, "1  Phone recording", "iPhone: RGB, LiDAR, camera pose\n(Stray Scanner). Printed tags on\ntable, bowl and plate.\n13 demos of my hand.", "p2p/human/")
    box(ax, xs[1], Y, W, H, "2  Metric 3D demo", "Tags give a table frame in cm;\nhand tracking gives the path;\ngrasp point = camera ray x rim\nplane; grasp/lift/place/release.", "p2p/human/")
    box(ax, xs[2], Y, W, H, "3  Retarget to the robot", "Anchored on bowl and plate:\nmy grasp point, carry and place\npoint; the approach to the bowl\nis planned (Q1, Q2).", "p2p/retarget/", accent=True)
    box(ax, xs[3], Y, W, H, "4  Generate and verify", "60 new scenes per demo, grasp\nand place perturbed; LIBERO\nchecks success. 10 demos ->\n438 robot demos (+ failures).", "p2p/data/")
    for i in range(3):
        ax.add_patch(FancyArrowPatch((xs[i] + W + 0.004, Y + H / 2), (xs[i + 1] - 0.004, Y + H / 2),
                                     arrowstyle="-|>", mutation_scale=14, color=MUTED, lw=1.2))
    # branches
    bx, bw, bh = 0.812, 0.176, 0.098
    outs = [("ACT policy", "imitation (LeRobot)", "p2p/policy/", 0.302),
            ("Offline RL", "IQL critic weights ACT's loss", "p2p/rl/", 0.187),
            ("World model", "imagines the camera; judges plans", "p2p/wm/", 0.072)]
    for t, b, c, y in outs:
        ax.add_patch(FancyBboxPatch((bx, y), bw, bh, boxstyle="round,pad=0.01,rounding_size=0.015", fc=BOX, ec="#d6d5cf", lw=1))
        ax.text(bx + 0.01, y + bh - 0.022, t, fontsize=10, fontweight="bold", color=INK, va="top")
        ax.text(bx + 0.01, y + bh - 0.06, b, fontsize=8.3, color=INK2, va="top")
        ax.text(bx + bw - 0.008, y + bh - 0.022, c, fontsize=7.5, color=MUTED, va="top", ha="right", family="DejaVu Sans Mono")
        ax.add_patch(FancyArrowPatch((xs[3] + W + 0.004, Y + H / 2), (bx - 0.004, y + bh / 2),
                                     arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1.0, connectionstyle="arc3,rad=0.0"))
    # thumbnails
    th = 0.4
    thumb(fig, [xs[0] + 0.03, 0.47, 0.11, th], img(phone), "phone frame (background blurred)")
    demo_thumb(fig, [xs[1] + 0.005, 0.47, 0.16, th])
    retarget_thumb(fig, [xs[2] - 0.005, 0.47, 0.18, th])
    thumb(fig, [xs[3] + 0.02, 0.47, 0.13, th], img(sim), "LIBERO scene, robot camera")
    thumb(fig, [bx + 0.022, 0.47, 0.125, th], img(wm), "world model: imagined view")
    fig.savefig(out, dpi=160, facecolor=SURF)


if __name__ == "__main__":
    main(*sys.argv[1:4], ROOT / "assets" / "pipeline.png")
