"""A hand-made human demo in the same format as p2p.human.extract produces (table frame).

Used to test retargeting and replay before (and independently of) real recordings.
"""
import json

import numpy as np

from p2p.config import HUMAN_BOWL_HEIGHT


def smoothstep(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def make_demo(name="synthetic", bowl=(0.01, -0.38), plate=(0.0, -0.21), rim_angle_deg=180.0,
              place_offset=(0.01, 0.005), fps=30.0, lift=0.10):
    bowl_start = np.array([*bowl, 0.0])
    plate_c = np.array([*plate, 0.012])
    a = np.radians(rim_angle_deg)
    radial = np.array([np.cos(a), np.sin(a), 0.0])
    rim_top = HUMAN_BOWL_HEIGHT
    grasp = bowl_start + 0.05 * radial + np.array([0, 0, rim_top - 0.01])
    bowl_end = np.array([plate_c[0] + place_offset[0], plate_c[1] + place_offset[1], plate_c[2]])
    place = bowl_end + (grasp - bowl_start)
    home = np.array([0.22, -0.48, 0.02])
    # (time, position, aperture) key poses; linear-in-smoothstep interpolation between them
    keys = [(0.0, home, 0.06), (0.5, home, 0.06), (1.5, grasp + [0, 0, 0.07], 0.07), (1.9, grasp, 0.07),
            (2.3, grasp, 0.03), (2.5, grasp, 0.03), (3.2, (grasp + place) / 2 + [0, 0, lift], 0.03),
            (3.8, place + [0, 0, 0.01], 0.03), (4.0, place, 0.03), (4.3, place, 0.065),
            (4.8, place + [0, 0, 0.08], 0.065), (5.6, home, 0.06)]
    t = np.arange(0, keys[-1][0], 1 / fps)
    P, A = np.zeros((len(t), 3)), np.zeros(len(t))
    for j in range(len(keys) - 1):
        (t0, p0, a0), (t1, p1, a1) = keys[j], keys[j + 1]
        m = (t >= t0) & (t < t1)
        u = smoothstep((t[m] - t0) / (t1 - t0))[:, None]
        P[m] = p0 + u * (np.asarray(p1) - p0)
        A[m] = a0 + u[:, 0] * (a1 - a0)
    B = np.tile(bowl_start, (len(t), 1))
    carrying = (t >= 2.5) & (t < 4.0)
    B[carrying] = P[carrying] - (grasp - bowl_start)
    B[t >= 4.0] = bowl_end
    k = lambda s: int(np.searchsorted(t, s))
    events = np.array([0, k(2.3), k(2.55), k(4.0), k(4.15), len(t) - 1])
    closing = np.tile(radial, (len(t), 1))
    quality = {"camera": {"source": "synthetic"}, "rim_angle_deg": rim_angle_deg}
    return dict(name=name, t=t, pinch=P, aperture=A, hand_ok=np.ones(len(t), bool), bowl=B,
                closing_dir=closing, bowl_start=bowl_start, bowl_end=bowl_end, plate_center=plate_c,
                grasp_point=grasp, grasp_point_lifted=grasp, events=events, quality=json.dumps(quality))
