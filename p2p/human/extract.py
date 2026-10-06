"""One phone recording -> one metric demonstration in the TABLE frame.

    python -m p2p.human.extract RAW_DIR OUT_DIR        (RAW_DIR holds one folder per recording)

Pipeline per recording (every 2nd frame, i.e. ~30 Hz):
  1. tags      : table / bowl / plate tags -> their poses in the camera frame (PnP)
  2. camera    : one camera pose per frame in the table frame, by fusing
                   - ARKit odometry (smooth, every frame, but in ARKit's own world frame) with
                   - the table tag (exact table frame, but only when visible and sharp).
                 We solve for the single fixed transform table<-ARKit-world that best agrees with all
                 tag sightings, and also for ARKit's camera-axis convention (see fuse_camera_poses).
  3. hand      : MediaPipe landmarks lifted with LiDAR (hands.py) -> pinch point, aperture
  4. objects   : bowl path from its tag; plate centre; both in the table frame
  5. events    : lift-off and set-down come from the BOWL (its tag), not the hand: a bowl that
                 moves is unambiguous, a closing hand is not. Grasp and release are found around them.
  6. grasp     : where on the rim the fingers were. Fingertip depth is unreliable from above, so we
                 intersect the camera ray through the pinch PIXEL with the plane at rim height.
Output: <name>.npz with all per-frame arrays + events, and <name>_qa.png for a visual check.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from p2p.config import HUMAN_BOWL_HEIGHT, HUMAN_BOWL_RIM_DIAMETER, HUMAN_PLATE_DIAMETER
from p2p.human.hands import INDEX_TIP, THUMB_TIP, WRIST, HandDetector, lift_hand, pick_hand
from p2p.human.markers import detect_tags, inv_T, table_frame, to_T, transform
from p2p.human.stray import StrayRecording

FLIP_YZ = np.diag([1.0, -1.0, -1.0, 1.0])   # OpenCV camera axes <-> OpenGL/ARKit camera axes


def quat2mat(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotation_mean(Rs: np.ndarray) -> np.ndarray:
    """Chordal L2 mean of rotations: average the matrices, then project back onto SO(3) with an SVD."""
    U, _, Vt = np.linalg.svd(Rs.mean(0))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def fuse_camera_poses(frames: list[int], tag_T: dict[int, np.ndarray], odo: dict | None):
    """Return ({frame: T_table_cam}, info).

    T_table_cam(i) = T_table_world @ T_world_cam(i) @ C
    T_world_cam(i) comes from ARKit; C is identity if the app stored OpenCV-style camera axes and
    FLIP_YZ if it stored ARKit's native ones (camera looking along -z). We do not trust either
    assumption: both are tried, and the one under which all tag sightings agree on a single
    T_table_world wins. The remaining disagreement is reported as a quality number.
    """
    if odo is None or len(tag_T) < 3:
        # fallback: tag only, hold the last seen pose in between
        out, last = {}, None
        for i in frames:
            last = tag_T.get(i, last)
            if last is not None:
                out[i] = last
        return out, {"source": "tag_only", "n_tag_frames": len(tag_T)}
    idx = {int(f): k for k, f in enumerate(odo["frame"])}
    T_wc = {i: to_T(quat2mat(odo["quat_xyzw"][idx[i]]), odo["position"][idx[i]]) for i in frames if i in idx}
    best = None
    for name, C in (("opencv", np.eye(4)), ("arkit", FLIP_YZ)):
        A = [tag_T[i] @ inv_T(T_wc[i] @ C) for i in tag_T if i in T_wc]
        t = np.array([a[:3, 3] for a in A])
        R = rotation_mean(np.array([a[:3, :3] for a in A]))
        t0 = np.median(t, 0)
        spread_mm = 1000 * float(np.median(np.linalg.norm(t - t0, axis=1)))
        spread_deg = float(np.median([rot_angle_deg(a[:3, :3] @ R.T) for a in A]))
        if best is None or spread_mm < best[1]:
            best = (name, spread_mm, spread_deg, to_T(R, t0), C)
    name, spread_mm, spread_deg, T_tw, C = best
    out = {i: T_tw @ T_wc[i] @ C for i in frames if i in T_wc}
    return out, {"source": "arkit+tag", "convention": name, "spread_mm": round(spread_mm, 1),
                 "spread_deg": round(spread_deg, 2), "n_tag_frames": len(tag_T)}


def first_run(mask: np.ndarray, n: int = 3) -> int | None:
    """Index of the first run of n consecutive True values."""
    run = 0
    for k, m in enumerate(mask):
        run = run + 1 if m else 0
        if run >= n:
            return k - n + 1
    return None


def extract(folder: Path, detector: HandDetector | None = None, step: int = 2) -> dict:
    rec = StrayRecording(folder)
    odo = rec.odometry()
    own_detector = detector is None
    detector = detector or HandDetector()
    ts = odo["timestamp"] - odo["timestamp"][0] if odo is not None else np.arange(rec.n_rgb) / rec.fps

    frames, tag_T, bowl_cam, plate_cam, hands, prev = [], {}, {}, {}, {}, None
    for i, bgr in rec.rgb_frames(step=step):
        frames.append(i)
        K = rec.K_rgb_at(i)
        tags = detect_tags(bgr, K)
        T = table_frame(tags)
        # only trust sharp, well-fitting table-tag sightings for the camera-pose fit
        if T is not None and tags[0].reproj_px < 1.5:
            tag_T[i] = T
        for t in tags.values():
            if t.role == "bowl":
                bowl_cam[i] = t.center_cam
            elif t.role == "plate":
                plate_cam[i] = t.center_cam
        h = pick_hand(detector(bgr, int(1000 * ts[min(i, len(ts) - 1)])), prev)
        prev = None if h is None else h.uv[WRIST]
        if h is not None:
            h3 = lift_hand(h, lambda uv, i=i: rec.depth_at_rgb_pixel(i, uv), K)
            if h3 is not None:
                hands[i] = (h, h3, K)
    if own_detector:
        detector.close()

    T_tc, cam_info = fuse_camera_poses(frames, tag_T, odo)
    frames = [i for i in frames if i in T_tc]
    N = len(frames)
    nan3 = np.full((N, 3), np.nan)
    pinch, palm, closing_dir, bowl_tag, plate_tag = nan3.copy(), nan3.copy(), nan3.copy(), nan3.copy(), nan3.copy()
    aperture, pinch_uv = np.full(N, np.nan), np.full((N, 2), np.nan)
    for k, i in enumerate(frames):
        T = T_tc[i]
        if i in hands:
            h, h3, _ = hands[i]
            pinch[k] = transform(T, h3.pinch)[0]
            palm[k] = transform(T, h3.palm)[0]
            d = T[:3, :3] @ (h3.points[INDEX_TIP] - h3.points[THUMB_TIP])
            closing_dir[k] = d / (np.linalg.norm(d) + 1e-9)
            aperture[k] = h3.aperture
            pinch_uv[k] = (h.uv[THUMB_TIP] + h.uv[INDEX_TIP]) / 2
        if i in bowl_cam:
            bowl_tag[k] = transform(T, bowl_cam[i])[0]
        if i in plate_cam:
            plate_tag[k] = transform(T, plate_cam[i])[0]
    t = ts[np.array(frames)]

    # --- objects ---------------------------------------------------------------------------------
    early = t < t[0] + 1.0
    z_rest = np.nanmedian(bowl_tag[early, 2]) if np.isfinite(bowl_tag[early, 2]).any() else np.nanmedian(bowl_tag[:15, 2])
    bowl = bowl_tag - np.array([0, 0, z_rest])        # bowl bottom centre (it rests on the table at the start)
    bowl_start = np.nanmedian(bowl[early], 0) if np.isfinite(bowl[early]).any() else np.nanmedian(bowl[:15], 0)
    plate_center = np.nanmedian(plate_tag[early], 0) if np.isfinite(plate_tag[early]).any() else np.nanmedian(plate_tag, 0)
    late = t > t[-1] - 0.7
    bowl_end = np.nanmedian(bowl[late], 0) if np.isfinite(bowl[late]).any() else bowl[np.where(np.isfinite(bowl[:, 0]))[0][-1]]

    # --- events (from the bowl) --------------------------------------------------------------------
    lifted = (bowl[:, 2] > 0.015) & np.isfinite(bowl[:, 2])
    k_lift = first_run(lifted)
    if k_lift is None:
        raise RuntimeError(f"{folder.name}: the bowl tag never rises 1.5 cm - was the bowl lifted, and is its tag visible?")
    above_end = (bowl[:, 2] > bowl_end[2] + 0.012) & np.isfinite(bowl[:, 2])
    k_down = int(np.where(above_end)[0][-1]) + 1 if above_end.any() else k_lift + 1
    fps = 1.0 / np.median(np.diff(t))
    # grasp: the hand is (nearly) still on the rim just before lift-off -> median pinch over that window
    win = (t >= t[k_lift] - 0.35) & (t <= t[k_lift] - 0.05) & np.isfinite(pinch[:, 0])
    if not win.any():
        win = (t >= t[k_lift] - 0.8) & (t <= t[k_lift]) & np.isfinite(pinch[:, 0])
    k_grasp = int(np.where(win)[0][len(np.where(win)[0]) // 2]) if win.any() else max(k_lift - int(0.2 * fps), 0)
    # release: after set-down, the fingers open or the hand moves away from where it set the bowl down
    k_release = min(k_down + int(0.3 * fps), N - 1)
    if np.isfinite(pinch[k_down, 0]):
        for k in range(k_down, N):
            if np.isfinite(pinch[k, 0]) and (np.linalg.norm(pinch[k] - pinch[k_down]) > 0.02
                                             or aperture[k] - aperture[k_down] > 0.015):
                k_release = k
                break
    hand_ok = np.isfinite(pinch[:, 0])

    # --- grasp point on the rim: ray through the pinch pixel ∩ plane at rim height -------------
    rim_top = bowl_start[2] + HUMAN_BOWL_HEIGHT
    grasp_lifted = np.nanmedian(pinch[win], 0) if win.any() else pinch[k_grasp]
    grasp_ray = grasp_lifted.copy()
    i_g = frames[k_grasp]
    if np.isfinite(pinch_uv[k_grasp, 0]):
        K = rec.K_rgb_at(i_g)
        d_cam = np.linalg.solve(K, np.array([*pinch_uv[k_grasp], 1.0]))
        T = T_tc[i_g]
        o, d = T[:3, 3], T[:3, :3] @ d_cam
        z_plane = rim_top - 0.01          # finger pads straddle the wall ~1 cm below its top
        grasp_ray = o + d * (z_plane - o[2]) / d[2]
    radial = grasp_ray[:2] - bowl_start[:2]
    place_offset = bowl_end[:2] - plate_center[:2]
    human_ok = bool(np.linalg.norm(place_offset) < (HUMAN_PLATE_DIAMETER - HUMAN_BOWL_RIM_DIAMETER) / 2)

    quality = {
        "camera": cam_info, "fps_processed": round(float(fps), 1), "duration_s": round(float(t[-1] - t[0]), 2),
        "hand_detected_frac": round(float(hand_ok.mean()), 3),
        "bowl_tag_frac": round(float(np.isfinite(bowl_tag[:, 0]).mean()), 3),
        "grasp_radius_cm": round(100 * float(np.linalg.norm(radial)), 1),   # expect ~ bowl radius (5 cm)
        "grasp_ray_vs_lifted_cm": round(100 * float(np.linalg.norm(grasp_ray[:2] - grasp_lifted[:2])), 1),
        "rim_angle_deg": round(float(np.degrees(np.arctan2(radial[1], radial[0]))), 1),
        "place_offset_cm": round(100 * float(np.linalg.norm(place_offset)), 1),
        "lift_height_cm": round(100 * float(np.nanmax(bowl[:, 2])), 1),
        "human_success": human_ok,
    }
    return dict(name=folder.name, t=t, frames=np.array(frames), T_table_cam=np.array([T_tc[i] for i in frames]),
                pinch=pinch, pinch_uv=pinch_uv, palm=palm, closing_dir=closing_dir, aperture=aperture,
                hand_ok=hand_ok, bowl=bowl, bowl_tag=bowl_tag, plate_tag=plate_tag,
                bowl_start=bowl_start, bowl_end=bowl_end, plate_center=plate_center,
                grasp_point=grasp_ray, grasp_point_lifted=grasp_lifted,
                events=np.array([0, k_grasp, k_lift, k_down, k_release, N - 1]),
                quality=json.dumps(quality))


EVENT_NAMES = ["start", "grasp", "lift", "down", "release", "end"]


def save(demo: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{demo['name']}.npz", **demo)
    qa_plot(demo, out_dir / f"{demo['name']}_qa.png")


def load(path: Path) -> dict:
    d = dict(np.load(path, allow_pickle=False))
    d["name"] = str(d["name"])
    d["quality"] = json.loads(str(d["quality"]))
    return d


def qa_plot(d: dict, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = json.loads(d["quality"]) if isinstance(d["quality"], str) else d["quality"]
    t, P, B, ev = d["t"] - d["t"][0], d["pinch"], d["bowl"], d["events"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    a = ax[0]
    a.set_title("top view (table frame, cm)")
    a.add_patch(plt.Circle(100 * d["plate_center"][:2], 100 * HUMAN_PLATE_DIAMETER / 2, fill=False, color="0.5"))
    for c, lab in ((d["bowl_start"], "bowl start"), (d["bowl_end"], "bowl end")):
        a.add_patch(plt.Circle(100 * c[:2], 100 * HUMAN_BOWL_RIM_DIAMETER / 2, fill=False, ls="--", color="C1"))
    a.plot(*(100 * B[:, :2].T), ".", color="C1", ms=3, label="bowl (tag)")
    sc = a.scatter(*(100 * P[:, :2].T), c=t, s=6, cmap="viridis", label="pinch")
    a.plot(*(100 * d["grasp_point"][:2]), "r*", ms=12, label="grasp (ray)")
    a.plot(*(100 * d["grasp_point_lifted"][:2]), "rx", ms=8, label="grasp (lifted)")
    a.add_patch(plt.Rectangle((-7.2, -7.2), 14.4, 14.4, fill=False, color="k"))
    a.set_aspect("equal"); a.set_xlabel("x (right)"); a.set_ylabel("y (away from you)")
    a.legend(fontsize=7, loc="best"); fig.colorbar(sc, ax=a, label="t (s)")
    a = ax[1]
    a.set_title("height above table (cm)")
    a.plot(t, 100 * P[:, 2], label="pinch"); a.plot(t, 100 * B[:, 2], label="bowl bottom")
    for k, n in zip(ev, EVENT_NAMES):
        a.axvline(t[k], color="0.7", lw=0.8); a.text(t[k], a.get_ylim()[1] * 0.95, n, fontsize=7, rotation=90)
    a.set_xlabel("t (s)"); a.legend(fontsize=7)
    a = ax[2]
    a.set_title("aperture (thumb-index, cm)")
    a.plot(t, 100 * d["aperture"])
    for k in ev:
        a.axvline(t[k], color="0.7", lw=0.8)
    a.set_xlabel("t (s)")
    fig.suptitle(f"{d['name']}  |  " + ", ".join(f"{k}={v}" for k, v in q.items() if k != "camera")
                 + f"\ncamera: {q['camera']}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(raw_dir: str, out_dir: str):
    raw, out = Path(raw_dir), Path(out_dir)
    folders = [raw] if (raw / "rgb.mp4").exists() else sorted(p for p in raw.iterdir() if (p / "rgb.mp4").exists())
    det = HandDetector()
    summary = {}
    for f in folders:
        try:
            d = extract(f, det)
            save(d, out)
            summary[f.name] = json.loads(d["quality"])
            print(f.name, summary[f.name])
        except Exception as e:  # keep going: one bad recording should not stop the batch
            summary[f.name] = {"error": f"{type(e).__name__}: {e}"}
            print(f.name, "FAILED:", e)
    det.close()
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
