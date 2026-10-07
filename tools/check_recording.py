"""Sanity-check Stray Scanner recordings right after you record them (runs on your Mac).

    pip install numpy opencv-python
    python tools/check_recording.py /path/to/recording_folder [more folders ...]
    python tools/check_recording.py /path/to/folder_with_many_recordings

Works for a handheld (moving) phone. For each recording it checks:
  1. all tags (table, bowl, plate) are seen together within the first second,
  2. metric consistency: distance to each tag from PnP (uses the printed size) agrees with the
     LiDAR depth there -> catches a wrong print scale or wrong intrinsics,
  3. camera placement and how much the phone moved / rotated during the demo,
  4. ARKit odometry agrees with the table tag about how far the camera moved
     (this is what later lets us track the hand through camera motion),
  5. the bowl tag is tracked while carried and the bowl is actually lifted, and it ends on the plate.
A contact sheet (8 frames with tag overlays) is written next to the recording, so you can check
by eye that your hand is visible in every tile.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2p.human.markers import TAG_SPECS, camera_geometry, detect_tags, table_frame, transform  # noqa: E402
from p2p.human.stray import StrayRecording  # noqa: E402

OK, WARN, FAIL = "OK  ", "WARN", "FAIL"


def contact_sheet(rec: StrayRecording, out_path: Path, n: int = 8):
    idx = set(np.linspace(0, rec.n_rgb - 1, n).astype(int).tolist())
    tiles = []
    for i, frame in rec.rgb_frames():
        if i not in idx:
            continue
        for t in detect_tags(frame, rec.K_rgb_at(i)).values():
            cv2.polylines(frame, [t.corners.astype(np.int32)], True, (0, 255, 0), 4)
            cv2.putText(frame, t.role, tuple(t.corners[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        tile = cv2.resize(frame, (480, int(480 * frame.shape[0] / frame.shape[1])))
        cv2.putText(tile, f"frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        tiles.append(tile)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out_path), np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]))


def check(folder: Path) -> bool:
    print(f"\n=== {folder.name}")
    rec = StrayRecording(folder)
    dur = rec.n_rgb / rec.fps
    print(f"{OK} {rec.n_rgb} frames @ {rec.fps:.0f} fps = {dur:.1f} s; rgb {rec.rgb_size}, depth {rec.depth_size}")
    good = True
    if not 4.0 <= dur <= 15.0:
        print(f"{WARN} duration {dur:.1f} s is outside the planned 5-12 s")

    # 0. things that need no tags: phone motion (ARKit) and how close anything got to the phone ---
    odo = rec.odometry()
    if odo is not None:
        P, Q = odo["position"], odo["quat_xyzw"]
        rot = np.degrees(2 * np.arccos(np.clip(np.abs(Q @ Q[0]), 0, 1))).max()
        moved_arkit = np.linalg.norm(P - P[0], axis=1).max()
        flag = OK if moved_arkit < 0.10 and rot < 20 else WARN
        print(f"{flag} ARKit: phone moved up to {100 * moved_arkit:.1f} cm and rotated up to {rot:.0f} deg "
              f"(aim < 10 cm, < 20 deg)")
    closest, at = np.inf, -1
    for i in range(0, len(rec.depth_paths), 4):
        d, c = rec.depth(i), rec.confidence(i)
        valid = (d > 0) & ((c >= 1) if c is not None else True)
        if valid.sum() > 100:
            q = float(np.percentile(d[valid], 0.5))
            if q < closest:
                closest, at = q, i
    flag = OK if closest >= 0.18 else WARN
    print(f"{flag} closest thing to the phone: {100 * closest:.0f} cm at frame {at} "
          f"(your forearm passing at ~20 cm is normal; under 18 cm usually means the hand or bowl came too close)")

    # One pass over the video (every 2nd frame): tags + per-frame table frame -----------------
    per_frame = {}
    for i, frame in rec.rgb_frames(step=2):
        tags = detect_tags(frame, rec.K_rgb_at(i))
        per_frame[i] = (tags, table_frame(tags))

    # 1. a reference frame in the first second where all three tags are visible ---------------
    ref = next((i for i, (tags, T) in per_frame.items()
                if i <= rec.fps and T is not None and {"bowl", "plate"} <= {t.role for t in tags.values()}), None)
    if ref is None:
        i0_roles = {t.role for t in per_frame[0][0].values()}
        print(f"{FAIL} table, bowl and plate tags are never all visible in the first second "
              f"(frame 0 sees: {sorted(i0_roles) or 'none'}). Nothing should cover them at the start.")
        contact_sheet(rec, folder.parent / f"check_{folder.name}.jpg")
        return False
    tags, T_ref = per_frame[ref]
    for role in ("table", "bowl", "plate"):
        t = next(t for t in tags.values() if t.role == role)
        flag = OK if t.side_px >= 40 else WARN
        print(f"{flag} {role:5s} tag id {t.tag_id} (frame {ref}): {t.side_px:.0f} px wide, reprojection {t.reproj_px:.2f} px")

    # 2. PnP distance vs LiDAR depth --------------------------------------------------------------
    for t in tags.values():
        p = t.center_cam
        uv = (rec.K_rgb_at(ref) @ p)[:2] / p[2]
        z = rec.depth_at_rgb_pixel(ref, uv[None])[0]
        if np.isnan(z):
            print(f"{WARN} no valid LiDAR depth at the {t.role} tag")
            continue
        diff = z - p[2]
        flag = OK if abs(diff) < 0.02 else (WARN if abs(diff) < 0.04 else FAIL)
        good &= flag != FAIL
        side = TAG_SPECS[t.tag_id][1]
        print(f"{flag} {t.role:5s} tag: z from PnP {p[2]:.3f} m vs LiDAR {z:.3f} m (diff {100 * diff:+.1f} cm; "
              f"if LiDAR is right, this tag is {1000 * side * z / p[2]:.1f} mm, config says {1000 * side:.1f} mm)")

    # 3. camera placement and motion -------------------------------------------------------------
    g = camera_geometry(T_ref)
    h, tilt = g["height_above_table_m"], g["tilt_from_vertical_deg"]
    flag = OK if (0.3 <= h <= 0.8 and 15 <= tilt <= 60) else WARN
    print(f"{flag} camera {h:.2f} m above table, tilted {tilt:.0f} deg from straight-down (aim: 0.35-0.7 m, 25-55 deg)")
    # which way is "away from you" in the image? (table +y, set by how the sheet is taped)
    K0 = rec.K_rgb_at(ref)
    T_cam_table = np.linalg.inv(T_ref)
    o, fwd = transform(T_cam_table, np.array([[0, 0, 0], [0, 0.1, 0]]))
    du, dv = (K0 @ fwd)[:2] / fwd[2] - (K0 @ o)[:2] / o[2]
    ang = float(np.degrees(np.arctan2(du, -dv)))
    flag = OK if abs(ang) < 45 else WARN
    print(f"{flag} 'away from you' points {ang:+.0f} deg from image-up (want within 45: hold the phone in "
          f"landscape so the video looks like what you see)")
    seen = {i: T for i, (_, T) in per_frame.items() if T is not None}
    frac_table = len(seen) / len(per_frame)
    flag = OK if frac_table >= 0.8 else WARN
    print(f"{flag} table tag visible in {100 * frac_table:.0f}% of frames (want >= 80%)")
    centres = {i: T[:3, 3] for i, T in seen.items()}            # camera centre in the table frame
    moved = max(np.linalg.norm(c - centres[ref]) for c in centres.values())
    tilts = [camera_geometry(T)["tilt_from_vertical_deg"] for T in seen.values()]
    flag = OK if moved < 0.10 and np.ptp(tilts) < 20 else WARN
    print(f"{flag} phone moved up to {100 * moved:.1f} cm and tilted over a {np.ptp(tilts):.0f} deg range "
          f"(handheld is fine; aim < 10 cm and < 20 deg to avoid blur)")

    # 4. ARKit odometry vs table tag: camera displacement from the reference frame -----------------
    if odo is None or len(odo["position"]) < rec.n_rgb * 0.9:
        print(f"{WARN} odometry.csv missing or short - needed for a moving camera")
    elif moved < 0.02:
        print(f"{OK} camera nearly static, odometry check not needed")
    else:
        pos = {int(f): p for f, p in zip(odo["frame"], odo["position"])}
        errs = [abs(np.linalg.norm(pos[i] - pos[ref]) - np.linalg.norm(c - centres[ref]))
                for i, c in centres.items() if i in pos and ref in pos]
        e = float(np.median(errs)) if errs else np.nan
        flag = OK if e < 0.015 else WARN
        print(f"{flag} ARKit vs table tag camera displacement: median disagreement {100 * e:.1f} cm (want < 1.5 cm)")

    # 5. bowl: tracked, lifted, ends on the plate --------------------------------------------------
    bowl_z, bowl_seen, total = [], 0, 0
    for i, (tags_i, T_i) in per_frame.items():
        total += 1
        b = next((t for t in tags_i.values() if t.role == "bowl"), None)
        T_use = T_i if T_i is not None else T_ref
        if b is not None:
            bowl_seen += 1
            bowl_z.append(transform(T_use, b.center_cam)[0, 2])
    if bowl_z:
        lift = float(np.max(bowl_z) - np.median(bowl_z[: max(3, len(bowl_z) // 10)]))
        frac = bowl_seen / total
        flag = OK if frac > 0.6 and lift > 0.04 else WARN
        print(f"{flag} bowl tag seen in {100 * frac:.0f}% of frames; max lift seen {100 * lift:.1f} cm (want > 60% and > 4 cm; "
              f"a low value usually means the tag was hidden while carrying)")
    plate0 = transform(T_ref, next(t for t in tags.values() if t.role == "plate").center_cam)[0]
    last = [i for i, (tg, T) in per_frame.items() if any(t.role == "bowl" for t in tg.values())]
    if last:
        tg, T = per_frame[last[-1]]
        b_end = transform(T if T is not None else T_ref, next(t for t in tg.values() if t.role == "bowl").center_cam)[0]
        off = np.linalg.norm(b_end[:2] - plate0[:2])
        flag = OK if off < 0.03 else WARN
        print(f"{flag} bowl ends {100 * off:.1f} cm from the plate centre (want < 3 cm)")
    sheet = folder.parent / f"check_{folder.name}.jpg"
    contact_sheet(rec, sheet)
    print(f"     contact sheet -> {sheet}  (your right hand must be visible in every tile)")
    return good


def main(paths: list[str]):
    folders = []
    for p in map(Path, paths):
        if (p / "rgb.mp4").exists():
            folders.append(p)
        else:
            folders += sorted(q for q in p.iterdir() if (q / "rgb.mp4").exists())
    if not folders:
        sys.exit("No Stray Scanner recordings found (folders containing rgb.mp4).")
    results = {f.name: check(f) for f in folders}
    print("\nSummary:", ", ".join(f"{k}: {'pass' if v else 'FAIL'}" for k, v in results.items()))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
