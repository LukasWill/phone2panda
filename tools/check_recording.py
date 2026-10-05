"""Sanity-check Stray Scanner recordings right after you record them (runs on your Mac).

    pip install numpy opencv-python
    python tools/check_recording.py /path/to/recording_folder [more folders ...]
    python tools/check_recording.py /path/to/folder_with_many_recordings

For each recording it checks the four things that silently ruin the data later:
  1. all tags are detected in the first frame (table, bowl, plate),
  2. metric consistency: the distance to each tag from PnP (uses the printed size) agrees
     with the LiDAR depth at that spot -> catches wrong print scale or wrong intrinsics,
  3. camera placement: height above the table and tilt (too flat = bad hand depth),
  4. the bowl tag is tracked while the bowl is carried, and the bowl is actually lifted.
It also writes a contact sheet (8 frames with tag overlays) so you can eyeball that your
hand stays in view the whole time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2p.human.markers import camera_geometry, detect_tags, table_frame, transform  # noqa: E402
from p2p.human.stray import StrayRecording  # noqa: E402

OK, WARN, FAIL = "OK  ", "WARN", "FAIL"


def contact_sheet(rec: StrayRecording, out_path: Path, n: int = 8):
    idx = set(np.linspace(0, rec.n_rgb - 1, n).astype(int).tolist())
    tiles = []
    for i, frame in rec.rgb_frames():
        if i not in idx:
            continue
        tags = detect_tags(frame, rec.K_rgb)
        for t in tags.values():
            cv2.polylines(frame, [t.corners.astype(np.int32)], True, (0, 255, 0), 4)
            cv2.putText(frame, f"{t.role}", tuple(t.corners[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        tile = cv2.resize(frame, (480, int(480 * frame.shape[0] / frame.shape[1])))
        cv2.putText(tile, f"frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        tiles.append(tile)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]
    cv2.imwrite(str(out_path), np.vstack(rows))


def check(folder: Path) -> bool:
    print(f"\n=== {folder.name}")
    rec = StrayRecording(folder)
    dur = rec.n_rgb / rec.fps
    print(f"{OK} {rec.n_rgb} frames @ {rec.fps:.0f} fps = {dur:.1f} s; rgb {rec.rgb_size}, depth {rec.depth_size}, "
          f"{len(rec.depth_paths)} depth maps")
    good = True
    if not 3.0 <= dur <= 15.0:
        print(f"{WARN} duration {dur:.1f} s is outside the 3-15 s we planned for")

    # 1. tags in first frame ------------------------------------------------------------------
    frame0 = rec.rgb(0)
    tags = detect_tags(frame0, rec.K_rgb)
    roles = {t.role for t in tags.values()}
    for role in ("table", "bowl", "plate"):
        if role in roles:
            t = next(t for t in tags.values() if t.role == role)
            flag = OK if t.side_px >= 40 else WARN
            print(f"{flag} {role:5s} tag id {t.tag_id}: {t.side_px:.0f} px wide, reprojection {t.reproj_px:.2f} px")
        else:
            print(f"{FAIL} {role} tag not found in the first frame")
            good = False
    T_table_cam = table_frame(tags)
    if T_table_cam is None:
        print(f"{FAIL} no table frame -> cannot check geometry. Is the big tag fully in view and flat?")
        contact_sheet(rec, folder.parent / f"check_{folder.name}.jpg")
        return False

    # 2. PnP distance vs LiDAR depth ------------------------------------------------------------
    for t in tags.values():
        p = t.center_cam
        uv = (rec.K_rgb @ p)[:2] / p[2]
        z_lidar = rec.depth_at_rgb_pixel(0, uv[None])[0]
        if np.isnan(z_lidar):
            print(f"{WARN} no valid LiDAR depth at the {t.role} tag")
            continue
        diff = z_lidar - p[2]
        flag = OK if abs(diff) < 0.02 else (WARN if abs(diff) < 0.04 else FAIL)
        good &= flag != FAIL
        print(f"{flag} {t.role:5s} tag: z from PnP {p[2]:.3f} m vs LiDAR {z_lidar:.3f} m (diff {100 * diff:+.1f} cm)")

    # 3. camera placement ---------------------------------------------------------------------
    g = camera_geometry(T_table_cam)
    h, tilt = g["height_above_table_m"], g["tilt_from_vertical_deg"]
    flag = OK if (0.45 <= h <= 1.0 and 25 <= tilt <= 60) else WARN
    print(f"{flag} camera {h:.2f} m above table, tilted {tilt:.0f} deg from straight-down "
          f"(aim: 0.5-0.9 m, 30-55 deg)")

    # object positions in the table frame
    for role in ("bowl", "plate"):
        t = next((t for t in tags.values() if t.role == role), None)
        if t is not None:
            x, y, z = transform(T_table_cam, t.center_cam)[0]
            print(f"     {role:5s} tag in table frame: x {100 * x:+.1f} cm, y {100 * y:+.1f} cm, z {100 * z:+.1f} cm")

    # 4. bowl tracking through the video ---------------------------------------------------------
    zs, seen, total = [], 0, 0
    for i, frame in rec.rgb_frames(step=3):
        total += 1
        b = next((t for t in detect_tags(frame, rec.K_rgb).values() if t.role == "bowl"), None)
        if b is not None:
            seen += 1
            zs.append(transform(T_table_cam, b.center_cam)[0, 2])
    if zs:
        lift = float(np.max(zs) - np.median(zs[: max(3, len(zs) // 10)]))
        frac = seen / total
        flag = OK if frac > 0.6 and lift > 0.04 else WARN
        print(f"{flag} bowl tag seen in {100 * frac:.0f}% of frames; max lift {100 * lift:.1f} cm "
              f"(want >60% and >4 cm)")
    sheet = folder.parent / f"check_{folder.name}.jpg"
    contact_sheet(rec, sheet)
    print(f"     contact sheet -> {sheet}  (check that your hand is visible in every tile)")
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
