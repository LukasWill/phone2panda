"""Sanity-check Stray Scanner recordings right after you record them (runs on your Mac).

    pip install numpy opencv-python
    python tools/check_recording.py /path/to/recording_folder [more folders ...]
    python tools/check_recording.py /path/to/folder_with_many_recordings

Only what the pipeline actually needs is a pass/fail check (everything else is INFO):
  1. the table, bowl and plate tags are all seen in the first second (object start positions),
  2. the table tag is seen sharply in enough frames to anchor ARKit's camera track,
  3. ARKit's camera motion agrees with the table tag (any amount of phone motion is fine),
  4. the bowl is seen resting on the plate near the end (this gives the placement you demonstrated).
Hand tracking needs MediaPipe and is checked by the full extraction (p2p/human/extract.py).
A contact sheet (8 frames) is written next to the recording: your hand should be in every tile
from just before the grasp until after the release.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2p.human.markers import detect_tags, table_frame, transform  # noqa: E402
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
    print(f"{OK} {rec.n_rgb} frames @ {rec.fps:.0f} fps = {rec.n_rgb / rec.fps:.1f} s")
    good = True
    per_frame = {}
    for i, frame in rec.rgb_frames(step=2):
        tags = detect_tags(frame, rec.K_rgb_at(i))
        per_frame[i] = (tags, table_frame(tags))

    # 1. all three tags together in the first second
    ref = next((i for i, (tags, T) in per_frame.items()
                if i <= rec.fps and T is not None and {"bowl", "plate"} <= {t.role for t in tags.values()}), None)
    if ref is None:
        print(f"{FAIL} table, bowl and plate tags are never all visible in the first second")
        contact_sheet(rec, folder.parent / f"check_{folder.name}.jpg")
        return False
    tags, T_ref = per_frame[ref]
    print(f"{OK} all tags seen at frame {ref}")

    # 2. sharp table-tag sightings (the camera track is ARKit anchored to these)
    sharp = {i: T for i, (tg, T) in per_frame.items() if T is not None and tg[0].reproj_px < 1.5}
    flag = OK if len(sharp) >= 10 else FAIL
    good &= flag != FAIL
    print(f"{flag} table tag seen sharply in {len(sharp)} frames (need >= 10; it may leave the view in between)")

    # 3. ARKit vs table tag: do both agree on how the camera moved?
    odo = rec.odometry()
    if odo is None:
        print(f"{FAIL} no odometry.csv"); good = False
    else:
        pos = {int(f): p for f, p in zip(odo["frame"], odo["position"])}
        c = {i: T[:3, 3] for i, T in sharp.items() if i in pos}
        errs = [abs(np.linalg.norm(pos[i] - pos[ref]) - np.linalg.norm(ci - c[ref])) for i, ci in c.items()] if ref in c else []
        e = float(np.median(errs)) if errs else np.nan
        flag = OK if e < 0.015 else FAIL
        good &= flag != FAIL
        print(f"{flag} ARKit and table tag agree on the camera motion to {100 * e:.1f} cm (need < 1.5 cm)")

    # 4. bowl resting on the plate near the end
    plate0 = transform(T_ref, next(t for t in tags.values() if t.role == "plate").center_cam)[0]
    bowl0 = transform(T_ref, next(t for t in tags.values() if t.role == "bowl").center_cam)[0]
    last = sorted(per_frame)[int(0.6 * len(per_frame)):]
    ends = []
    for i in last:
        tg, T = per_frame[i]
        b = next((t for t in tg.values() if t.role == "bowl"), None)
        if b is not None:
            p = transform(T if T is not None else T_ref, b.center_cam)[0]
            if np.linalg.norm(p[:2] - bowl0[:2]) > 0.05:      # it has moved away from its start
                ends.append(p)
    if len(ends) < 3:
        print(f"{FAIL} the bowl is not seen on the plate in the last 40% of the clip: keep it in view ~1 s after releasing")
        good = False
    else:
        off = np.linalg.norm(np.median(ends, 0)[:2] - plate0[:2])
        flag = OK if off < 0.05 else WARN
        print(f"{flag} bowl ends {100 * off:.1f} cm from the plate centre")

    # INFO only
    for t in tags.values():
        p = t.center_cam
        z = rec.depth_at_rgb_pixel(ref, ((rec.K_rgb_at(ref) @ p)[:2] / p[2])[None])[0]
        if np.isfinite(z):
            print(f"INFO {t.role:5s} tag: PnP {p[2]:.3f} m vs LiDAR {z:.3f} m ({100 * (z / p[2] - 1):+.1f}%)")
    sheet = folder.parent / f"check_{folder.name}.jpg"
    contact_sheet(rec, sheet)
    print(f"     contact sheet -> {sheet}")
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
