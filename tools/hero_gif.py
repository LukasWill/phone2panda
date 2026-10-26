"""README animations from the notebook 16 renders (no simulator needed).

    python tools/hero_gif.py HERO_DIR        # HERO_DIR = unzipped results_zips/hero_videos.zip (notebook 16, Hub)

Which episodes succeeded: results/hero/summary.json (all 8 teacher and all 8 policy episodes).

Writes
  assets/hero.gif              a generated robot demo next to the learned policy, same evaluation scene
  assets/policy_8_scenes.gif   the learned policy in the first 8 evaluation scenes, none left out

Videos are 20 Hz; the GIFs keep every second frame and play at 10 frames/s, i.e. in real time.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SURF, INK, INK2, OK, FAIL = (252, 252, 251), (11, 11, 11), (82, 81, 78), (27, 175, 122), (214, 69, 65)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
STRIDE, FPS, HOLD_S = 2, 10, 1.5


def load(path: Path, size: int) -> list[np.ndarray]:
    return [np.asarray(Image.fromarray(f).resize((size, size), Image.LANCZOS)) for f in iio.imread(path)[::STRIDE]]


def labelled(frames: list[np.ndarray], title: str, sub: str, h: int, fs: int) -> list[np.ndarray]:
    """Put a two-line caption strip above every frame."""
    w = frames[0].shape[1]
    strip = Image.new("RGB", (w, h), SURF)
    d = ImageDraw.Draw(strip)
    d.text((2, 2), title, font=ImageFont.truetype(FONT_B, fs), fill=INK)
    d.text((2, 4 + fs + 2), sub, font=ImageFont.truetype(FONT, int(fs * 0.8)), fill=INK2)
    top = np.asarray(strip)
    return [np.vstack([top, f]) for f in frames]


def badge(frame: np.ndarray, text: str, colour) -> np.ndarray:
    im = Image.fromarray(frame.copy())
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT_B, 13)
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=f)
    W = im.width
    d.rectangle([W - (x1 - x0) - 12, 4, W - 4, 4 + (y1 - y0) + 8], fill=colour)
    d.text((W - (x1 - x0) - 8, 4 + 4 - y0), text, font=f, fill=(255, 255, 255))
    return np.asarray(im)


def side_by_side(cols: list[list[np.ndarray]], gap: int) -> list[np.ndarray]:
    n = max(len(c) for c in cols)
    cols = [c + [c[-1]] * (n - len(c)) for c in cols]          # the shorter episode holds its last frame
    H = cols[0][0].shape[0]
    sep = np.full((H, gap, 3), SURF, np.uint8)
    out = []
    for t in range(n):
        parts = []
        for i, c in enumerate(cols):
            parts += ([sep] if i else []) + [c[t]]
        out.append(np.hstack(parts))
    return out


def write_gif(frames: list[np.ndarray], out: Path):
    frames = frames + [frames[-1]] * int(HOLD_S * FPS)
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            Image.fromarray(f).save(f"{tmp}/{i:04d}.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", f"{tmp}/%04d.png", "-vf",
                        "split[a][b];[a]palettegen=max_colors=192:stats_mode=full[p];"
                        "[b][p]paletteuse=dither=bayer:bayer_scale=3", "-loop", "0", str(out)], check=True)
    print(f"{out}  {out.stat().st_size / 1e6:.1f} MB, {len(frames)} frames")


def main(hero: Path):
    s = json.loads((hero / "summary.json").read_text())
    cam = "agentview"
    t0 = next(e for e in s["teacher"] if e["init"] == 0)
    p0 = next(e for e in s["policy"] if e["init"] == 0)
    vid = lambda e: hero / next(v for v in e["videos"] if v.endswith(f"_{cam}.mp4"))

    size = 360
    teacher = labelled(load(vid(t0), size), "Generated robot demo",
                       "contact retargeted from my hand, approach planned", 44, 17)
    policy = labelled(load(vid(p0), size), "Learned policy (ACT)",
                      "sees two cameras and the arm state", 44, 17)
    write_gif(side_by_side([teacher, policy], 12), ROOT / "assets" / "hero.gif")

    size = 200
    tiles = []
    for e in sorted(s["policy"], key=lambda e: e["init"])[:8]:
        fr = load(vid(e), size)
        tag = ("success" if e["success"] else "failure", OK if e["success"] else FAIL)
        tiles.append([badge(f, f"scene {e['init']}", INK2) for f in fr[:-1]] + [badge(fr[-1], *tag)])
    n = max(len(t) for t in tiles)
    tiles = [t + [t[-1]] * (n - len(t)) for t in tiles]
    rows = [side_by_side(tiles[:4], 6), side_by_side(tiles[4:], 6)]
    gap = np.full((6, rows[0][0].shape[1], 3), SURF, np.uint8)
    write_gif([np.vstack([a, gap, b]) for a, b in zip(*rows)], ROOT / "assets" / "policy_8_scenes.gif")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
