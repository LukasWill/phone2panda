"""Generate the printable ArUco tag sheet used for recording.

Why tags at all?
  The phone gives us metric 3D (LiDAR depth + intrinsics), but every recording
  lives in its own camera frame. A big tag taped to the table defines ONE table
  frame shared by all demos (origin, x/y on the table, z up, metric scale).
  Small tags on the bowl's inner bottom and the plate's centre give the object
  positions automatically, so no manual clicking is needed later.

Tags are drawn as vector rectangles (not a scaled bitmap), so the printed
cells are perfectly sharp. The tag ID encodes its size, see TAG_SIZES_MM;
`p2p/human/markers.py` reads the same table.

Usage:  python tools/make_tags_pdf.py  ->  assets/phone2panda_tags_A4.pdf
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2p.config import NOMINAL_TAGS  # single source of truth for ids and sizes  # noqa: E402
from p2p.human.markers import DICT  # noqa: E402

# id -> (printed label, nominal black-square side in mm)
TAG_SIZES_MM = {i: ((role.upper() if i != 3 else "BOWL-small"), side * 1000) for i, (role, side) in NOMINAL_TAGS.items()}


def marker_bits(tag_id: int) -> np.ndarray:
    """6x6 boolean grid (4x4 payload + 1-cell black border); True = black."""
    d = cv2.aruco.getPredefinedDictionary(DICT)
    img = cv2.aruco.generateImageMarker(d, tag_id, 6)  # one pixel per cell
    return img < 128


def draw_tag(c, tag_id, x_mm, y_mm, side_mm):
    """Draw tag with lower-left corner at (x_mm, y_mm). Row 0 of the bitmap is the top."""
    bits = marker_bits(tag_id)
    cell = side_mm / 6.0
    c.setFillColorRGB(0, 0, 0)
    for r in range(6):
        for col in range(6):
            if bits[r, col]:
                c.rect((x_mm + col * cell) * mm, (y_mm + (5 - r) * cell) * mm,
                       cell * mm + 0.01, cell * mm + 0.01, stroke=0, fill=1)


def cut_box(c, x_mm, y_mm, side_mm, margin_mm):
    c.setDash(2, 2)
    c.setStrokeColorRGB(0.6, 0.6, 0.6)
    c.setLineWidth(0.4)
    c.rect((x_mm - margin_mm) * mm, (y_mm - margin_mm) * mm,
           (side_mm + 2 * margin_mm) * mm, (side_mm + 2 * margin_mm) * mm, stroke=1, fill=0)
    c.setDash()


def scale_bar(c, x_mm, y_mm):
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.6)
    c.line(x_mm * mm, y_mm * mm, (x_mm + 100) * mm, y_mm * mm)
    for i in range(11):
        h = 3 if i % 5 == 0 else 1.5
        c.line((x_mm + 10 * i) * mm, y_mm * mm, (x_mm + 10 * i) * mm, (y_mm + h) * mm)
    c.setFont("Helvetica", 8)
    c.drawString(x_mm * mm, (y_mm - 4.5) * mm,
                 "This bar must measure exactly 100 mm. If not, reprint at 100% / 'Actual size' (not 'Fit to page').")


def main(out=Path(__file__).resolve().parents[1] / "assets" / "phone2panda_tags_A4.pdf"):
    out.parent.mkdir(parents=True, exist_ok=True)
    W, H = A4[0] / mm, A4[1] / mm  # 210 x 297
    c = canvas.Canvas(str(out), pagesize=A4)

    # Page 1: the table tag.
    side = TAG_SIZES_MM[0][1]
    x0, y0 = (W - side) / 2, 95
    draw_tag(c, 0, x0, y0, side)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(20 * mm, 275 * mm, "TABLE tag  (ArUco 4x4_50, id 0, 150 mm)  -  do NOT cut this page")
    c.setFont("Helvetica", 9.5)
    # The tag's own axes become the table frame: x = tag right, y = tag up (= away from you
    # when this text reads normally from your seat). p2p/retarget maps this to LIBERO's frame
    # (robot x = forward, y = left), so the page orientation matters.
    lines = [
        "Tape the WHOLE sheet flat on the table, beyond the far edge of your workspace,",
        "rotated so that you can read this text normally from where you sit. This fixes the axes.",
        "It must stay fully visible in every frame: never put objects or your hand over it.",
        "Measure the black square after printing; if it is not 150 mm, tell Claude the real size.",
    ]
    for i, t in enumerate(lines):
        c.drawString(20 * mm, (266 - 5.5 * i) * mm, t)
    scale_bar(c, 20, 55)
    c.showPage()

    # Page 2: object tags (with spares).
    c.setFont("Helvetica-Bold", 13)
    c.drawString(20 * mm, 275 * mm, "OBJECT tags  (cut along the dashed lines)")
    c.setFont("Helvetica", 9.5)
    lines = [
        "BOWL: tape ONE bowl tag flat on the bowl's inner bottom (id 1 = 40 mm; use id 3 = 30 mm if it does not fit).",
        "PLATE: tape the plate tag (id 2 = 50 mm) at the plate's centre.",
        "The tag id tells the code its size, so do not swap sizes. Spares included.",
    ]
    for i, t in enumerate(lines):
        c.drawString(20 * mm, (266 - 5.5 * i) * mm, t)
    # (tag id, x, y) of each tag's lower-left corner in mm; cut boxes must not overlap.
    layout = [(1, 20, 200), (1, 78, 200), (3, 136, 200),
              (2, 20, 130), (2, 88, 130), (3, 156, 130)]
    c.setFont("Helvetica", 8)
    for tag_id, x, y in layout:
        role, s = TAG_SIZES_MM[tag_id]
        draw_tag(c, tag_id, x, y, s)
        cut_box(c, x, y, s, 6)
        c.drawString(x * mm, (y - 10) * mm, f"{role}  id {tag_id}  {s:.0f} mm")
    scale_bar(c, 20, 55)
    c.showPage()
    c.save()
    print("wrote", out)


if __name__ == "__main__":
    main()
