"""Everything measured in the real world lives here (metres).

Printed tags: the sheet was printed at 96% (the 150 mm table square measured 144 mm and the
100 mm scale bar 96 mm), so every tag is 0.96x its nominal size. PnP recovers metric distance
from the tag's true size, so a 4% size error would become a 4% error in every 3D position.
"""

# --- ArUco tags -------------------------------------------------------------------------------
PRINT_SCALE = 0.96
# id -> (role, nominal black-square side as designed in tools/make_tags_pdf.py)
NOMINAL_TAGS = {0: ("table", 0.150), 1: ("bowl", 0.040), 2: ("plate", 0.050), 3: ("bowl", 0.030)}

# --- my real objects ---------------------------------------------------------------------------
HUMAN_BOWL_RIM_DIAMETER = 0.100
HUMAN_BOWL_HEIGHT = 0.058
HUMAN_PLATE_DIAMETER = 0.198
HUMAN_PLATE_HEIGHT = 0.020
