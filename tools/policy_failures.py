"""Sort a policy's evaluation episodes the way tools/failure_modes.py sorts generated data.

    python tools/policy_failures.py results/act/ours_all_s0/eval.json results/act/official_s0/eval.json ...

Needs eval.json files written after the bowl fields were added to p2p/policy/act.py (bowl_lift_cm and
bowl_to_plate_cm per episode). For every file, prints one markdown row:
  success rate | never lifted | left off plate | on plate, > 3 cm | within 3 cm, not counted |
  successes: median steps, median final bowl-to-plate distance, median lift
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2p.stats import fmt_rate  # noqa: E402

PLATE_RADIUS_CM, LIBERO_TOL_CM, LIFTED_CM = 6.85, 3.0, 2.0


def kind(e: dict) -> str:
    if e["success"]:
        return "success"
    if e["bowl_lift_cm"] < LIFTED_CM:
        return "never lifted"
    if e["bowl_to_plate_cm"] > PLATE_RADIUS_CM:
        return "left off plate"
    if e["bowl_to_plate_cm"] >= LIBERO_TOL_CM:
        return "on plate, > 3 cm"
    return "within 3 cm, not counted"


def row(path: str) -> str:
    s = json.loads(Path(path).read_text())
    eps = s["episodes"]
    if "bowl_lift_cm" not in eps[0]:
        return f"| {path} | {s['rate']} | (no bowl fields: re-evaluate with the current act.py) |"
    k = [kind(e) for e in eps]
    ok = [e for e in eps if e["success"]]
    med = lambda key: f"{np.median([e[key] for e in ok]):.1f}" if ok else "-"
    p = Path(path)
    name = f"{p.parent.parent.name}/{p.parent.name}" if p.stem == "eval" else f"{p.parent.name}/{p.stem}"
    cols = [fmt_rate(len(ok), len(eps)), s.get("execution", "n_action_steps=25")] + \
           [str(k.count(c)) for c in ("never lifted", "left off plate", "on plate, > 3 cm", "within 3 cm, not counted")] + \
           [med("steps"), med("bowl_to_plate_cm"), med("bowl_lift_cm")]
    return f"| {name} | " + " | ".join(cols) + " |"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print("| run | success | execution | never lifted | left off plate | on plate, > 3 cm | within 3 cm, not counted "
          "| successes: steps | place error (cm) | lift (cm) |")
    print("|---" * 10 + "|")
    for p in sys.argv[1:]:
        print(row(p))
