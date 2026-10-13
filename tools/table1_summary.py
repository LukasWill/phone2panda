"""Table 1: merge the per-demo replay CSVs and write the summary.

    python tools/table1_summary.py [results/table1_per_demo] [results/table1]

Input: one CSV per human demo (rows = rollouts in LIBERO's eval scenes, all three strategies) with a
`group` column: `main` (the 13 demos that pass every gate) or `natural` (the 6 natural-grasp demos,
whose grasp point is ambiguous). Output: <out>.csv (all rollouts) and <out>.md (the table).
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2p.retarget.strategies import STRATEGIES  # noqa: E402
from p2p.stats import fmt_rate  # noqa: E402


def main(in_dir: Path, out: Path):
    rows = []
    for f in sorted(in_dir.glob("*.csv")):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    for r in rows:
        r["success"] = int(r["success"])
        r["bowl_to_plate_cm"] = float(r["bowl_to_plate_cm"])
    rows.sort(key=lambda r: (r["group"], r["strategy"], r["demo"], int(r["init"])))
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    groups = [g for g in ("main", "natural") if any(r["group"] == g for r in rows)]
    n_init = len({r["init"] for r in rows})
    lines = [f"# Table 1: retargeting strategies, replayed in LIBERO-Goal's eval scenes ({n_init} per demo)", "",
             "Success rate with 95% Wilson interval over all rollouts. Rollouts of one demo are not independent "
             "(same human trajectory), so the per-demo column is the more honest unit.", "",
             "| demos | strategy | success | demos with >= 50% success | most common failure: last waypoint reached |",
             "|---|---|---|---|---|"]
    for g in groups:
        for s in STRATEGIES:
            r = [x for x in rows if x["group"] == g and x["strategy"] == s]
            per_demo = defaultdict(list)
            for x in r:
                per_demo[x["demo"]].append(x["success"])
            good = sum(np.mean(v) >= 0.5 for v in per_demo.values())
            fails = Counter(x["last_waypoint"] for x in r if not x["success"]).most_common(1)
            fail = f"{fails[0][0]} ({fails[0][1]})" if fails else "-"
            lines.append(f"| {g} ({len(per_demo)}) | {s} | {fmt_rate(sum(x['success'] for x in r), len(r))} | "
                         f"{good}/{len(per_demo)} | {fail} |")
    lines += ["", "## Per demo (successes out of %d)" % n_init, "", "| demo | group | " + " | ".join(STRATEGIES) + " |",
              "|---|---|" + "---|" * len(STRATEGIES)]
    for g in groups:
        for d in sorted({r["demo"] for r in rows if r["group"] == g}):
            cnt = [sum(x["success"] for x in rows if x["demo"] == d and x["strategy"] == s) for s in STRATEGIES]
            lines.append(f"| {d} | {g} | " + " | ".join(map(str, cnt)) + " |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(Path(a[0]) if a else Path("results/table1_per_demo"), Path(a[1]) if len(a) > 1 else Path("results/table1"))
