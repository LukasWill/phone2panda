"""Small statistics helpers used in every results table."""
from __future__ import annotations

import math


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a success rate.

    Why not mean +- 1.96 * sqrt(p(1-p)/n)? That normal approximation breaks down near 0% and
    100% (it can even leave [0, 1]), which is exactly where small robot evaluations live.
    With n = 50 episodes, 45/50 = 90% has an interval of roughly 79-96%: two policies whose
    intervals overlap this much have not been shown to differ.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_rate(successes: int, n: int) -> str:
    lo, hi = wilson_ci(successes, n)
    return f"{100 * successes / max(n, 1):.0f}% [{100 * lo:.0f}-{100 * hi:.0f}] (n={n})"


def diff_ci(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """95% interval for the difference of two success rates, k1/n1 - k2/n2 (Newcombe's hybrid score method,
    built from the two Wilson intervals; it behaves well near 0% and 100%, unlike the normal approximation).
    Treats the two evaluations as independent, which overstates the noise slightly when both use the same
    scenes, so the interval errs on the wide side."""
    p1, p2 = k1 / n1, k2 / n2
    (l1, u1), (l2, u2) = wilson_ci(k1, n1), wilson_ci(k2, n2)
    d = p1 - p2
    return (d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))
