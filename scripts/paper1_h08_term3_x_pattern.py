#!/usr/bin/env python3
"""Exact X^5/X^4 pattern for the visible H08 term-3 source probes."""

from __future__ import annotations

import sympy as sp

from scripts.paper1_h08_term3_onemode_source_probe import paper1_h08_term3_onemode_source_probe
from scripts.paper1_h08_term3_twomode_probe import paper1_h08_term3_twomode_probe


def paper1_h08_term3_x_pattern() -> dict[str, object]:
    one = paper1_h08_term3_onemode_source_probe()
    two = paper1_h08_term3_twomode_probe()
    r_one = sp.simplify(one["X5"] / one["X4"])
    r_two = sp.simplify(two["X5"] / two["X4"])
    return {
        "one_mode_X4": sp.expand(one["X4"]),
        "one_mode_X5": sp.expand(one["X5"]),
        "one_mode_ratio": r_one,
        "two_mode_X4": sp.expand(two["X4"]),
        "two_mode_X5": sp.expand(two["X5"]),
        "two_mode_ratio": r_two,
        "statement": (
            "For both the one-mode and first two-mode visible source probes of "
            "the H08 term -R'_k R_l R_m R'_{lm}/(omega_k omega_l omega_m), the "
            "K=0 source polynomial satisfies X5/X4 = -63/20 exactly. The source "
            "therefore carries a fixed over-octic contamination pattern that "
            "must be removed by the final effective reduction before a true "
            "octic scalar can be read."
        ),
    }


if __name__ == "__main__":
    print(paper1_h08_term3_x_pattern())
