#!/usr/bin/env python3
"""Exact total X0 contribution derived so far in the CeDiTT4 octic program."""

from __future__ import annotations

import sympy as sp

from scripts.paper1_h08_visible_source_x0_summary import paper1_h08_visible_source_x0_summary
from scripts.paper1_octic_x0_decomposition import paper1_octic_x0_decomposition


def paper1_octic_derived_x0_total() -> dict[str, object]:
    completion = paper1_octic_x0_decomposition()
    visible = paper1_h08_visible_source_x0_summary()
    total = sp.expand(
        completion["total_X0"]
        + visible["visible_term3_reduced_octic_X0"]
        + visible["visible_term5_pure_octic_X0"]
    )
    return {
        "completion_X0_total": completion["total_X0"],
        "visible_H08_k4_X0": visible["visible_H08_k4_octic_X0"],
        "visible_term3_reduced_X0": visible["visible_term3_reduced_octic_X0"],
        "visible_term5_pure_X0": visible["visible_term5_pure_octic_X0"],
        "visible_term1_X0": visible["visible_term1_naive_X0"],
        "visible_term4_X0": visible["visible_term4_true_linear_X0"],
        "derived_X0_total_so_far": total,
        "statement": (
            "This is the exact X0 contribution derived so far by combining the Eq. (99) "
            "completion channels with the rigorously isolated visible one-line H08 source channels."
        ),
    }


if __name__ == "__main__":
    print(paper1_octic_derived_x0_total())
