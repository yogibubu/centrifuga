#!/usr/bin/env python3
"""Exact X0 total-so-far rewritten in printed/tensorial primitive coefficients."""

from __future__ import annotations

import sympy as sp

from scripts.paper1_h08_visible_source_x0_summary import paper1_h08_visible_source_x0_summary
from scripts.paper1_octic_x0_decomposition import paper1_octic_x0_decomposition


def paper1_octic_derived_x0_primitives() -> dict[str, object]:
    termwise = paper1_octic_x0_decomposition()["termwise_X0"]
    visible = paper1_h08_visible_source_x0_summary()

    completion_without_h08k4 = sp.expand(
        sum(expr for name, expr in termwise.items() if name != "H08_k4")
    )

    total = sp.expand(
        completion_without_h08k4
        + visible["visible_H08_k4_octic_X0"]
        + visible["visible_term3_reduced_octic_X0"]
        + visible["visible_term5_pure_octic_X0"]
    )

    q004, q022, q040 = sp.symbols("q004 q022 q040")
    w006, w024, w042, w060 = sp.symbols("w006 w024 w042 w060")
    s113, s131, S111 = sp.symbols("s113 s131 S111")
    tau_zzzz, tau_yyzz, tau_yyyy = sp.symbols("tau_zzzz tau_yyzz tau_yyyy")
    Wzzz, Wzzy, Wyyz, Wyyy = sp.symbols("Wzzz Wzzy Wyyz Wyyy")
    S113, S131 = sp.symbols("S113 S131")
    rho_k, r_k, omega_k = sp.symbols("rho_k r_k omega_k")

    primitive_total = sp.expand(
        total.subs(
            {
                q004: sp.Rational(1, 4) * tau_zzzz,
                q022: sp.Rational(3, 2) * tau_yyzz,
                q040: sp.Rational(1, 4) * tau_yyyy,
                w006: Wzzz,
                w024: 2 * Wzzy,
                w042: 2 * Wyyz,
                w060: Wyyy,
                s113: -2 * S113,
                s131: -2 * S131,
                rho_k: r_k,
            }
        )
    )

    return {
        "completion_without_h08k4": completion_without_h08k4,
        "visible_H08_k4_octic_X0": visible["visible_H08_k4_octic_X0"],
        "visible_term3_reduced_octic_X0": visible["visible_term3_reduced_octic_X0"],
        "visible_term5_pure_octic_X0": visible["visible_term5_pure_octic_X0"].subs(rho_k, r_k),
        "primitive_total_so_far": primitive_total,
        "statement": (
            "This is the exact X0 total-so-far rewritten only in the printed/tensorial "
            "primitive coefficients already closed in the current CeDiTT4 program: "
            "quartic tau components, sextic printed W components, printed S components, "
            "the visible quartic H08 tensor block, and the one-index linear scalar r_k."
        ),
    }


if __name__ == "__main__":
    print(paper1_octic_derived_x0_primitives())
