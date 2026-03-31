#!/usr/bin/env python3
"""Exact structural comparison between the current tensorial X0 formula and the Aliev L model."""

from __future__ import annotations

import sympy as sp


def paper1_vs_aliev_linear_L_structure() -> dict[str, object]:
    B, omega_k, omega_n, D_J = sp.symbols("B omega_k omega_n D_J", nonzero=True)
    S111, tau_yyyy = sp.symbols("S111 tau_yyyy")
    quartic_visible = sp.Symbol("(1/24) k'_{klmn} C_k^{yy} C_l^{yy} C_m^{yy} C_n^{yy}")
    rk_contracted = sp.Symbol("Σ_l C_l ((3/(4B)) omega_k omega_l C_k C_l + (1/2) Σ_m k'_{klm} C_m)")
    rn = sp.Symbol("r_n")
    Cn = sp.Symbol("C_n")
    quartic_aliev = sp.Symbol("(1/24) Σ k' C C C C")

    K_term3 = sp.Symbol("K_term3")
    ours = sp.expand(
        quartic_visible
        + sp.Rational(18, 35) * S111**2 * tau_yyyy
        - sp.Rational(63, 256) * K_term3
        - rk_contracted**2 / (2 * omega_k)
    )
    aliev = sp.expand(
        quartic_aliev
        + 8 * B * D_J * Cn**2 / omega_n
        - rn**2 / (2 * omega_n)
        - 8 * D_J**3 / B**2
    )

    return {
        "ours_current_compact_linear": ours,
        "aliev_compact_linear": aliev,
        "shared_structural_blocks": (
            "quartic block",
            "negative quadratic square block",
        ),
        "ours_only_blocks": (
            "18/35 S111^2 tau_yyyy",
            "-(63/256) K_term3 normalized term-3 contribution",
            "contracted square built from printed R'_k and R'_{kl} reductions",
        ),
        "aliev_only_blocks": (
            "8 B D_J C_n^2 / omega_n",
            "-8 D_J^3 / B^2",
            "explicit r_n^2 / (2 omega_n) mode family",
        ),
        "operational_gap": (
            "The current tensorial formula is exact but not yet directly evaluable on "
            "the Gaussian payload because S111, tau_yyyy, and the visible quartic "
            "Einstein-style tensor block have not yet been mapped to the payload's "
            "mode-resolved conventions."
        ),
    }


if __name__ == "__main__":
    print(paper1_vs_aliev_linear_L_structure())
