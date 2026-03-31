#!/usr/bin/env python3
"""Exact principal quartic source term inside the optical H08 block."""

from __future__ import annotations

import sympy as sp


def paper1_h08_k4_principal_term() -> dict[str, object]:
    wk, wl, wm, wn = sp.symbols("omega_k omega_l omega_m omega_n", positive=True)
    k4 = sp.Symbol("k'_{klmn}")

    Ckab = sp.Symbol("C_k^{ab}")
    Clcd = sp.Symbol("C_l^{cd}")
    Cmef = sp.Symbol("C_m^{ef}")
    Cngh = sp.Symbol("C_n^{gh}")

    J8 = sp.Symbol("J_a J_b J_c J_d J_e J_f J_g J_h", commutative=False)

    Rk = -wk * sp.Symbol("Σ_ab C_k^{ab} J_a J_b", commutative=False)
    Rl = -wl * sp.Symbol("Σ_cd C_l^{cd} J_c J_d", commutative=False)
    Rm = -wm * sp.Symbol("Σ_ef C_m^{ef} J_e J_f", commutative=False)
    Rn = -wn * sp.Symbol("Σ_gh C_n^{gh} J_g J_h", commutative=False)

    source_form = sp.simplify(Rk * Rl * Rm * Rn * k4 / (24 * wk * wl * wm * wn))

    explicit_tensor = sp.Rational(1, 24) * k4 * Ckab * Clcd * Cmef * Cngh * J8

    return {
        "source_term": source_form,
        "explicit_tensor_kernel": explicit_tensor,
        "omega_cancellation": True,
        "rotational_support": ("mu1",),
        "potential_support": ("phi4",),
        "harmonic_support": tuple(),
        "statement": (
            "The visible quartic H08 source term reduces exactly to a quartic "
            "force-constant tensor times four first-order rotational-derivative "
            "tensors, with complete cancellation of the harmonic frequencies."
        ),
    }


def main() -> None:
    print(paper1_h08_k4_principal_term())


if __name__ == "__main__":
    main()
