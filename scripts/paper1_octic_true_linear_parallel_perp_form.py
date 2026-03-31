#!/usr/bin/env python3
"""Clean parallel/perpendicular form of the current CeDiTT4 linear octic equations."""

from __future__ import annotations

import sympy as sp


def paper1_octic_true_linear_parallel_perp_form() -> dict[str, object]:
    B = sp.Symbol("B", nonzero=True)
    tau_perp = sp.Symbol("tau_perp")
    S_perp = sp.Symbol("S_perp")

    omega_n = sp.Symbol("omega_n^parallel", nonzero=True)
    omega_m = sp.Symbol("omega_m^parallel", nonzero=True)
    C_n = sp.Symbol("C_n^parallel")
    C_m = sp.Symbol("C_m^parallel")
    C_p = sp.Symbol("C_p^parallel")
    C_q = sp.Symbol("C_q^parallel")
    k3_nmp = sp.Symbol("k3_{nmp}^parallel")
    k4_nmpq = sp.Symbol("k4_{nmpq}^parallel")

    r_nm = sp.Symbol("r_{nm}^parallel")
    r_n = sp.Symbol("r_n^parallel")

    r_nm_def = sp.Eq(
        r_nm,
        sp.Rational(3, 4) * omega_n * omega_m * C_n * C_m / B
        + sp.Rational(1, 2) * sp.Symbol("Σ_p k3_{nmp}^parallel C_p^parallel"),
    )
    r_n_def = sp.Eq(r_n, sp.Symbol("Σ_m C_m^parallel r_{nm}^parallel"))

    quartic_block = sp.Symbol("L_4^parallel")
    quartic_block_def = sp.Eq(
        quartic_block,
        sp.Rational(1, 24) * sp.Symbol("Σ_{nmpq} k4_{nmpq}^parallel C_n^parallel C_m^parallel C_p^parallel C_q^parallel"),
    )

    term3_block = sp.Symbol("L_3^parallel")
    term3_block_def = sp.Eq(
        term3_block,
        -sp.Rational(63, 256)
        * sp.Symbol("Σ_n (r_n^parallel/omega_n^parallel)")
        * sp.Symbol("Σ_{mp} C_m^parallel C_p^parallel r_{mp}^parallel"),
    )

    term5_block = sp.Symbol("L_5^parallel")
    term5_block_def = sp.Eq(
        term5_block,
        -sp.Rational(1, 2) * sp.Symbol("Σ_n (r_n^parallel)^2/omega_n^parallel"),
    )

    pure_rotational = sp.Symbol("L_pure^parallel")
    pure_rotational_def = sp.Eq(
        pure_rotational,
        quartic_block + term3_block + term5_block,
    )

    degenerate_block = sp.Symbol("L_deg^perp")
    degenerate_block_def = sp.Eq(
        degenerate_block,
        sp.Rational(18, 35) * S_perp**2 * tau_perp,
    )

    total = sp.Symbol("L")
    total_def = sp.Eq(total, pure_rotational + degenerate_block)

    tau_perp_def = sp.Eq(
        tau_perp,
        sp.Symbol("tau_yyyy"),
    )

    return {
        "tau_perp_definition": tau_perp_def,
        "r_nm_definition": r_nm_def,
        "r_n_definition": r_n_def,
        "quartic_block_definition": quartic_block_def,
        "term3_block_definition": term3_block_def,
        "term5_block_definition": term5_block_def,
        "pure_rotational_definition": pure_rotational_def,
        "degenerate_block_definition": degenerate_block_def,
        "total_definition": total_def,
        "statement": (
            "In the exact linear CeDiTT4 projection, the current octic result is "
            "best written as a sum of a pure rotational parallel-mode scalar branch "
            "and a separate degenerate perpendicular branch. The latter is written "
            "with tau_perp instead of tau_yyyy and with a perpendicular S-sector "
            "symbol S_perp to emphasize that this is not a parallel-mode scalar."
        ),
    }


if __name__ == "__main__":
    print(paper1_octic_true_linear_parallel_perp_form())
