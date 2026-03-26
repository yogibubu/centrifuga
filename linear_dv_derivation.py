#!/usr/bin/env python3
"""Symbolic scaffolding for deriving the linear-molecule `Dv` equations.

This file does not claim to be the final Aliev-complete implementation.
Its purpose is narrower and more important for the long-term design:

- define the projected observable target symbolically,
- separate the perturbative building blocks at operator level,
- make the linear reduction `Dv = D_J - sum beta_i w_i(v)` explicit,
- provide a stable place where the reduced builder can later be replaced by
  the full Aliev-equivalent expressions without changing the public API.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class LinearDvEquationSet:
    """Symbolic equations for the present linear `Dv` derivation layer."""

    dv_expression: sp.Expr
    beta_by_mode: tuple[sp.Expr, ...]
    weights_by_mode: tuple[sp.Expr, ...]
    coriolis_by_mode: tuple[sp.Expr, ...]
    quartic_direct_by_mode: tuple[sp.Expr, ...]
    cubic_dressing_by_mode: tuple[sp.Expr, ...]


def build_linear_state_weight(mode_kind: str, v_symbol: sp.Symbol) -> sp.Expr:
    kind = str(mode_kind).strip().lower()
    if kind == "parallel":
        return v_symbol + sp.Rational(1, 2)
    if kind == "perpendicular":
        return v_symbol + 1
    raise ValueError(f"Unsupported mode kind {mode_kind!r}.")


def derive_reduced_linear_dv_equations(n_modes: int, mode_kinds: tuple[str, ...]) -> LinearDvEquationSet:
    """Return the current symbolic `Dv` equations in reduced semi-diagonal form.

    The returned expressions mirror the equations implemented numerically in
    `quartic_observable_model.py`.  They are a derivation scaffold, not the
    final Aliev-complete result.
    """

    if len(mode_kinds) != n_modes:
        raise ValueError(f"Expected {n_modes} mode kinds, got {len(mode_kinds)}.")

    DJ = sp.Symbol("D_J", real=True)
    omega = sp.symbols(f"omega0:{n_modes}", positive=True)
    v = sp.symbols(f"v0:{n_modes}", integer=True, nonnegative=True)

    C = {
        (i, j): sp.Symbol(f"C_{i}_{j}", real=True)
        for i in range(n_modes)
        for j in range(n_modes)
        if i < j
    }
    phi3_diag = {i: sp.Symbol(f"Phi_{i}{i}{i}", real=True) for i in range(n_modes)}
    phi3_semidiag = {(i, k): sp.Symbol(f"Phi_{i}{i}{k}", real=True) for i in range(n_modes) for k in range(n_modes)}
    phi4_semidiag = {(i, k): sp.Symbol(f"Phi_{i}{i}{i}{k}", real=True) for i in range(n_modes) for k in range(n_modes)}
    partner_weight = sp.Symbol("p", real=True)

    weights = tuple(build_linear_state_weight(mode_kinds[i], v[i]) for i in range(n_modes))
    cor_terms = [sp.Integer(0) for _ in range(n_modes)]
    quart_terms = [sp.Integer(0) for _ in range(n_modes)]
    cubic_terms = [sp.Integer(0) for _ in range(n_modes)]

    for i in range(n_modes):
        for j in range(i + 1, n_modes):
            sij = C[(i, j)] ** 2 / (omega[i] + omega[j])
            cor_terms[i] += sij / (2 * omega[i])
            cor_terms[j] += sij / (2 * omega[j])

    for i in range(n_modes):
        for k in range(n_modes):
            s4 = phi4_semidiag[(i, k)] / (8 * omega[i] * (omega[i] + omega[k]))
            s33 = -(phi3_diag[i] * phi3_semidiag[(i, k)]) / (
                (2 * omega[i] + omega[k]) * 8 * omega[i] * (omega[i] + omega[k])
            )
            quart_terms[i] += s4
            cubic_terms[i] += s33
            if k != i:
                quart_terms[k] += partner_weight * s4
                cubic_terms[k] += partner_weight * s33

    beta = tuple(sp.simplify(cor_terms[i] + quart_terms[i] + cubic_terms[i]) for i in range(n_modes))
    dv = sp.simplify(DJ - sum(beta[i] * weights[i] for i in range(n_modes)))

    return LinearDvEquationSet(
        dv_expression=dv,
        beta_by_mode=beta,
        weights_by_mode=weights,
        coriolis_by_mode=tuple(sp.simplify(x) for x in cor_terms),
        quartic_direct_by_mode=tuple(sp.simplify(x) for x in quart_terms),
        cubic_dressing_by_mode=tuple(sp.simplify(x) for x in cubic_terms),
    )


def render_equation_summary(n_modes: int, mode_kinds: tuple[str, ...]) -> str:
    """Return a compact plain-text summary of the current symbolic equations."""

    eqs = derive_reduced_linear_dv_equations(n_modes, mode_kinds)
    lines = ["Current reduced symbolic linear-Dv equations:", f"Dv = {sp.sstr(eqs.dv_expression)}", "beta_i terms:"]
    for i, expr in enumerate(eqs.beta_by_mode):
        lines.append(f"  beta_{i} = {sp.sstr(expr)}")
    return "\n".join(lines)
