#!/usr/bin/env python3
"""Operator-level linear-`Dv` derivation scaffolding.

This module makes the carrier operator explicit for the linear-molecule quartic
sector.  The immediate purpose is to bridge the current reduced coefficient
builder and the future full Van Vleck derivation.

The central statement is:

    <v|H_eff^(4,lin)|v> = A(v) [J(J+1)-l^2]^2 + ...

and therefore

    Dv = -A(v)

in the sign convention used in the Aliev-style expansion

    E_d = [-D_J + ...] [J(J+1)-l^2]^2 + ...
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from linear_dv_derivation import build_linear_state_weight


@dataclass(frozen=True)
class LinearQuarticCarrier:
    """Linear-molecule quartic rotational carrier."""

    j: sp.Expr
    l: sp.Expr

    @property
    def polynomial(self) -> sp.Expr:
        return (self.j * (self.j + 1) - self.l * self.l) ** 2


@dataclass(frozen=True)
class LinearQuarticOperatorBlocks:
    """Symbolic operator blocks contributing to the linear quartic sector."""

    equilibrium_scalar: sp.Expr
    coriolis_scalar: sp.Expr
    quartic_direct_scalar: sp.Expr
    cubic_dressing_scalar: sp.Expr

    @property
    def total_scalar(self) -> sp.Expr:
        return (
            self.equilibrium_scalar
            + self.coriolis_scalar
            + self.quartic_direct_scalar
            + self.cubic_dressing_scalar
        )


@dataclass(frozen=True)
class LinearDvProjection:
    """Projected linear-`Dv` coefficient set from operator blocks."""

    carrier: LinearQuarticCarrier
    blocks: LinearQuarticOperatorBlocks
    dv_expression: sp.Expr
    beta_by_mode: tuple[sp.Expr, ...]
    weights_by_mode: tuple[sp.Expr, ...]


def build_linear_quartic_carrier() -> LinearQuarticCarrier:
    j, l = sp.symbols("J l", integer=True, nonnegative=True)
    return LinearQuarticCarrier(j=j, l=l)


def derive_linear_dv_from_mode_coefficients(
    *,
    equilibrium_symbol: str = "D_J",
    mode_kinds: tuple[str, ...],
    beta_by_mode: tuple[sp.Expr, ...],
) -> LinearDvProjection:
    """Build the operator-level linear `Dv` expression from modal coefficients."""

    n_modes = len(mode_kinds)
    if len(beta_by_mode) != n_modes:
        raise ValueError(f"Expected {n_modes} beta coefficients, got {len(beta_by_mode)}.")

    carrier = build_linear_quartic_carrier()
    v = sp.symbols(f"v0:{n_modes}", integer=True, nonnegative=True)
    weights = tuple(build_linear_state_weight(mode_kinds[i], v[i]) for i in range(n_modes))
    dj = sp.Symbol(equilibrium_symbol, real=True)

    cor_syms = sp.symbols(f"beta_cor_0:{n_modes}", real=True)
    dir_syms = sp.symbols(f"beta_dir_0:{n_modes}", real=True)
    cub_syms = sp.symbols(f"beta_cub_0:{n_modes}", real=True)
    beta_total = tuple(sp.simplify(beta_by_mode[i]) for i in range(n_modes))

    blocks = LinearQuarticOperatorBlocks(
        equilibrium_scalar=-dj * carrier.polynomial,
        coriolis_scalar=sum(cor_syms[i] * weights[i] for i in range(n_modes)) * carrier.polynomial,
        quartic_direct_scalar=sum(dir_syms[i] * weights[i] for i in range(n_modes)) * carrier.polynomial,
        cubic_dressing_scalar=sum(cub_syms[i] * weights[i] for i in range(n_modes)) * carrier.polynomial,
    )
    dv = sp.simplify(dj - sum(beta_total[i] * weights[i] for i in range(n_modes)))
    return LinearDvProjection(
        carrier=carrier,
        blocks=blocks,
        dv_expression=dv,
        beta_by_mode=beta_total,
        weights_by_mode=weights,
    )


def derive_reduced_linear_operator_projection(n_modes: int, mode_kinds: tuple[str, ...]) -> LinearDvProjection:
    """Return the current reduced operator-level projection template.

    This exposes the correct carrier structure at operator level while the full
    Aliev-equivalent beta builder is still under derivation.
    """

    if len(mode_kinds) != n_modes:
        raise ValueError(f"Expected {n_modes} mode kinds, got {len(mode_kinds)}.")

    omega = sp.symbols(f"omega0:{n_modes}", positive=True)
    C = {
        (i, j): sp.Symbol(f"C_{i}_{j}", real=True)
        for i in range(n_modes)
        for j in range(i + 1, n_modes)
    }
    phi3_diag = {i: sp.Symbol(f"Phi_{i}{i}{i}", real=True) for i in range(n_modes)}
    phi3_semidiag = {(i, k): sp.Symbol(f"Phi_{i}{i}{k}", real=True) for i in range(n_modes) for k in range(n_modes)}
    phi4_semidiag = {(i, k): sp.Symbol(f"Phi_{i}{i}{i}{k}", real=True) for i in range(n_modes) for k in range(n_modes)}
    partner_weight = sp.Symbol("p", real=True)

    beta: list[sp.Expr] = []
    for i in range(n_modes):
        cor_expr = sp.Integer(0)
        quart_expr = sp.Integer(0)
        cubic_expr = sp.Integer(0)
        for j in range(n_modes):
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            cor_expr += C[(a, b)] ** 2 / (2 * omega[i] * (omega[i] + omega[j]))
        for k in range(n_modes):
            quart_expr += phi4_semidiag[(i, k)] / (8 * omega[i] * (omega[i] + omega[k]))
            cubic_expr -= (phi3_diag[i] * phi3_semidiag[(i, k)]) / (
                (2 * omega[i] + omega[k]) * 8 * omega[i] * (omega[i] + omega[k])
            )
            if k != i:
                quart_expr += partner_weight * phi4_semidiag[(k, i)] / (8 * omega[k] * (omega[k] + omega[i]))
                cubic_expr -= partner_weight * (phi3_diag[k] * phi3_semidiag[(k, i)]) / (
                    (2 * omega[k] + omega[i]) * 8 * omega[k] * (omega[k] + omega[i])
                )
        beta.append(sp.simplify(cor_expr + quart_expr + cubic_expr))

    return derive_linear_dv_from_mode_coefficients(
        equilibrium_symbol="D_J",
        mode_kinds=mode_kinds,
        beta_by_mode=tuple(beta),
    )
