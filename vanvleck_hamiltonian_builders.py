#!/usr/bin/env python3
"""Generic tensorial Hamiltonian builders for the symbolic Van Vleck engine.

The notation follows the same operator conventions already used in the quartic
Watson script:
- H_{r0}: pure vibrational potential term of q-degree r
- H_{r2}: rotational derivative term of q-degree r times J_a J_b

This module is generic in the q-degree and therefore can seed the BCH engine
past the quartic ceiling without changing the algebraic backend.
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable

import sympy as sp

import vanvleck_bch_engine as vbe


def _q_prefactor(indices: tuple[int, ...], hbar: sp.Symbol, omega: tuple[sp.Symbol, ...]) -> sp.Expr:
    pref = sp.Integer(1)
    for idx in indices:
        pref *= sp.sqrt(hbar / (2 * omega[idx]))
    return sp.simplify(pref)


def build_potential_block(
    order_q: int,
    *,
    phi: sp.MutableDenseNDimArray,
    omega: tuple[sp.Symbol, ...],
    hbar: sp.Symbol,
    origin: tuple[str, ...],
) -> Dict[vbe.Key, sp.Expr]:
    """Build a pure vibrational block H_{r0} with q-degree r."""
    out: Dict[vbe.Key, sp.Expr] = {}
    n_modes = len(omega)
    for indices in itertools.product(range(n_modes), repeat=order_q):
        coeff = sp.Rational(1, sp.factorial(order_q)) * phi[indices] * _q_prefactor(indices, hbar, omega)
        for ops in itertools.product(("a", "ad"), repeat=order_q):
            vword = tuple((ops[i], indices[i]) for i in range(order_q))
            out = vbe.add_expr(out, vbe.one_term(coeff, vword=vword, jword=(), origin=origin))
    return out


def build_rotder_block(
    order_q: int,
    *,
    mu: sp.MutableDenseNDimArray,
    omega: tuple[sp.Symbol, ...],
    hbar: sp.Symbol,
    origin: tuple[str, ...],
    rot_pairs: set[tuple[int, int]] | None = None,
    diag_rot_only: bool = False,
) -> Dict[vbe.Key, sp.Expr]:
    """Build a rotational derivative block H_{r2} with q-degree r and J_a J_b carrier."""
    out: Dict[vbe.Key, sp.Expr] = {}
    n_modes = len(omega)
    for a in range(3):
        for b in range(3):
            if rot_pairs is not None and (a, b) not in rot_pairs:
                continue
            if diag_rot_only and a != b:
                continue
            for indices in itertools.product(range(n_modes), repeat=order_q):
                coeff = (
                    sp.Rational(1, 2 * sp.factorial(order_q))
                    * mu[(a, b) + indices]
                    * _q_prefactor(indices, hbar, omega)
                )
                jword = (vbe.JOPS[a], vbe.JOPS[b])
                for ops in itertools.product(("a", "ad"), repeat=order_q):
                    vword = tuple((ops[i], indices[i]) for i in range(order_q))
                    out = vbe.add_expr(out, vbe.one_term(coeff, vword=vword, jword=jword, origin=origin))
    return out


def symbolic_tensor(shape: tuple[int, ...], prefix: str) -> sp.MutableDenseNDimArray:
    size = 1
    for dim in shape:
        size *= dim
    return sp.MutableDenseNDimArray(sp.symbols(f"{prefix}_0:{size}", real=True), shape)


def build_generic_blocks(
    *,
    n_modes: int,
    potential_orders: Iterable[int] = (),
    rotder_orders: Iterable[int] = (),
    diag_rot_only: bool = False,
    rot_pairs: set[tuple[int, int]] | None = None,
) -> dict[str, object]:
    """Build a generic family of H_{r0}/H_{r2} blocks with symbolic tensors."""
    hbar = sp.symbols("hbar", positive=True)
    omega = tuple(sp.symbols(f"omega0:{n_modes}", positive=True))
    blocks: dict[str, Dict[vbe.Key, sp.Expr]] = {}

    for r in potential_orders:
        phi = symbolic_tensor((n_modes,) * r, f"phi{r}")
        label = f"H{r}0"
        blocks[label] = build_potential_block(r, phi=phi, omega=omega, hbar=hbar, origin=(label,))

    for r in rotder_orders:
        mu = symbolic_tensor((3, 3) + (n_modes,) * r, f"mu{r}")
        label = f"H{r}2"
        blocks[label] = build_rotder_block(
            r,
            mu=mu,
            omega=omega,
            hbar=hbar,
            origin=(label,),
            rot_pairs=rot_pairs,
            diag_rot_only=diag_rot_only,
        )

    hprime: Dict[vbe.Key, sp.Expr] = {}
    for block in blocks.values():
        hprime = vbe.add_expr(hprime, block)

    return {
        "hprime": hprime,
        "blocks": blocks,
        "omega": omega,
        "hbar": hbar,
    }


__all__ = [
    "build_potential_block",
    "build_rotder_block",
    "symbolic_tensor",
    "build_generic_blocks",
]
