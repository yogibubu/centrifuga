#!/usr/bin/env python3
"""Perturbative-input composer for the generic Van Vleck BCH engine."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable

import sympy as sp

import vanvleck_bch_engine as vbe


def compose_input_series(order_to_blocks: dict[int, Iterable[Dict[vbe.Key, sp.Expr]]]) -> vbe.Series:
    """Compose a perturbative input series from explicit block assignments."""
    out: vbe.Series = {}
    max_order = max(order_to_blocks) if order_to_blocks else 0
    for order in range(1, max_order + 1):
        expr: Dict[vbe.Key, sp.Expr] = {}
        for block in order_to_blocks.get(order, ()):
            expr = vbe.add_expr(expr, block)
        out[order] = expr
    return out


def collapsed_class_symbols(symbol_names: Iterable[str]) -> tuple[sp.Symbol, ...]:
    return tuple(sp.Symbol(name, real=True) for name in symbol_names)


def collapse_block_coefficients(
    block: Dict[vbe.Key, sp.Expr],
    *,
    class_symbol: sp.Symbol,
    weight: sp.Expr = sp.Integer(1),
) -> Dict[vbe.Key, sp.Expr]:
    """Replace all block coefficients by a collapsed class symbol times a common weight."""
    out: Dict[vbe.Key, sp.Expr] = {}
    for key in block:
        out[key] = class_symbol * weight
    return out


def perturbative_origin_catalog(series: vbe.Series) -> dict[int, tuple[vbe.Origin, ...]]:
    """List the origin classes present at each perturbative order."""
    out: dict[int, tuple[vbe.Origin, ...]] = {}
    for order, expr in series.items():
        origins = sorted({origin for (_v, _j, origin) in expr})
        out[order] = tuple(origins)
    return out


__all__ = [
    "compose_input_series",
    "collapsed_class_symbols",
    "collapse_block_coefficients",
    "perturbative_origin_catalog",
]
