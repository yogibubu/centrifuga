#!/usr/bin/env python3
"""Exact harmonic rotational block H02 for the generic Van Vleck framework."""

from __future__ import annotations

import sympy as sp

import vanvleck_bch_engine as vbe


def build_h02(*, origin: tuple[str, ...] = ("H02",)) -> dict[vbe.Key, sp.Expr]:
    A, B, C = sp.symbols("A B C", real=True)
    return vbe.add_expr(
        vbe.one_term(A, vword=(), jword=(vbe.Jx, vbe.Jx), origin=origin),
        vbe.one_term(B, vword=(), jword=(vbe.Jy, vbe.Jy), origin=origin),
        vbe.one_term(C, vword=(), jword=(vbe.Jz, vbe.Jz), origin=origin),
    )


if __name__ == "__main__":
    print(build_h02())
