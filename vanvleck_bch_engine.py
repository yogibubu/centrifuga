#!/usr/bin/env python3
"""Generic symbolic Van Vleck BCH engine.

This module lifts the reusable algebraic core out of the quartic Watson script:
- bosonic normal ordering
- non-commuting rotational words
- BCH/Lie-transform recursion
- diagonal/off-diagonal splitting with respect to H0

Unlike the quartic script, this engine is not hardwired to the H12/H22/H30/H40
channel set. Origin filtering is optional and externally controlled.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Callable, Dict, Iterable, Tuple

import sympy as sp


Jx, Jy, Jz = sp.symbols("Jx Jy Jz", commutative=False)
JOPS = (Jx, Jy, Jz)

VibWord = Tuple[Tuple[str, int], ...]
JWord = Tuple[sp.Symbol, ...]
Origin = Tuple[str, ...]
Key = Tuple[VibWord, JWord, Origin]
Series = Dict[int, Dict[Key, sp.Expr]]


def _clean(expr: Dict[Key, sp.Expr]) -> Dict[Key, sp.Expr]:
    return {k: v for k, v in expr.items() if v != 0}


def normalize_origin(origin: Origin) -> Origin:
    return tuple(sorted(origin))


def combine_origin(left: Origin, right: Origin) -> Origin:
    return normalize_origin(left + right)


def rot_degree(key: Key) -> int:
    return len(key[1])


def boson_degree(key: Key) -> int:
    return len(key[0])


@lru_cache(maxsize=None)
def min_normal_ordered_boson_degree(word: VibWord) -> int:
    counts: dict[int, list[int]] = {}
    for op, mode in word:
        pair = counts.setdefault(mode, [0, 0])
        if op == "ad":
            pair[0] += 1
        else:
            pair[1] += 1
    return sum(abs(nad - na) for nad, na in counts.values())


@lru_cache(maxsize=None)
def normal_order_word(word: VibWord) -> Tuple[Tuple[VibWord, sp.Expr], ...]:
    for i in range(len(word) - 1):
        left = word[i]
        right = word[i + 1]
        if left[0] == "a" and right[0] == "ad":
            swapped = word[:i] + (right, left) + word[i + 2 :]
            terms: Dict[VibWord, sp.Expr] = defaultdict(lambda: sp.Integer(0))
            for w2, c2 in normal_order_word(swapped):
                terms[w2] += c2
            if left[1] == right[1]:
                contracted = word[:i] + word[i + 2 :]
                for w2, c2 in normal_order_word(contracted):
                    terms[w2] += c2
            return tuple((w, c) for w, c in terms.items() if c != 0)
    return ((word, sp.Integer(1)),)


def prune_expr(
    expr: Dict[Key, sp.Expr],
    *,
    max_j: int,
    max_v: int,
    keep_origin_fn: Callable[[Origin], bool] | None = None,
) -> Dict[Key, sp.Expr]:
    out: Dict[Key, sp.Expr] = {}
    for key, coeff in expr.items():
        if coeff == 0:
            continue
        if rot_degree(key) > max_j:
            continue
        if boson_degree(key) > max_v:
            continue
        if keep_origin_fn is not None and not keep_origin_fn(key[2]):
            continue
        out[key] = coeff
    return out


def filter_product_expr(
    expr: Dict[Key, sp.Expr],
    *,
    max_j: int,
    max_v: int,
    keep_origin_fn: Callable[[Origin], bool] | None = None,
) -> Dict[Key, sp.Expr]:
    out: Dict[Key, sp.Expr] = {}
    for key, coeff in expr.items():
        if coeff == 0:
            continue
        if rot_degree(key) > max_j:
            continue
        if min_normal_ordered_boson_degree(key[0]) % 2 == 1:
            continue
        if min_normal_ordered_boson_degree(key[0]) > max_v:
            continue
        if keep_origin_fn is not None and not keep_origin_fn(key[2]):
            continue
        out[key] = coeff
    return out


def normal_order_expr(
    expr: Dict[Key, sp.Expr],
    *,
    max_j: int,
    max_v: int,
    keep_origin_fn: Callable[[Origin], bool] | None = None,
) -> Dict[Key, sp.Expr]:
    out: Dict[Key, sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for (vword, jword, origin), coeff in expr.items():
        for nv, c in normal_order_word(vword):
            out[(nv, jword, origin)] += coeff * c
    return _clean(prune_expr(out, max_j=max_j, max_v=max_v, keep_origin_fn=keep_origin_fn))


def add_expr(*exprs: Dict[Key, sp.Expr]) -> Dict[Key, sp.Expr]:
    out: Dict[Key, sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for expr in exprs:
        for key, val in expr.items():
            out[key] += val
    return _clean(out)


def scale_expr(expr: Dict[Key, sp.Expr], factor: sp.Expr) -> Dict[Key, sp.Expr]:
    return _clean({k: factor * v for k, v in expr.items()})


@lru_cache(maxsize=None)
def _mul_keypair_structure(
    va: VibWord,
    ja: JWord,
    vb: VibWord,
    jb: JWord,
    max_j: int,
    max_v: int,
) -> Tuple[Tuple[VibWord, JWord, sp.Expr], ...]:
    jword = ja + jb
    if len(jword) > max_j:
        return ()

    vword = va + vb
    min_vdeg = min_normal_ordered_boson_degree(vword)
    if min_vdeg > max_v or min_vdeg % 2 == 1:
        return ()

    out: Dict[Tuple[VibWord, JWord], sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for nv, c in normal_order_word(vword):
        key = (nv, jword)
        if len(jword) > max_j:
            continue
        if len(nv) > max_v:
            continue
        if min_normal_ordered_boson_degree(nv) > max_v:
            continue
        if min_normal_ordered_boson_degree(nv) % 2 == 1:
            continue
        out[key] += c
    return tuple((nv, jword, v) for (nv, jword), v in out.items() if v != 0)


def mul_expr(
    a: Dict[Key, sp.Expr],
    b: Dict[Key, sp.Expr],
    *,
    max_j: int,
    max_v: int,
    keep_origin_fn: Callable[[Origin], bool] | None = None,
    combine_origin_fn: Callable[[Origin, Origin], Origin] | None = None,
) -> Dict[Key, sp.Expr]:
    if not a or not b:
        return {}
    combine = combine_origin if combine_origin_fn is None else combine_origin_fn
    out: Dict[Key, sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for (va, ja, oa), ca in a.items():
        for (vb, jb, ob), cb in b.items():
            origin = combine(oa, ob)
            for nv, jword, prefactor in _mul_keypair_structure(va, ja, vb, jb, max_j, max_v):
                key = (nv, jword, origin)
                if keep_origin_fn is not None and not keep_origin_fn(key[2]):
                    continue
                out[key] += ca * cb * prefactor
    return _clean(out)


def comm_expr(
    a: Dict[Key, sp.Expr],
    b: Dict[Key, sp.Expr],
    *,
    max_j: int,
    max_v: int,
    keep_origin_fn: Callable[[Origin], bool] | None = None,
    combine_origin_fn: Callable[[Origin, Origin], Origin] | None = None,
) -> Dict[Key, sp.Expr]:
    return add_expr(
        mul_expr(
            a,
            b,
            max_j=max_j,
            max_v=max_v,
            keep_origin_fn=keep_origin_fn,
            combine_origin_fn=combine_origin_fn,
        ),
        scale_expr(
            mul_expr(
                b,
                a,
                max_j=max_j,
                max_v=max_v,
                keep_origin_fn=keep_origin_fn,
                combine_origin_fn=combine_origin_fn,
            ),
            -1,
        ),
    )


def one_term(coeff: sp.Expr, vword: VibWord = (), jword: JWord = (), origin: Origin = ()) -> Dict[Key, sp.Expr]:
    if coeff == 0:
        return {}
    return {(vword, jword, normalize_origin(origin)): coeff}


def delta_energy(vword: VibWord, hbar: sp.Symbol, omega: Iterable[sp.Symbol]) -> sp.Expr:
    w = tuple(omega)
    de = sp.Integer(0)
    for op, mode in vword:
        de += hbar * w[mode] if op == "a" else -hbar * w[mode]
    return sp.simplify(de)


def comm_with_h0(
    expr: Dict[Key, sp.Expr],
    *,
    omega: Iterable[sp.Symbol],
    hbar: sp.Symbol,
    max_j: int,
    max_v: int,
    keep_origin_fn: Callable[[Origin], bool] | None = None,
) -> Dict[Key, sp.Expr]:
    out: Dict[Key, sp.Expr] = {}
    for (vword, jword, origin), coeff in expr.items():
        de = delta_energy(vword, hbar, omega)
        if sp.simplify(de) != 0:
            if keep_origin_fn is None or keep_origin_fn(origin):
                out[(vword, jword, origin)] = coeff * de
    return _clean(prune_expr(out, max_j=max_j, max_v=max_v, keep_origin_fn=keep_origin_fn))


def add_series(a: Series, b: Series, max_order: int) -> Series:
    out: Series = {}
    for n in range(max_order + 1):
        out[n] = add_expr(a.get(n, {}), b.get(n, {}))
    return out


def ad_series(
    s_series: Series,
    y_series: Series,
    y_has_h0: bool,
    *,
    omega: Iterable[sp.Symbol],
    hbar: sp.Symbol,
    max_order: int,
    max_j: int,
    max_v: int,
    keep_origin_fn: Callable[[Origin], bool] | None = None,
    combine_origin_fn: Callable[[Origin, Origin], Origin] | None = None,
) -> Series:
    out: Series = {n: {} for n in range(max_order + 1)}
    for i, si in s_series.items():
        for j, yj in y_series.items():
            if i + j <= max_order and si and yj:
                c = comm_expr(
                    si,
                    yj,
                    max_j=max_j,
                    max_v=max_v,
                    keep_origin_fn=keep_origin_fn,
                    combine_origin_fn=combine_origin_fn,
                )
                out[i + j] = add_expr(out[i + j], c)
    if y_has_h0:
        for i, si in s_series.items():
            if i <= max_order and si:
                out[i] = add_expr(
                    out[i],
                    comm_with_h0(
                        si,
                        omega=omega,
                        hbar=hbar,
                        max_j=max_j,
                        max_v=max_v,
                        keep_origin_fn=keep_origin_fn,
                    ),
                )
    for n in range(max_order + 1):
        out[n] = prune_expr(out[n], max_j=max_j, max_v=max_v, keep_origin_fn=keep_origin_fn)
    return out


def bch_transform(
    h_series: Series,
    s_series: Series,
    *,
    omega: Iterable[sp.Symbol],
    hbar: sp.Symbol,
    max_order: int,
    max_j: int,
    max_v: int,
    keep_origin_fn: Callable[[Origin], bool] | None = None,
    combine_origin_fn: Callable[[Origin, Origin], Origin] | None = None,
) -> Series:
    k: Series = {n: h_series.get(n, {}) for n in range(max_order + 1)}
    term = {n: h_series.get(n, {}) for n in range(max_order + 1)}
    has_h0 = True
    for n in range(1, max_order + 1):
        term = ad_series(
            s_series,
            term,
            has_h0,
            omega=omega,
            hbar=hbar,
            max_order=max_order,
            max_j=max_j,
            max_v=max_v,
            keep_origin_fn=keep_origin_fn,
            combine_origin_fn=combine_origin_fn,
        )
        has_h0 = False
        term = {m: scale_expr(term.get(m, {}), sp.Rational(1, n)) for m in range(max_order + 1)}
        k = add_series(k, term, max_order)
    for n in range(max_order + 1):
        k[n] = prune_expr(
            normal_order_expr(k[n], max_j=max_j, max_v=max_v, keep_origin_fn=keep_origin_fn),
            max_j=max_j,
            max_v=max_v,
            keep_origin_fn=keep_origin_fn,
        )
    return k


def split_diag_offdiag(
    expr: Dict[Key, sp.Expr],
    *,
    omega: Iterable[sp.Symbol],
    hbar: sp.Symbol,
    generator_keep_fn: Callable[[VibWord, JWord], bool] | None = None,
) -> tuple[Dict[Key, sp.Expr], Dict[Key, sp.Expr]]:
    diag: Dict[Key, sp.Expr] = {}
    off: Dict[Key, sp.Expr] = {}
    for (v, j, origin), c in expr.items():
        if generator_keep_fn is not None and not generator_keep_fn(v, j):
            continue
        de = delta_energy(v, hbar, omega)
        if sp.simplify(de) == 0:
            diag[(v, j, origin)] = c
        else:
            off[(v, j, origin)] = c
    return _clean(diag), _clean(off)


def solve_s_order(offdiag: Dict[Key, sp.Expr], *, omega: Iterable[sp.Symbol], hbar: sp.Symbol) -> Dict[Key, sp.Expr]:
    out: Dict[Key, sp.Expr] = {}
    for (v, j, origin), c in offdiag.items():
        de = delta_energy(v, hbar, omega)
        out[(v, j, origin)] = -c / de
    return _clean(out)


def build_effective_to_order(
    h_input: Series,
    *,
    omega: Iterable[sp.Symbol],
    hbar: sp.Symbol,
    max_order: int,
    max_j: int,
    max_v: int,
    keep_origin_fn: Callable[[Origin], bool] | None = None,
    generator_keep_fn: Callable[[VibWord, JWord], bool] | None = None,
    combine_origin_fn: Callable[[Origin, Origin], Origin] | None = None,
) -> tuple[Series, Series]:
    h_series: Series = {0: {}}
    for n in range(1, max_order + 1):
        h_series[n] = _clean(h_input.get(n, {}))
    s_series: Series = {}
    for m in range(1, max_order + 1):
        partial = bch_transform(
            h_series,
            s_series,
            omega=omega,
            hbar=hbar,
            max_order=m,
            max_j=max_j,
            max_v=max_v,
            keep_origin_fn=keep_origin_fn,
            combine_origin_fn=combine_origin_fn,
        )
        _, off_m = split_diag_offdiag(
            partial[m],
            omega=omega,
            hbar=hbar,
            generator_keep_fn=generator_keep_fn,
        )
        s_series[m] = prune_expr(
            solve_s_order(off_m, omega=omega, hbar=hbar),
            max_j=max_j,
            max_v=max_v,
            keep_origin_fn=keep_origin_fn,
        )
    k_full = bch_transform(
        h_series,
        s_series,
        omega=omega,
        hbar=hbar,
        max_order=max_order,
        max_j=max_j,
        max_v=max_v,
        keep_origin_fn=keep_origin_fn,
        combine_origin_fn=combine_origin_fn,
    )
    return k_full, s_series


__all__ = [
    "Jx",
    "Jy",
    "Jz",
    "JOPS",
    "VibWord",
    "JWord",
    "Origin",
    "Key",
    "Series",
    "normalize_origin",
    "combine_origin",
    "normal_order_word",
    "normal_order_expr",
    "add_expr",
    "scale_expr",
    "comm_expr",
    "one_term",
    "delta_energy",
    "comm_with_h0",
    "add_series",
    "ad_series",
    "bch_transform",
    "split_diag_offdiag",
    "solve_s_order",
    "build_effective_to_order",
]
