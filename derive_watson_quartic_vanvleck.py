#!/usr/bin/env python3
"""Symbolic Van Vleck derivation of quartic rotational Watson constants.

Extends the perturbative construction to fourth order (lambda^4) using a
Lie-transform series with bosonic normal ordering and non-commuting J operators.
"""

from __future__ import annotations

import argparse
import itertools
import random
from collections import defaultdict
from functools import lru_cache
from typing import Dict, Iterable, Tuple

import sympy as sp


# --- Rotational operators (non-commuting) -------------------------------------------
Jx, Jy, Jz = sp.symbols("Jx Jy Jz", commutative=False)
JOPS = (Jx, Jy, Jz)

VibWord = Tuple[Tuple[str, int], ...]  # ("ad"|"a", mode)
JWord = Tuple[sp.Symbol, ...]
Origin = Tuple[str, ...]
Key = Tuple[VibWord, JWord, Origin]
Series = Dict[int, Dict[Key, sp.Expr]]
PRUNE_MAX_J = 4
PRUNE_MAX_V = 4
CHANNEL_AWARE = False
TARGET_SYMBOL_CLASS = ""
ALLOWED_CONTRIBUTION_CLASSES = {
    ("H12",),
    ("H22",),
    ("H30",),
    ("H40",),
    ("H12", "H12"),
    ("H12", "H30"),
    ("H22",),
    ("H30", "H30"),
    ("H40",),
}


def _clean(expr: Dict[Key, sp.Expr]) -> Dict[Key, sp.Expr]:
    out: Dict[Key, sp.Expr] = {}
    for k, v in expr.items():
        if v != 0:
            out[k] = v
    return out


def rot_degree(key: Key) -> int:
    return len(key[1])


def boson_degree(key: Key) -> int:
    return len(key[0])


@lru_cache(maxsize=None)
def min_normal_ordered_boson_degree(word: VibWord) -> int:
    """Lower bound on boson word length after all same-mode contractions.

    For each mode, the normal-ordered residual is |n_ad - n_a|. Summing over
    modes gives the minimal possible surviving boson degree after contractions.
    """
    counts: dict[int, list[int]] = {}
    for op, mode in word:
        pair = counts.setdefault(mode, [0, 0])
        if op == "ad":
            pair[0] += 1
        else:
            pair[1] += 1
    return sum(abs(nad - na) for nad, na in counts.values())


def normalize_origin(origin: Origin) -> Origin:
    return tuple(sorted(origin))


def combine_origin(left: Origin, right: Origin) -> Origin:
    merged = normalize_origin(left + right)
    if len(merged) <= 2:
        return merged
    # For nested BCH products we only need the physical contribution class,
    # not the full multiplication history. Keep the highest-order irreducible
    # class among the allowed channels.
    priority = [
        ("H12", "H30"),
        ("H30", "H30"),
        ("H12", "H12"),
        ("H22",),
        ("H40",),
        ("H12",),
        ("H30",),
    ]
    counts = {label: merged.count(label) for label in ("H12", "H22", "H30", "H40")}
    candidates = []
    if counts["H12"] >= 1 and counts["H30"] >= 1:
        candidates.append(("H12", "H30"))
    if counts["H30"] >= 2:
        candidates.append(("H30", "H30"))
    if counts["H12"] >= 2:
        candidates.append(("H12", "H12"))
    if counts["H22"] >= 1:
        candidates.append(("H22",))
    if counts["H40"] >= 1:
        candidates.append(("H40",))
    if counts["H12"] >= 1:
        candidates.append(("H12",))
    if counts["H30"] >= 1:
        candidates.append(("H30",))
    for cls in priority:
        if cls in candidates:
            return cls
    return merged


def keep_origin(origin: Origin) -> bool:
    origin = normalize_origin(origin)
    return origin in ALLOWED_CONTRIBUTION_CLASSES


def filter_product_expr(expr: Dict[Key, sp.Expr], max_j: int = 4, max_v: int = 4) -> Dict[Key, sp.Expr]:
    max_j_eff = min(max_j, PRUNE_MAX_J)
    max_v_eff = min(max_v, PRUNE_MAX_V)
    out: Dict[Key, sp.Expr] = {}
    for key, coeff in expr.items():
        if coeff == 0:
            continue
        if rot_degree(key) > max_j_eff:
            continue
        if min_normal_ordered_boson_degree(key[0]) % 2 == 1:
            continue
        if min_normal_ordered_boson_degree(key[0]) > max_v_eff:
            continue
        if not keep_origin(key[2]):
            continue
        out[key] = coeff
    return out


def prune_expr(expr: Dict[Key, sp.Expr], max_j: int = 4, max_v: int = 4) -> Dict[Key, sp.Expr]:
    max_j_eff = min(max_j, PRUNE_MAX_J)
    max_v_eff = min(max_v, PRUNE_MAX_V)
    out: Dict[Key, sp.Expr] = {}
    for key, coeff in expr.items():
        if coeff == 0:
            continue
        if rot_degree(key) > max_j_eff:
            continue
        if boson_degree(key) > max_v_eff:
            continue
        out[key] = coeff
    return out


def keep_for_generator(vword: VibWord, jword: JWord) -> bool:
    """Heuristic channel filter for S^(n) construction.

    For quartic rotational targets, dominant channels feeding S are low-J off-diagonal
    blocks from Hrv1/Hrv2/V3/V4. Keeping j<=2 avoids explosive high-J off-diagonal
    sectors that do not affect the J^4 projected block in this workflow.
    """
    if not CHANNEL_AWARE:
        return True
    return len(jword) <= 2 and len(vword) <= max(3, PRUNE_MAX_V)


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


def normal_order_expr(expr: Dict[Key, sp.Expr]) -> Dict[Key, sp.Expr]:
    out: Dict[Key, sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for (vword, jword, origin), coeff in expr.items():
        for nv, c in normal_order_word(vword):
            out[(nv, jword, origin)] += coeff * c
    return _clean(prune_expr(out))


def add_expr(*exprs: Dict[Key, sp.Expr]) -> Dict[Key, sp.Expr]:
    out: Dict[Key, sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for e in exprs:
        for k, v in e.items():
            out[k] += v
    return _clean(out)


def scale_expr(expr: Dict[Key, sp.Expr], factor: sp.Expr) -> Dict[Key, sp.Expr]:
    return _clean({k: factor * v for k, v in expr.items()})


@lru_cache(maxsize=None)
def mul_keypair(
    va: VibWord,
    ja: JWord,
    oa: Origin,
    vb: VibWord,
    jb: JWord,
    ob: Origin,
) -> Tuple[Tuple[Key, sp.Expr], ...]:
    """Cached structural product of two operator monomials.

    This routine performs all pruning and bosonic normal ordering at the level of
    a single key pair. The expensive symbolic scalar multiplication is left to
    ``mul_expr`` so that repeated BCH commutators can reuse the structural part.
    """
    origin = combine_origin(oa, ob)
    if not keep_origin(origin):
        return ()
    jword = ja + jb
    if len(jword) > PRUNE_MAX_J:
        return ()

    vword = va + vb
    min_vdeg = min_normal_ordered_boson_degree(vword)
    if min_vdeg > PRUNE_MAX_V or min_vdeg % 2 == 1:
        return ()

    out: Dict[Key, sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for nv, c in normal_order_word(vword):
        key = (nv, jword, origin)
        if rot_degree(key) > PRUNE_MAX_J:
            continue
        if boson_degree(key) > PRUNE_MAX_V:
            continue
        if min_normal_ordered_boson_degree(nv) > PRUNE_MAX_V:
            continue
        if min_normal_ordered_boson_degree(nv) % 2 == 1:
            continue
        out[key] += c
    return tuple((k, v) for k, v in out.items() if v != 0)


def mul_expr(a: Dict[Key, sp.Expr], b: Dict[Key, sp.Expr]) -> Dict[Key, sp.Expr]:
    if not a or not b:
        return {}
    out: Dict[Key, sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for (va, ja, oa), ca in a.items():
        for (vb, jb, ob), cb in b.items():
            for key, prefactor in mul_keypair(va, ja, oa, vb, jb, ob):
                out[key] += ca * cb * prefactor
    return _clean(out)


def comm_expr(a: Dict[Key, sp.Expr], b: Dict[Key, sp.Expr]) -> Dict[Key, sp.Expr]:
    return add_expr(mul_expr(a, b), scale_expr(mul_expr(b, a), -1))


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


def comm_with_h0(expr: Dict[Key, sp.Expr], omega: Iterable[sp.Symbol], hbar: sp.Symbol) -> Dict[Key, sp.Expr]:
    out: Dict[Key, sp.Expr] = {}
    for (vword, jword, origin), coeff in expr.items():
        de = delta_energy(vword, hbar, omega)
        if sp.simplify(de) != 0:
            out[(vword, jword, origin)] = coeff * de
    return _clean(prune_expr(out))


# --- Hamiltonian builders ------------------------------------------------------------

def build_hprime(
    n_modes: int, diag_rot_only: bool = False, phi4_reduced: bool = False, rot_pairs: set[tuple[int, int]] | None = None
) -> tuple[
    Dict[Key, sp.Expr],
    Dict[Key, sp.Expr],
    Dict[Key, sp.Expr],
    Dict[Key, sp.Expr],
    Dict[Key, sp.Expr],
    tuple[sp.Symbol, ...],
    sp.Symbol,
]:
    hbar = sp.symbols("hbar", positive=True)
    omega = sp.symbols(f"omega0:{n_modes}", positive=True)

    mu1 = sp.MutableDenseNDimArray(sp.symbols(f"mu1_0:{3*3*n_modes}", real=True), (3, 3, n_modes))
    mu2 = sp.MutableDenseNDimArray(sp.symbols(f"mu2_0:{3*3*n_modes*n_modes}", real=True), (3, 3, n_modes, n_modes))
    phi3 = sp.MutableDenseNDimArray(sp.symbols(f"phi3_0:{n_modes*n_modes*n_modes}", real=True), (n_modes, n_modes, n_modes))
    phi4 = sp.MutableDenseNDimArray(sp.symbols(f"phi4_0:{n_modes*n_modes*n_modes*n_modes}", real=True), (n_modes, n_modes, n_modes, n_modes))

    hrv1: Dict[Key, sp.Expr] = {}
    for a in range(3):
        for b in range(3):
            if rot_pairs is not None and (a, b) not in rot_pairs:
                continue
            if diag_rot_only and a != b:
                continue
            for k in range(n_modes):
                c = sp.Rational(1, 2) * mu1[a, b, k] * sp.sqrt(hbar / (2 * omega[k]))
                jword = (JOPS[a], JOPS[b])
                hrv1 = add_expr(
                    hrv1,
                    one_term(c, (("a", k),), jword, ("H12",)),
                    one_term(c, (("ad", k),), jword, ("H12",)),
                )

    hrv2: Dict[Key, sp.Expr] = {}
    for a in range(3):
        for b in range(3):
            if rot_pairs is not None and (a, b) not in rot_pairs:
                continue
            if diag_rot_only and a != b:
                continue
            for k in range(n_modes):
                for l in range(n_modes):
                    c = (
                        sp.Rational(1, 4)
                        * mu2[a, b, k, l]
                        * sp.sqrt(hbar / (2 * omega[k]))
                        * sp.sqrt(hbar / (2 * omega[l]))
                    )
                    jword = (JOPS[a], JOPS[b])
                    for op1 in ("a", "ad"):
                        for op2 in ("a", "ad"):
                            hrv2 = add_expr(hrv2, one_term(c, ((op1, k), (op2, l)), jword, ("H22",)))

    v3: Dict[Key, sp.Expr] = {}
    for i in range(n_modes):
        for j in range(n_modes):
            for k in range(n_modes):
                c = (
                    sp.Rational(1, 6)
                    * phi3[i, j, k]
                    * sp.sqrt(hbar / (2 * omega[i]))
                    * sp.sqrt(hbar / (2 * omega[j]))
                    * sp.sqrt(hbar / (2 * omega[k]))
                )
                for op_i in ("a", "ad"):
                    for op_j in ("a", "ad"):
                        for op_k in ("a", "ad"):
                            v3 = add_expr(v3, one_term(c, ((op_i, i), (op_j, j), (op_k, k)), (), ("H30",)))

    def phi4_allowed(i: int, j: int, k: int, l: int) -> bool:
        if not phi4_reduced:
            return True
        # Keep only classes: iiii, iiij (3+1), iijj/ijij (2+2).
        counts = sorted([(i, j, k, l).count(x) for x in set((i, j, k, l))], reverse=True)
        return counts in ([4], [3, 1], [2, 2])

    v4: Dict[Key, sp.Expr] = {}
    for i in range(n_modes):
        for j in range(n_modes):
            for k in range(n_modes):
                for l in range(n_modes):
                    if not phi4_allowed(i, j, k, l):
                        continue
                    c = (
                        sp.Rational(1, 24)
                        * phi4[i, j, k, l]
                        * sp.sqrt(hbar / (2 * omega[i]))
                        * sp.sqrt(hbar / (2 * omega[j]))
                        * sp.sqrt(hbar / (2 * omega[k]))
                        * sp.sqrt(hbar / (2 * omega[l]))
                    )
                    for op_i in ("a", "ad"):
                        for op_j in ("a", "ad"):
                            for op_k in ("a", "ad"):
                                for op_l in ("a", "ad"):
                                    v4 = add_expr(
                                        v4,
                                        one_term(c, ((op_i, i), (op_j, j), (op_k, k), (op_l, l)), (), ("H40",)),
                                    )

    hprime = add_expr(hrv1, hrv2, v3, v4)
    hprime = prune_expr(hprime)
    return hprime, hrv1, hrv2, v3, v4, omega, hbar


def build_hprime_collapsed(
    n_modes: int,
    seed: int = 7,
    diag_rot_only: bool = False,
    phi4_reduced: bool = False,
    rot_pairs: set[tuple[int, int]] | None = None,
) -> tuple[
    Dict[Key, sp.Expr],
    Dict[Key, sp.Expr],
    Dict[Key, sp.Expr],
    Dict[Key, sp.Expr],
    Dict[Key, sp.Expr],
    tuple[sp.Symbol, ...],
    sp.Symbol,
    tuple[sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol],
]:
    """Collapsed-coupling variant for faster family-structure probing.

    Couplings are random rational weights times class parameters:
      A -> mu1, B -> mu2, C -> phi3, D -> phi4
    """
    rng = random.Random(seed)
    hbar = sp.symbols("hbar", positive=True)
    omega = tuple(sp.Rational(i + 2, 1) for i in range(n_modes))
    A, B, C, D = sp.symbols("A B C D", real=True)

    def rr():
        return sp.Rational(rng.randint(1, 9), rng.randint(1, 5))

    mu1 = sp.MutableDenseNDimArray([A * rr() for _ in range(3 * 3 * n_modes)], (3, 3, n_modes))
    mu2 = sp.MutableDenseNDimArray([B * rr() for _ in range(3 * 3 * n_modes * n_modes)], (3, 3, n_modes, n_modes))
    phi3 = sp.MutableDenseNDimArray([C * rr() for _ in range(n_modes * n_modes * n_modes)], (n_modes, n_modes, n_modes))
    phi4 = sp.MutableDenseNDimArray([D * rr() for _ in range(n_modes * n_modes * n_modes * n_modes)], (n_modes, n_modes, n_modes, n_modes))

    hrv1: Dict[Key, sp.Expr] = {}
    for a in range(3):
        for b in range(3):
            if rot_pairs is not None and (a, b) not in rot_pairs:
                continue
            if diag_rot_only and a != b:
                continue
            for k in range(n_modes):
                c = sp.Rational(1, 2) * mu1[a, b, k] * sp.sqrt(hbar / (2 * omega[k]))
                jword = (JOPS[a], JOPS[b])
                hrv1 = add_expr(
                    hrv1,
                    one_term(c, (("a", k),), jword, ("H12",)),
                    one_term(c, (("ad", k),), jword, ("H12",)),
                )

    hrv2: Dict[Key, sp.Expr] = {}
    for a in range(3):
        for b in range(3):
            if rot_pairs is not None and (a, b) not in rot_pairs:
                continue
            if diag_rot_only and a != b:
                continue
            for k in range(n_modes):
                for l in range(n_modes):
                    c = (
                        sp.Rational(1, 4)
                        * mu2[a, b, k, l]
                        * sp.sqrt(hbar / (2 * omega[k]))
                        * sp.sqrt(hbar / (2 * omega[l]))
                    )
                    jword = (JOPS[a], JOPS[b])
                    for op1 in ("a", "ad"):
                        for op2 in ("a", "ad"):
                            hrv2 = add_expr(hrv2, one_term(c, ((op1, k), (op2, l)), jword, ("H22",)))

    v3: Dict[Key, sp.Expr] = {}
    for i in range(n_modes):
        for j in range(n_modes):
            for k in range(n_modes):
                c = (
                    sp.Rational(1, 6)
                    * phi3[i, j, k]
                    * sp.sqrt(hbar / (2 * omega[i]))
                    * sp.sqrt(hbar / (2 * omega[j]))
                    * sp.sqrt(hbar / (2 * omega[k]))
                )
                for op_i in ("a", "ad"):
                    for op_j in ("a", "ad"):
                        for op_k in ("a", "ad"):
                            v3 = add_expr(v3, one_term(c, ((op_i, i), (op_j, j), (op_k, k)), (), ("H30",)))

    def phi4_allowed(i: int, j: int, k: int, l: int) -> bool:
        if not phi4_reduced:
            return True
        # Keep only classes: iiii, iiij (3+1), iijj/ijij (2+2).
        counts = sorted([(i, j, k, l).count(x) for x in set((i, j, k, l))], reverse=True)
        return counts in ([4], [3, 1], [2, 2])

    v4: Dict[Key, sp.Expr] = {}
    for i in range(n_modes):
        for j in range(n_modes):
            for k in range(n_modes):
                for l in range(n_modes):
                    if not phi4_allowed(i, j, k, l):
                        continue
                    c = (
                        sp.Rational(1, 24)
                        * phi4[i, j, k, l]
                        * sp.sqrt(hbar / (2 * omega[i]))
                        * sp.sqrt(hbar / (2 * omega[j]))
                        * sp.sqrt(hbar / (2 * omega[k]))
                        * sp.sqrt(hbar / (2 * omega[l]))
                    )
                    for op_i in ("a", "ad"):
                        for op_j in ("a", "ad"):
                            for op_k in ("a", "ad"):
                                for op_l in ("a", "ad"):
                                    v4 = add_expr(
                                        v4,
                                        one_term(c, ((op_i, i), (op_j, j), (op_k, k), (op_l, l)), (), ("H40",)),
                                    )

    hprime = add_expr(hrv1, hrv2, v3, v4)
    hprime = prune_expr(hprime)
    return hprime, hrv1, hrv2, v3, v4, omega, hbar, (A, B, C, D)


# --- Perturbative Lie transform to 4th order ----------------------------------------

def add_series(a: Series, b: Series, max_order: int) -> Series:
    out: Series = {}
    for n in range(max_order + 1):
        out[n] = add_expr(a.get(n, {}), b.get(n, {}))
    return out


def ad_series(
    s_series: Series,
    y_series: Series,
    y_has_h0: bool,
    omega: Iterable[sp.Symbol],
    hbar: sp.Symbol,
    max_order: int,
) -> Series:
    out: Series = {n: {} for n in range(max_order + 1)}

    for i, si in s_series.items():
        for j, yj in y_series.items():
            if i + j <= max_order and si and yj:
                c = comm_expr(si, yj)
                out[i + j] = add_expr(out[i + j], c)

    if y_has_h0:
        for i, si in s_series.items():
            if i <= max_order and si:
                out[i] = add_expr(out[i], comm_with_h0(si, omega, hbar))

    for n in range(max_order + 1):
        out[n] = prune_expr(out[n])
    return out


def bch_transform(
    h_series: Series,
    s_series: Series,
    omega: Iterable[sp.Symbol],
    hbar: sp.Symbol,
    max_order: int,
) -> Series:
    k: Series = {n: h_series.get(n, {}) for n in range(max_order + 1)}

    term = {n: h_series.get(n, {}) for n in range(max_order + 1)}
    has_h0 = True

    for n in range(1, max_order + 1):
        term = ad_series(s_series, term, has_h0, omega, hbar, max_order)
        has_h0 = False
        term = {m: scale_expr(term.get(m, {}), sp.Rational(1, n)) for m in range(max_order + 1)}
        k = add_series(k, term, max_order)

    for n in range(max_order + 1):
        k[n] = prune_expr(normal_order_expr(k[n]))
    return k


def split_diag_offdiag(expr: Dict[Key, sp.Expr], omega: Iterable[sp.Symbol], hbar: sp.Symbol) -> tuple[Dict[Key, sp.Expr], Dict[Key, sp.Expr]]:
    diag: Dict[Key, sp.Expr] = {}
    off: Dict[Key, sp.Expr] = {}
    for (v, j, origin), c in expr.items():
        if CHANNEL_AWARE and not keep_for_generator(v, j):
            continue
        de = delta_energy(v, hbar, omega)
        if sp.simplify(de) == 0:
            diag[(v, j, origin)] = c
        else:
            off[(v, j, origin)] = c
    return _clean(prune_expr(diag)), _clean(prune_expr(off))


def solve_s_order(offdiag: Dict[Key, sp.Expr], omega: Iterable[sp.Symbol], hbar: sp.Symbol) -> Dict[Key, sp.Expr]:
    out: Dict[Key, sp.Expr] = {}
    for (v, j, origin), c in offdiag.items():
        de = delta_energy(v, hbar, omega)
        out[(v, j, origin)] = -c / de
    return _clean(prune_expr(out))


def build_effective_to_order(h_input: Series, omega: Iterable[sp.Symbol], hbar: sp.Symbol, max_order: int = 4) -> tuple[Series, Series]:
    h_series: Series = {0: {}}
    for n in range(1, max_order + 1):
        # Keep full input blocks (especially V3 with vib word length 3);
        # pruning is applied during BCH/ad propagation.
        h_series[n] = _clean(h_input.get(n, {}))

    s_series: Series = {}

    for m in range(1, max_order + 1):
        partial = bch_transform(h_series, s_series, omega, hbar, max_order=m)
        _, off_m = split_diag_offdiag(partial[m], omega, hbar)
        s_series[m] = solve_s_order(off_m, omega, hbar)
        s_series[m] = prune_expr(s_series[m])

    k_full = bch_transform(h_series, s_series, omega, hbar, max_order=max_order)
    return k_full, s_series


# --- Projection to tau / Watson constants --------------------------------------------

def extract_quartic_rot_ground(expr: Dict[Key, sp.Expr]) -> Dict[JWord, sp.Expr]:
    out: Dict[JWord, sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for (vword, jword, _origin), coeff in normal_order_expr(expr).items():
        if len(vword) == 0 and len(jword) == 4:
            out[jword] += coeff
    return {k: sp.simplify(v) for k, v in out.items() if sp.simplify(v) != 0}


def extract_quartic_rot_ground_by_origin(expr: Dict[Key, sp.Expr]) -> Dict[Origin, Dict[JWord, sp.Expr]]:
    out: Dict[Origin, Dict[JWord, sp.Expr]] = defaultdict(lambda: defaultdict(lambda: sp.Integer(0)))
    for (vword, jword, origin), coeff in normal_order_expr(expr).items():
        if len(vword) == 0 and len(jword) == 4:
            out[normalize_origin(origin)][jword] += coeff
    cleaned: Dict[Origin, Dict[JWord, sp.Expr]] = {}
    for origin, terms in out.items():
        reduced = {jw: sp.simplify(c) for jw, c in terms.items() if sp.simplify(c) != 0}
        if reduced:
            cleaned[origin] = reduced
    return cleaned


def commuting_projection(quartic_ops: Dict[JWord, sp.Expr]) -> Dict[Tuple[int, int, int], sp.Expr]:
    x, y, z = sp.symbols("x y z", real=True)
    repl = {Jx: x, Jy: y, Jz: z}

    poly = sp.Integer(0)
    for jword, coeff in quartic_ops.items():
        mon = sp.Integer(1)
        for j in jword:
            mon *= repl[j]
        poly += coeff * mon

    p = sp.Poly(sp.expand(poly), x, y, z)
    return {
        (4, 0, 0): sp.simplify(p.coeff_monomial(x**4)),
        (0, 4, 0): sp.simplify(p.coeff_monomial(y**4)),
        (0, 0, 4): sp.simplify(p.coeff_monomial(z**4)),
        (2, 2, 0): sp.simplify(p.coeff_monomial(x**2 * y**2)),
        (2, 0, 2): sp.simplify(p.coeff_monomial(x**2 * z**2)),
        (0, 2, 2): sp.simplify(p.coeff_monomial(y**2 * z**2)),
    }


def wilson_tau4_from_quartic_ops(
    quartic_ops: Dict[JWord, sp.Expr],
) -> Dict[Tuple[int, int, int, int], sp.Expr]:
    """Return the full four-index quartic tensor before six-component compression.

    The tensor is built directly from the ordered quartic rotational monomials
    extracted from the effective Hamiltonian. This is the correct starting point
    for a Gaussian-consistent projection, which proceeds through Wilson's Tau,
    then Tau', then T = Tau'/4.
    """
    idx = {Jx: 0, Jy: 1, Jz: 2}
    raw: Dict[Tuple[int, int, int, int], sp.Expr] = {
        (i, j, k, l): sp.Integer(0)
        for i in range(3)
        for j in range(3)
        for k in range(3)
        for l in range(3)
    }
    for jword, coeff in quartic_ops.items():
        key = tuple(idx[j] for j in jword)
        raw[key] = sp.simplify(raw[key] + coeff)

    out: Dict[Tuple[int, int, int, int], sp.Expr] = {}
    for key in raw:
        perms = set(itertools.permutations(key))
        out[key] = sp.simplify(sum(raw[p] for p in perms) / len(perms))
    return out


def tau_constants_from_poly(c: Dict[Tuple[int, int, int], sp.Expr]) -> Dict[str, sp.Expr]:
    return {
        "tau_xxxx": sp.simplify(c[(4, 0, 0)]),
        "tau_yyyy": sp.simplify(c[(0, 4, 0)]),
        "tau_zzzz": sp.simplify(c[(0, 0, 4)]),
        "tau_xxyy": sp.simplify(c[(2, 2, 0)]),
        "tau_xxzz": sp.simplify(c[(2, 0, 2)]),
        "tau_yyzz": sp.simplify(c[(0, 2, 2)]),
    }


def watson_a_constants_from_poly(c: Dict[Tuple[int, int, int], sp.Expr]) -> Dict[str, sp.Expr]:
    """Provisional direct map from compressed quartic polynomial to Watson A.

    Kept for backwards compatibility with the earlier compressed-six-component
    workflow. Gaussian's ``qcent`` conventions instead project through the full
    Wilson quartic tensor, then Tau', then T = Tau'/4.
    """
    c_x4 = c[(4, 0, 0)]
    c_y4 = c[(0, 4, 0)]
    c_z4 = c[(0, 0, 4)]
    c_xy = c[(2, 2, 0)]
    c_xz = c[(2, 0, 2)]
    c_yz = c[(0, 2, 2)]

    DJ = sp.simplify(sp.Rational(1, 8) * (c_xy + c_xz + c_yz))
    DJK = sp.simplify(sp.Rational(1, 8) * (-2 * c_xy + c_xz + c_yz))
    DK = sp.simplify(sp.Rational(1, 8) * (c_x4 + c_y4 + c_z4 - 2 * c_xz - 2 * c_yz))
    d1 = sp.simplify(sp.Rational(1, 8) * (c_xz - c_yz))
    d2 = sp.simplify(sp.Rational(1, 16) * (c_x4 - c_y4))

    return {"DJ": DJ, "DJK": DJK, "DK": DK, "d1": d1, "d2": d2}


def gaussian_tauprime_from_wilson_tau(
    tau4: Dict[Tuple[int, int, int, int], sp.Expr],
) -> Dict[Tuple[int, int], sp.Expr]:
    out: Dict[Tuple[int, int], sp.Expr] = {}
    for i in range(3):
        for j in range(3):
            if i == j:
                out[(i, j)] = sp.simplify(tau4[(i, i, i, i)])
            else:
                out[(i, j)] = sp.simplify(tau4[(i, i, j, j)] + 2 * tau4[(i, j, i, j)])
    return out


def gaussian_t_from_tauprime(
    tau_prime: Dict[Tuple[int, int], sp.Expr],
) -> Dict[Tuple[int, int], sp.Expr]:
    return {(i, j): sp.simplify(tau_prime[(i, j)] / 4) for i in range(3) for j in range(3)}


def gaussian_cylindrical_from_t(
    tmat: Dict[Tuple[int, int], sp.Expr],
) -> Dict[str, sp.Expr]:
    t11 = tmat[(0, 0)]
    t22 = tmat[(1, 1)]
    t33 = tmat[(2, 2)]
    t12 = tmat[(0, 1)]
    t13 = tmat[(0, 2)]
    t23 = tmat[(1, 2)]
    t400 = sp.simplify((3 * t11 + 3 * t22 + 2 * t12) / 8)
    t220 = sp.simplify(t13 + t23 - 2 * t400)
    t040 = sp.simplify(t33 - t220 - t400)
    t202 = sp.simplify((t11 - t22) / 4)
    t022 = sp.simplify((t13 - t23) / 2 - t202)
    t004 = sp.simplify((t11 + t22 - 2 * t12) / 16)
    return {"T400": t400, "T220": t220, "T040": t040, "T202": t202, "T022": t022, "T004": t004}


def gaussian_asymmetric_a_from_t(
    tmat: Dict[Tuple[int, int], sp.Expr],
    sigma: sp.Expr,
) -> Dict[str, sp.Expr]:
    cyl = gaussian_cylindrical_from_t(tmat)
    t400 = cyl["T400"]
    t220 = cyl["T220"]
    t040 = cyl["T040"]
    t202 = cyl["T202"]
    t022 = cyl["T022"]
    t004 = cyl["T004"]
    return {
        "DJ": sp.simplify(-t400 - 2 * t004),
        "DK": sp.simplify(-t040 - 10 * t004),
        "DJK": sp.simplify(-t220 + 12 * t004),
        "dJ": sp.simplify(-t202),
        "dK": sp.simplify(-t022 - 4 * sigma * t004),
    }


def gaussian_symmetric_from_t(
    tmat: Dict[Tuple[int, int], sp.Expr],
    sigma1: sp.Expr,
) -> Dict[str, sp.Expr]:
    cyl = gaussian_cylindrical_from_t(tmat)
    t400 = cyl["T400"]
    t220 = cyl["T220"]
    t040 = cyl["T040"]
    t202 = cyl["T202"]
    t022 = cyl["T022"]
    t004 = cyl["T004"]
    return {
        "DJ": sp.simplify(-t400 + sp.Rational(1, 2) * t022 * sigma1),
        "DK": sp.simplify(-t040 + sp.Rational(5, 2) * t022 * sigma1),
        "DJK": sp.simplify(-t220 - 3 * t022 * sigma1),
        "d1": sp.simplify(t202),
        "d2": sp.simplify(t004 + t022 * sigma1 / 4),
    }


def watson_s_constants_from_poly(c: Dict[Tuple[int, int, int], sp.Expr]) -> Dict[str, sp.Expr]:
    c_x4 = c[(4, 0, 0)]
    c_y4 = c[(0, 4, 0)]
    c_z4 = c[(0, 0, 4)]
    c_xy = c[(2, 2, 0)]
    c_xz = c[(2, 0, 2)]
    c_yz = c[(0, 2, 2)]

    DJ = sp.simplify(sp.Rational(1, 8) * (c_xy + c_xz + c_yz))
    DJK = sp.simplify(sp.Rational(1, 8) * (-c_xy + c_xz + c_yz))
    DK = sp.simplify(sp.Rational(1, 8) * (c_x4 + c_y4 + c_z4 - c_xz - c_yz))
    d1 = sp.simplify(sp.Rational(1, 16) * (c_xz - c_yz))
    d2 = sp.simplify(sp.Rational(1, 16) * (c_x4 - c_y4))

    return {"DJ": DJ, "DJK": DJK, "DK": DK, "d1": d1, "d2": d2}


def classify_term_by_symbols(term: sp.Expr) -> str:
    names = {sym.name for sym in term.free_symbols}
    has_mu1 = any(name.startswith("mu1_") for name in names) or "A" in names
    has_mu2 = any(name.startswith("mu2_") for name in names) or "B" in names
    has_phi3 = any(name.startswith("phi3_") for name in names) or "C" in names
    has_phi4 = any(name.startswith("phi4_") for name in names) or "D" in names
    if has_phi4:
        return "H40"
    if has_phi3 and has_mu2:
        return "H12,H30"
    if has_phi3:
        return "H30,H30"
    if has_mu2:
        return "H22"
    if has_mu1:
        return "H12,H12"
    return "other"


def decompose_expr_by_symbol_class(expr: sp.Expr) -> Dict[str, sp.Expr]:
    pieces: Dict[str, sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for term in sp.Add.make_args(sp.expand(expr)):
        pieces[classify_term_by_symbols(term)] += term
    return {label: sp.simplify(val) for label, val in pieces.items() if sp.simplify(val) != 0}


def decompose_tau_by_symbol_class(tau: Dict[str, sp.Expr]) -> Dict[str, Dict[str, sp.Expr]]:
    classes: Dict[str, Dict[str, sp.Expr]] = defaultdict(dict)
    for tau_name, expr in tau.items():
        for label, piece in decompose_expr_by_symbol_class(expr).items():
            classes[label][tau_name] = piece
    return classes


def build_targeted_input(symbol_class_only: str, hrv1: Dict[Key, sp.Expr], hrv2: Dict[Key, sp.Expr], v3: Dict[Key, sp.Expr], v4: Dict[Key, sp.Expr]) -> Series:
    """Return the minimal perturbative input needed for a requested symbolic class."""
    if symbol_class_only == "H30,H30":
        # The final H30,H30 symbolic family still carries rotational structure
        # generated through H12-mediated chains, even though its symbol-class
        # decomposition contains no explicit mu2/phi4 contribution.
        return {1: add_expr(hrv1, v3), 2: {}}
    if symbol_class_only == "H12,H30":
        return {1: add_expr(hrv1, v3), 2: hrv2}
    if symbol_class_only == "H12,H12":
        return {1: hrv1, 2: {}}
    if symbol_class_only == "H22":
        return {1: {}, 2: hrv2}
    if symbol_class_only == "H40":
        return {1: {}, 2: v4}
    return {
        1: add_expr(hrv1, v3),
        2: add_expr(hrv2, v4),
    }


def main() -> None:
    global PRUNE_MAX_J, PRUNE_MAX_V, CHANNEL_AWARE, TARGET_SYMBOL_CLASS
    ap = argparse.ArgumentParser(description="Van Vleck quartic Watson derivation up to chosen perturbative order.")
    ap.add_argument("--max-order", type=int, default=4, choices=(2, 3, 4), help="Perturbative order.")
    ap.add_argument("--n-modes", type=int, default=1, help="Number of vibrational modes in explicit symbolic expansion.")
    ap.add_argument("--collapsed-couplings", action="store_true", help="Use random numeric couplings times class symbols A,B,C,D.")
    ap.add_argument("--seed", type=int, default=7, help="Random seed for collapsed-coupling probe.")
    ap.add_argument("--max-vib-word", type=int, default=4, help="Pruning cap for vibrational word length.")
    ap.add_argument("--max-j-word", type=int, default=4, help="Pruning cap for rotational word length.")
    ap.add_argument("--channel-aware", action="store_true", help="Filter generator channels to low-J off-diagonal sectors.")
    ap.add_argument("--diag-rot-only", action="store_true", help="Keep only diagonal rotational couplings mu_{aa,*}.")
    ap.add_argument(
        "--phi4-reduced",
        action="store_true",
        help="Keep only Phi4 classes iiii, iiij, and 2+2 (iijj/ijij permutations).",
    )
    ap.add_argument(
        "--rot-pairs",
        type=str,
        default="",
        help="Comma-separated rotational index pairs among xx,xy,xz,yx,yy,yz,zx,zy,zz.",
    )
    ap.add_argument(
        "--symbol-class-only",
        type=str,
        default="",
        help="If set, print only the fourth-order tau block for the given symbolic class (e.g. H30,H30) and exit.",
    )
    ap.add_argument(
        "--show-gaussian-projection",
        action="store_true",
        help="Print Wilson Tau(ijkl), Tau', T=Tau'/4, and Gaussian-style Watson projections.",
    )
    args = ap.parse_args()
    PRUNE_MAX_V = args.max_vib_word
    PRUNE_MAX_J = args.max_j_word
    CHANNEL_AWARE = args.channel_aware
    TARGET_SYMBOL_CLASS = args.symbol_class_only.strip()

    rot_pairs = None
    if args.rot_pairs.strip():
        code = {"x": 0, "y": 1, "z": 2}
        rot_pairs = set()
        for tok in args.rot_pairs.split(","):
            t = tok.strip().lower()
            if len(t) != 2 or t[0] not in code or t[1] not in code:
                raise ValueError(f"Invalid rot pair token: {tok}")
            rot_pairs.add((code[t[0]], code[t[1]]))

    n_modes = args.n_modes

    class_syms = None
    if args.collapsed_couplings:
        _, hrv1, hrv2, v3, v4, omega, hbar, class_syms = build_hprime_collapsed(
            n_modes=n_modes,
            seed=args.seed,
            diag_rot_only=args.diag_rot_only,
            phi4_reduced=args.phi4_reduced,
            rot_pairs=rot_pairs,
        )
    else:
        _, hrv1, hrv2, v3, v4, omega, hbar = build_hprime(
            n_modes=n_modes,
            diag_rot_only=args.diag_rot_only,
            phi4_reduced=args.phi4_reduced,
            rot_pairs=rot_pairs,
        )

    # Standard perturbative assignment:
    # order-1: Hrv1 + V3
    # order-2: Hrv2 + V4
    # When a single symbolic class is requested, keep only the minimal input
    # blocks that can feed that class.
    h_input: Series = build_targeted_input(TARGET_SYMBOL_CLASS, hrv1, hrv2, v3, v4)

    k_series, s_series = build_effective_to_order(h_input, omega=omega, hbar=hbar, max_order=args.max_order)

    quartic_by_order: Dict[int, Dict[JWord, sp.Expr]] = {}
    quartic_by_order_origin: Dict[int, Dict[Origin, Dict[JWord, sp.Expr]]] = {}
    for order in range(1, args.max_order + 1):
        quartic_by_order[order] = extract_quartic_rot_ground(k_series[order])
        quartic_by_order_origin[order] = extract_quartic_rot_ground_by_origin(k_series[order])

    total_quartic: Dict[JWord, sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for order in range(1, args.max_order + 1):
        for jw, c in quartic_by_order[order].items():
            total_quartic[jw] += c
    total_quartic = {k: sp.simplify(v) for k, v in total_quartic.items() if v != 0}

    tau4 = wilson_tau4_from_quartic_ops(total_quartic)
    tau_prime = gaussian_tauprime_from_wilson_tau(tau4)
    t_cart = gaussian_t_from_tauprime(tau_prime)
    proj = commuting_projection(total_quartic)
    tau = tau_constants_from_poly(proj)

    if args.symbol_class_only:
        if args.max_order < 4:
            raise ValueError("--symbol-class-only requires --max-order 4")
        proj_order4 = commuting_projection(quartic_by_order[4]) if quartic_by_order[4] else {
            (4, 0, 0): sp.Integer(0),
            (0, 4, 0): sp.Integer(0),
            (0, 0, 4): sp.Integer(0),
            (2, 2, 0): sp.Integer(0),
            (2, 0, 2): sp.Integer(0),
            (0, 2, 2): sp.Integer(0),
        }
        tau_order4 = tau_constants_from_poly(proj_order4)
        symbol_classes = decompose_tau_by_symbol_class(tau_order4)
        block = symbol_classes.get(args.symbol_class_only, {})
        for name in ("tau_xxxx", "tau_yyyy", "tau_zzzz", "tau_xxyy", "tau_xxzz", "tau_yyzz"):
            print(f"{name} = {sp.simplify(block.get(name, sp.Integer(0)))}")
        return

    print("=== Van Vleck effective Hamiltonian to 4th order ===")
    for n in range(1, args.max_order + 1):
        print(f"order {n}: terms={len(k_series[n])}, quartic-ground={len(quartic_by_order[n])}")

    print("\n=== Quartic tau tensor coefficients (primary output) ===")
    for name in ("tau_xxxx", "tau_yyyy", "tau_zzzz", "tau_xxyy", "tau_xxzz", "tau_yyzz"):
        print(f"{name} = {sp.simplify(tau[name])}")

    if args.show_gaussian_projection:
        print("\n=== Wilson quartic tensor Tau(ijkl) ===")
        for key in sorted(tau4):
            if sp.simplify(tau4[key]) != 0:
                i, j, k, l = key
                print(f"Tau({i+1},{j+1},{k+1},{l+1}) = {sp.simplify(tau4[key])}")

        print("\n=== Gaussian Tau' matrix ===")
        for i in range(3):
            for j in range(3):
                print(f"TauPrime({i+1},{j+1}) = {sp.simplify(tau_prime[(i,j)])}")

        print("\n=== Gaussian T matrix (Tau'/4) ===")
        for i in range(3):
            for j in range(3):
                print(f"T({i+1},{j+1}) = {sp.simplify(t_cart[(i,j)])}")

        print("\n=== Gaussian cylindrical quartic constants ===")
        cyl = gaussian_cylindrical_from_t(t_cart)
        for name in ("T400", "T220", "T040", "T202", "T022", "T004"):
            print(f"{name} = {sp.simplify(cyl[name])}")

        print("\n=== Gaussian symmetric-top reduction (parameterized by Sigma1) ===")
        sigma1 = sp.symbols("Sigma1", real=True)
        gsym = gaussian_symmetric_from_t(t_cart, sigma1)
        for name in ("DJ", "DK", "DJK", "d1", "d2"):
            print(f"{name} = {sp.simplify(gsym[name])}")

        print("\n=== Gaussian asymmetric-top A reduction (parameterized by Sigma) ===")
        sigma = sp.symbols("Sigma", real=True)
        gasym = gaussian_asymmetric_a_from_t(t_cart, sigma)
        for name in ("DJ", "DK", "DJK", "dJ", "dK"):
            print(f"{name} = {sp.simplify(gasym[name])}")

    print("\n=== Quartic tau tensor by perturbative order ===")
    for order in range(1, args.max_order + 1):
        proj_order = commuting_projection(quartic_by_order[order]) if quartic_by_order[order] else {
            (4, 0, 0): sp.Integer(0),
            (0, 4, 0): sp.Integer(0),
            (0, 0, 4): sp.Integer(0),
            (2, 2, 0): sp.Integer(0),
            (2, 0, 2): sp.Integer(0),
            (0, 2, 2): sp.Integer(0),
        }
        tau_order = tau_constants_from_poly(proj_order)
        print(f"\norder {order}:")
        for name in ("tau_xxxx", "tau_yyyy", "tau_zzzz", "tau_xxyy", "tau_xxzz", "tau_yyzz"):
            print(f"  {name} = {sp.simplify(tau_order[name])}")

    if args.max_order >= 4:
        print("\n=== Fourth-order tau contributions by Hmn class ===")
        preferred_order = [
            ("H22",),
            ("H40",),
            ("H30", "H30"),
            ("H12", "H30"),
            ("H12", "H12"),
        ]
        seen: set[Origin] = set()
        for origin in preferred_order + sorted(quartic_by_order_origin[4].keys(), key=str):
            if origin in seen or origin not in quartic_by_order_origin[4]:
                continue
            seen.add(origin)
            proj_origin = commuting_projection(quartic_by_order_origin[4][origin])
            tau_origin = tau_constants_from_poly(proj_origin)
            print(f"\norigin {origin}:")
            for name in ("tau_xxxx", "tau_yyyy", "tau_zzzz", "tau_xxyy", "tau_xxzz", "tau_yyzz"):
                print(f"  {name} = {sp.simplify(tau_origin[name])}")

        proj_order4 = commuting_projection(quartic_by_order[4]) if quartic_by_order[4] else {
            (4, 0, 0): sp.Integer(0),
            (0, 4, 0): sp.Integer(0),
            (0, 0, 4): sp.Integer(0),
            (2, 2, 0): sp.Integer(0),
            (2, 0, 2): sp.Integer(0),
            (0, 2, 2): sp.Integer(0),
        }
        tau_order4 = tau_constants_from_poly(proj_order4)
        symbol_classes = decompose_tau_by_symbol_class(tau_order4)
        print("\n=== Fourth-order tau contributions by symbolic class ===")
        for label in ("H22", "H40", "H30,H30", "H12,H30", "H12,H12", "other"):
            if label not in symbol_classes:
                continue
            print(f"\nclass {label}:")
            for name in ("tau_xxxx", "tau_yyyy", "tau_zzzz", "tau_xxyy", "tau_xxzz", "tau_yyzz"):
                if name in symbol_classes[label]:
                    print(f"  {name} = {sp.simplify(symbol_classes[label][name])}")

    print("\n=== Recommended final projection ===")
    print("Use --show-gaussian-projection for Gaussian-consistent Tau -> Tau' -> T -> Watson constants.")

    all_tau = sp.simplify(sum(tau.values(), sp.Integer(0)))
    print("\n=== Symbol presence in final tau coefficients ===")
    for lbl in ("mu1_", "mu2_", "phi3_", "phi4_"):
        print(f"{lbl[:-1]} contributes: {lbl in str(all_tau)}")

    print("\n=== Example quartic operator coefficients (first 10) ===")
    for i, (jw, c) in enumerate(sorted(total_quartic.items(), key=lambda kv: str(kv[0]))):
        if i >= 10:
            break
        print(jw, ":", sp.simplify(c))

    if class_syms is not None:
        A, B, C, D = class_syms
        print("\n=== Family polynomial decomposition (A=mu1, B=mu2, C=phi3, D=phi4) ===")
        for name in ("tau_xxxx", "tau_yyyy", "tau_zzzz", "tau_xxyy", "tau_xxzz", "tau_yyzz"):
            expr = sp.expand(tau[name])
            P = sp.Poly(expr, A, B, C, D)
            print(f"\n{name}:")
            for powers, coeff in sorted(P.terms(), key=lambda x: x[0]):
                if coeff != 0:
                    print(f"  A^{powers[0]} B^{powers[1]} C^{powers[2]} D^{powers[3]} : {sp.simplify(coeff)}")


if __name__ == "__main__":
    main()
