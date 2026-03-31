#!/usr/bin/env python3
"""Exact so(3) audit of the J_y^8 coefficient for octic commutator channels."""

from __future__ import annotations

from functools import lru_cache

import sympy as sp

X = "x"
Y = "y"
Z = "z"
I = sp.I

Word = tuple[str, ...]


def _comm_pair(a: str, b: str) -> dict[Word, sp.Expr]:
    if (a, b) == (X, Y):
        return {(Z,): I}
    if (a, b) == (Y, X):
        return {(Z,): -I}
    if (a, b) == (Y, Z):
        return {(X,): I}
    if (a, b) == (Z, Y):
        return {(X,): -I}
    if (a, b) == (Z, X):
        return {(Y,): I}
    if (a, b) == (X, Z):
        return {(Y,): -I}
    return {}


def _word_add(out: dict[Word, sp.Expr], word: Word, coeff: sp.Expr) -> None:
    out[word] = sp.expand(out.get(word, sp.Integer(0)) + coeff)
    if out[word] == 0:
        del out[word]


@lru_cache(maxsize=None)
def normal_order_word(word: Word) -> tuple[tuple[Word, sp.Expr], ...]:
    order = {X: 0, Y: 1, Z: 2}
    for i in range(len(word) - 1):
        a, b = word[i], word[i + 1]
        if order[a] > order[b]:
            out: dict[Word, sp.Expr] = {}
            swapped = word[:i] + (b, a) + word[i + 2 :]
            for w, c in normal_order_word(swapped):
                _word_add(out, w, c)
            for cw, cc in _comm_pair(a, b).items():
                reduced = word[:i] + cw + word[i + 2 :]
                for w, c in normal_order_word(reduced):
                    _word_add(out, w, sp.expand(cc * c))
            return tuple(out.items())
    return ((word, sp.Integer(1)),)


def mul_poly(a: dict[Word, sp.Expr], b: dict[Word, sp.Expr]) -> dict[Word, sp.Expr]:
    out: dict[Word, sp.Expr] = {}
    for wa, ca in a.items():
        for wb, cb in b.items():
            for w, c in normal_order_word(wa + wb):
                _word_add(out, w, sp.expand(ca * cb * c))
    return out


def sub_poly(a: dict[Word, sp.Expr], b: dict[Word, sp.Expr]) -> dict[Word, sp.Expr]:
    out = dict(a)
    for w, c in b.items():
        _word_add(out, w, -c)
    return out


def comm_poly(a: dict[Word, sp.Expr], b: dict[Word, sp.Expr]) -> dict[Word, sp.Expr]:
    return sub_poly(mul_poly(a, b), mul_poly(b, a))


def _mon(word: Word, coeff: sp.Expr = sp.Integer(1)) -> dict[Word, sp.Expr]:
    return {word: coeff}


def _add_polys(*polys: dict[Word, sp.Expr]) -> dict[Word, sp.Expr]:
    out: dict[Word, sp.Expr] = {}
    for poly in polys:
        for w, c in poly.items():
            _word_add(out, w, c)
    return out


def exact_c080_audit() -> dict[str, object]:
    A, B, C = sp.symbols("A B C")
    S111 = sp.Symbol("S111")
    s311, s131, s113 = sp.symbols("s311 s131 s113")
    q400, q040, q004, q220, q202, q022 = sp.symbols("q400 q040 q004 q220 q202 q022")
    w600, w060, w006, w420, w402, w240, w204, w042, w024, w222 = sp.symbols(
        "w600 w060 w006 w420 w402 w240 w204 w042 w024 w222"
    )

    sigma3 = -sp.Rational(3, 2) * S111

    S03 = _mon((X, Y, Z), sigma3)
    H02 = _add_polys(_mon((X, X), A), _mon((Y, Y), B), _mon((Z, Z), C))
    H04 = _add_polys(
        _mon((X, X, X, X), q400),
        _mon((Y, Y, Y, Y), q040),
        _mon((Z, Z, Z, Z), q004),
        _mon((X, X, Y, Y), q220),
        _mon((X, X, Z, Z), q202),
        _mon((Y, Y, Z, Z), q022),
    )
    H06 = _add_polys(
        _mon((X, X, X, X, X, X), w600),
        _mon((Y, Y, Y, Y, Y, Y), w060),
        _mon((Z, Z, Z, Z, Z, Z), w006),
        _mon((X, X, X, X, Y, Y), w420),
        _mon((X, X, X, X, Z, Z), w402),
        _mon((X, X, Y, Y, Y, Y), w240),
        _mon((X, X, Z, Z, Z, Z), w204),
        _mon((Y, Y, Y, Y, Z, Z), w042),
        _mon((Y, Y, Z, Z, Z, Z), w024),
        _mon((X, X, Y, Y, Z, Z), w222),
    )
    S05 = _add_polys(
        _mon((X, X, X, Y, Z), s311),
        _mon((X, Y, Y, Y, Z), s131),
        _mon((X, Y, Z, Z, Z), s113),
    )

    channels = {
        "S03S03S03H02": comm_poly(S03, comm_poly(S03, comm_poly(S03, H02))),
        "S03S03H04": comm_poly(S03, comm_poly(S03, H04)),
        "S03H06": comm_poly(S03, H06),
        "S05H04": comm_poly(S05, H04),
        "S05S03H02": comm_poly(S05, comm_poly(S03, H02)),
    }

    y8 = (Y,) * 8
    c080 = {name: sp.expand(poly.get(y8, sp.Integer(0))) for name, poly in channels.items()}
    return {"c080_exact": c080}


def main() -> None:
    out = exact_c080_audit()
    for key, val in out["c080_exact"].items():
        print(key, "->", val)


if __name__ == "__main__":
    main()
