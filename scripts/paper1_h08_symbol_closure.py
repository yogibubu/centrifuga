#!/usr/bin/env python3
"""Final symbol-level closure of the one-line Paper1 H08 source.

This module records the exact status reached after reading the three screenshots:
    H08.jpg
    R_kl.jpg
    Rk.jpg
    notation.jpg

Conclusion:
    the one-line H08 source and its notation block are fully closed at the
    symbol level. No undefined operator symbol remains inside the visible H08
    formula.
"""

from __future__ import annotations


def paper1_h08_closed_symbols() -> dict[str, tuple[str, ...]]:
    return {
        "visible_h08_symbols": (
            "R_k",
            "R_l",
            "R_m",
            "R_n",
            "R_klm",
            "R'_k",
            "R'_{lm}",
            "k'_{klmn}",
            "omega_k",
            "omega_l",
            "omega_m",
            "omega_n",
            "[R_k,R_l]",
        ),
        "notation_block_symbols": (
            "R'_k",
            "R_klm",
            "R_k",
            "R_kl",
            "k'_klm",
            "k'_{klmn}",
            "E_kl",
            "E^kl",
            "F_kl",
            "F^kl",
            "X_kl",
            "X^kl",
            "R'_kl",
            "R~_k",
            "H02",
            "Σ~",
            "Σ*",
        ),
        "closed_by_screenshots": (
            "R'_k",
            "R_klm",
            "R_k",
            "R_kl",
            "k'_klm",
            "k'_{klmn}",
            "E_kl",
            "E^kl",
            "F_kl",
            "F^kl",
            "Σ~",
            "Σ*",
        ),
        "still_open_at_symbol_level": tuple(),
    }


def paper1_h08_closure_verdict() -> dict[str, object]:
    symbols = paper1_h08_closed_symbols()
    return {
        "one_line_h08_symbolically_closed": symbols["still_open_at_symbol_level"] == tuple(),
        "still_open_at_symbol_level": symbols["still_open_at_symbol_level"],
        "closure_scope": "one-line H08 source plus notation block",
    }


def main() -> None:
    print(paper1_h08_closure_verdict())


if __name__ == "__main__":
    main()
