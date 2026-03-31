#!/usr/bin/env python3
from __future__ import annotations

from scripts.paper1_h08_symbol_closure import (
    paper1_h08_closed_symbols,
    paper1_h08_closure_verdict,
)


def test_h08_symbol_level_is_closed() -> None:
    out = paper1_h08_closure_verdict()
    assert out["one_line_h08_symbolically_closed"] is True
    assert out["still_open_at_symbol_level"] == tuple()


def test_expected_symbols_are_marked_closed() -> None:
    out = paper1_h08_closed_symbols()
    for key in ("R'_k", "R_klm", "R_k", "R_kl", "k'_klm", "k'_{klmn}", "E_kl", "F_kl"):
        assert key in out["closed_by_screenshots"]
