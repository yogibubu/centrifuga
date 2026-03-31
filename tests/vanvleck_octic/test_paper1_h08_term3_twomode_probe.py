#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "scripts"))

from paper1_h08_term3_twomode_probe import paper1_h08_term3_twomode_probe


def test_term3_twomode_probe_has_nonzero_x4_and_x5() -> None:
    out = paper1_h08_term3_twomode_probe()
    assert out["X4"] != 0
    assert out["X5"] != 0

