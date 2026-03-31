#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "scripts"))

from paper1_h08_term1_onemode_source_probe import paper1_h08_term1_onemode_source_probe


def test_term1_source_probe_vanishes() -> None:
    out = paper1_h08_term1_onemode_source_probe()
    assert all(val == 0 for val in out["k0_values"].values())

