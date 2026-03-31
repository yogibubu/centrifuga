#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "scripts"))

from paper1_h08_term3_onemode_scaling_audit import paper1_h08_term3_onemode_scaling_audit


def test_one_mode_term3_scaling_is_not_geometric() -> None:
    out = paper1_h08_term3_onemode_scaling_audit()
    assert "C**7" in str(out["X4_no_cubic"])
    assert "omega**3" in str(out["X4_no_cubic"])

