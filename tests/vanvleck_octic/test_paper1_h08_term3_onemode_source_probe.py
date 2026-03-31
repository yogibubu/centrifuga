#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "scripts"))

from paper1_h08_term3_onemode_source_probe import paper1_h08_term3_onemode_source_probe


def test_term3_source_probe_has_nonzero_x5() -> None:
    out = paper1_h08_term3_onemode_source_probe()
    assert out["X5"] != 0

