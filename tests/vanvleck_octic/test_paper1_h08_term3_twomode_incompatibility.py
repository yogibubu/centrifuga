#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "scripts"))

from paper1_h08_term3_twomode_incompatibility import paper1_h08_term3_twomode_incompatibility


def test_twomode_probe_ratio_differs_from_known_branch() -> None:
    out = paper1_h08_term3_twomode_incompatibility()
    assert out["difference_no_cubic"] != 0

