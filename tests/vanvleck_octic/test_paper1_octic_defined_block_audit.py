#!/usr/bin/env python3
from __future__ import annotations

from scripts.paper1_octic_defined_block_audit import (
    paper1_octic_defined_block_verdict,
    paper1_octic_defined_support_map,
)


def test_defined_h08_blocks_are_mu2_free() -> None:
    verdict = paper1_octic_defined_block_verdict()
    assert verdict["mu2_in_defined_H08_blocks"] is False
    assert all(flag is False for flag in verdict["mu2_in_each_block"].values())


def test_defined_h08_blocks_have_expected_support() -> None:
    verdict = paper1_octic_defined_block_verdict()
    assert verdict["rotational_support_defined_H08"] == ("mu1",)
    assert verdict["potential_support_defined_H08"] == ("phi3", "phi4")
    assert verdict["harmonic_support_defined_H08"] == ("B", "omega", "zeta")


def test_u_v_remain_in_defined_printed_algebra() -> None:
    dep = paper1_octic_defined_support_map()
    for key in ("U", "V"):
        assert dep[key].rot == frozenset({"mu1"})
        assert dep[key].pot == frozenset({"phi4"})
        assert dep[key].harm == frozenset({"B", "omega", "zeta"})
