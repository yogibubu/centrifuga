#!/usr/bin/env python3

from tools.vanvleck.extract_vanvleck_generators import extract_reference_generators


def test_extract_reference_generators_exposes_first_three_bch_orders():
    out = extract_reference_generators(n_modes=1, max_order=3, max_j=6, max_v=6)
    gens = out["generators"]
    assert "S^(1)" in gens
    assert "S^(2)" in gens
    assert "S^(3)" in gens
    assert gens["S^(1)"]["watson_label"] == "S03"
    assert gens["S^(2)"]["watson_label"] == "S05"
    assert gens["S^(3)"]["watson_label"] == "S07"


def test_extract_reference_generators_has_nonempty_first_generator():
    out = extract_reference_generators(n_modes=1, max_order=2, max_j=4, max_v=4)
    s1 = out["generators"]["S^(1)"]["summary"]
    assert s1["term_count"] > 0
