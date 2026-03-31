#!/usr/bin/env python3

from tools.vanvleck.map_vanvleck_generators_to_watson import map_generators_to_watson


def test_map_generators_to_watson_has_first_three_bridges():
    out = map_generators_to_watson(n_modes=1, max_order=3, max_j=6, max_v=6)
    mapping = out["mapping"]
    assert mapping["S^(1)"]["watson_label"] == "S03"
    assert mapping["S^(2)"]["watson_label"] == "S05"
    assert mapping["S^(3)"]["watson_label"] == "S07"


def test_map_generators_to_watson_exposes_origin_vj_table():
    out = map_generators_to_watson(n_modes=1, max_order=2, max_j=4, max_v=4)
    table = out["mapping"]["S^(1)"]["origin_vj_table"]
    assert len(table) > 0
