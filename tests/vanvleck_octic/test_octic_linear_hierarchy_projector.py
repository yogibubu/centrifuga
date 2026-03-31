#!/usr/bin/env python3

from scripts.octic_linear_hierarchy_projector import octic_linear_hierarchy_projector


def test_linear_hierarchy_projector_is_exact() -> None:
    summary = octic_linear_hierarchy_projector()
    assert summary["left_inverse_exact"] is True
