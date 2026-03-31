#!/usr/bin/env python3
"""Exact hierarchy counts for the octic operator reduction."""

from __future__ import annotations


def octic_hierarchy_bridge() -> dict[str, object]:
    """Return the exact count hierarchy used in the octic manuscript."""
    asymmetric_cartesian = 15
    asymmetric_determinable = 9
    symmetric_top = 5
    linear = 1
    return {
        "hierarchy": (asymmetric_cartesian, asymmetric_determinable, symmetric_top, linear),
        "labels": ("cartesian", "asymmetric-top determinable", "symmetric-top", "linear"),
        "drops": (
            asymmetric_cartesian - asymmetric_determinable,
            asymmetric_determinable - symmetric_top,
            symmetric_top - linear,
        ),
        "statement": (
            "The rigorous octic count hierarchy is 15 -> 9 -> 5 -> 1: "
            "15 commuting cartesian coefficients, 9 determinable asymmetric-top "
            "combinations, 5 symmetric-top constants, and 1 linear constant."
        ),
    }


if __name__ == "__main__":
    print(octic_hierarchy_bridge())
