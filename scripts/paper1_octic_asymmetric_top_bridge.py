#!/usr/bin/env python3
"""Dimension bridge between the cartesian octic basis and Aliev 1983."""

from __future__ import annotations


def paper1_octic_asymmetric_top_bridge() -> dict[str, object]:
    """Return the exact dimension bridge used in the octic manuscript."""
    cartesian_commuting = 15
    aliev_determinable = 9
    redundant_directions = cartesian_commuting - aliev_determinable
    return {
        "cartesian_commuting_coefficients": cartesian_commuting,
        "aliev1983_determinable_combinations": aliev_determinable,
        "redundant_directions": redundant_directions,
        "statement": (
            "The commuting asymmetric-top octic operator has 15 cartesian "
            "coefficients, whereas Aliev 1983 reports only 9 determinable "
            "asymmetric-top combinations. The non-linear reduction problem "
            "therefore contains a 6-dimensional redundancy between the raw "
            "commuting cartesian basis and the spectroscopically determinable "
            "octic parameter set."
        ),
    }


if __name__ == "__main__":
    print(paper1_octic_asymmetric_top_bridge())
