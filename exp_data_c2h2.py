#!/usr/bin/env python3
"""Experimental rovibrational centrifugal distortion data for acetylene.

Units are cm^-1.

The values currently stored here are a lightweight working dataset for
comparison and ranking diagnostics, not yet the final curated benchmark table.
"""

from __future__ import annotations


def get_c2h2_data() -> dict[str, object]:
    data: dict[str, object] = {}

    data["D0"] = 1.759e-6
    data["H0"] = 1.10e-12

    # Experimental-mode ordering:
    #   nu1, nu2, nu3, nu4, nu5
    data["modes"] = ["v1_CH", "v2_CC", "v3_asym", "v4_bend", "v5_bend"]

    data["Dv"] = [
        1.762e-6,
        1.760e-6,
        1.765e-6,
        1.780e-6,
        1.775e-6,
    ]

    data["Hv"] = [
        1.12e-12,
        1.11e-12,
        1.14e-12,
        1.30e-12,
        1.25e-12,
    ]

    return data
