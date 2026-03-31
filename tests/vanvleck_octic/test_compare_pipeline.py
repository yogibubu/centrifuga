#!/usr/bin/env python3
"""Tests for the experimental C2H2 comparison helpers."""

from __future__ import annotations

import numpy as np

from compare_pipeline import (
    align_c2h2_linear_aliev_modes,
    build_comparison_table,
    build_exp_quantities,
    format_comparison_report,
)
from exp_data_c2h2 import get_c2h2_data


def test_build_exp_quantities_returns_increments() -> None:
    q = build_exp_quantities(get_c2h2_data())
    assert q.modes == ("v1_CH", "v2_CC", "v3_asym", "v4_bend", "v5_bend")
    assert np.allclose(q.beta_exp, np.array([3e-9, 1e-9, 6e-9, 21e-9, 16e-9]))
    assert np.allclose(q.dH_exp, np.array([0.2e-13, 0.1e-13, 0.4e-13, 2.0e-13, 1.5e-13]))


def test_align_c2h2_linear_aliev_modes_uses_irrep_and_frequency_structure() -> None:
    payload = {
        "omega_parallel": [2088.0, 3441.0, 3541.0],
        "omega_perpendicular": [535.0, 775.0],
        "metadata": {
            "parallel_mode_irreps": ["Sigma_g+", "Sigma_u+", "Sigma_g+"],
            "perpendicular_pair_irreps": ["Pi_g", "Pi_u"],
        },
    }
    aligned = align_c2h2_linear_aliev_modes(
        payload=payload,
        beta_parallel=[20.0, 30.0, 10.0],
        beta_perpendicular=[40.0, 50.0],
        L_mode_calc=[2.0, 3.0, 1.0, 4.0, 5.0],
        L_total_calc=1.23,
    )
    assert aligned.modes == ("v1_CH", "v2_CC", "v3_asym", "v4_bend", "v5_bend")
    assert np.allclose(aligned.beta_calc, np.array([-10.0, -20.0, -30.0, -40.0, -50.0]))
    assert np.allclose(aligned.L_mode_calc, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert aligned.L_total_calc == 1.23


def test_format_comparison_report_mentions_total_L_if_modewise_not_available() -> None:
    table = build_comparison_table(
        ("v1_CH", "v2_CC"),
        [2.0, -3.0],
        [1.0, -1.0],
    )
    text = format_comparison_report(table, L_total_calc=1.0e-12)
    assert "QUARTIC CONSTANTS" in text
    assert "Modewise Delta H_k comparison not available" in text


def test_format_comparison_report_accepts_modewise_optical_table() -> None:
    beta = build_comparison_table(("v1_CH",), [2.0], [1.0])
    dH = build_comparison_table(("v1_CH",), [3.0e-14], [1.0e-14])
    text = format_comparison_report(beta, beta_note="qualitative only", dH_table=dH, dH_note="provisional mode partition")
    assert "OPTICAL CONSTANTS" in text
    assert "qualitative only" in text
    assert "provisional mode partition" in text
