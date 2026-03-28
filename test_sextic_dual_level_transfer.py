#!/usr/bin/env python3
"""Check dual-level sextic transfer using distinct omega and cubic mode sets."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ceditt_gui import _mode_overlap_mapping_from_fchks, _raw_cubic_to_reduced_with_target_freqs
from channel_contributions import parse_model
from compare_gaussian_sextic import sextic_cubic_hierarchy_hz
from gaussian_vpt_parser import align_gaussian_cubic_force_constants, parse_gaussian_anharmonic_force_data


COMPS = ("aaa", "aab", "aac", "abb", "abc", "acc", "bbb", "bbc", "bcc", "ccc")


def _summary(model, phi3_reduced_cm):
    levels = sextic_cubic_hierarchy_hz(model, phi3_reduced_cm, {"a": 0, "b": 1, "c": 2})
    dominant = max(COMPS, key=lambda key: abs(levels[key]["total_full_hz"]))
    max_geom = max(abs(levels[key]["geometry_hz"]) for key in COMPS)
    max_iij = max(abs(levels[key]["cubic_sd_hz"]) for key in COMPS)
    max_ijk = max(abs(levels[key]["cubic_3ind_hz"]) for key in COMPS)
    max_total = max(abs(levels[key]["total_full_hz"]) for key in COMPS)
    return dominant, max_geom, max_iij, max_ijk, max_total


def _dual_level_summary(base_tag: str, high_tag: str):
    model_hi, _, meta_hi = parse_model(Path(f"{high_tag}.fchk"), Path(f"{high_tag}.log"))
    anh_lo = parse_gaussian_anharmonic_force_data(f"{base_tag}.log")
    overlap = _mode_overlap_mapping_from_fchks(f"{base_tag}.fchk", f"{high_tag}.fchk")
    _, _phi3r_lo, phi3raw_lo = align_gaussian_cubic_force_constants(anh_lo, np.abs(parse_model(Path(f"{base_tag}.fchk"), Path(f"{base_tag}.log"))[0].vib_freq_cm))
    from gaussian_vpt_parser import reorder_cubic_force_constants, apply_mode_signs_to_cubic_force_constants

    raw_target = reorder_cubic_force_constants(phi3raw_lo, overlap["mapping"])
    raw_target = apply_mode_signs_to_cubic_force_constants(raw_target, overlap["target_signs"])
    phi3_reduced_dual = _raw_cubic_to_reduced_with_target_freqs(raw_target, model_hi.vib_freq_cm)
    return overlap, _summary(model_hi, phi3_reduced_dual), _summary(model_hi, meta_hi["phi3_reduced_cm"])


def test_h2o_dual_level_transfer_tracks_full_dpcs3() -> None:
    overlap, dual, full = _dual_level_summary("h2o", "h2o_dpcs3")
    assert overlap["min_abs_overlap"] > 0.9999
    assert dual[0] == full[0] == "aaa"
    assert abs(dual[4] - full[4]) / full[4] < 0.02
    assert abs(dual[2] - full[2]) / full[2] < 0.10


def test_hdo_dual_level_transfer_tracks_full_dpcs3() -> None:
    overlap, dual, full = _dual_level_summary("hdo", "hdo_dpcs3")
    assert overlap["min_abs_overlap"] > 0.9997
    assert dual[0] == full[0] == "aaa"
    assert abs(dual[4] - full[4]) / full[4] < 0.02
    assert abs(dual[2] - full[2]) / full[2] < 0.05
    assert abs(dual[3] - full[3]) / full[3] < 0.05


def test_nno_dual_level_transfer_is_finite_and_nontrivial() -> None:
    overlap, dual, full = _dual_level_summary(
        "/Users/vincenzobarone/centrifugal/nno_HPCS2",
        "/Users/vincenzobarone/centrifugal/nno_DPCS3",
    )
    assert overlap["min_abs_overlap"] > 0.95
    assert overlap["max_offdiag_abs_overlap"] > 0.2
    assert dual[0] == full[0] == "bbc"
    assert np.isfinite(dual[4])
    assert np.isfinite(full[4])
