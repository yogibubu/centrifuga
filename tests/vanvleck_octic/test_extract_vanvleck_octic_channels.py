#!/usr/bin/env python3

from tools.vanvleck.extract_vanvleck_octic_channels import extract_reference_octic_channels


def test_extract_reference_octic_channels_exposes_expected_labels():
    out = extract_reference_octic_channels(n_modes=1, max_order=4, max_j=8, max_v=8)
    summaries = out["summaries"]
    assert "[S^(1),[S^(1),[S^(1),H02]]]" in summaries
    assert "[S^(1),[S^(1),H04]]" in summaries
    assert "[S^(1),H06]" in summaries
    assert "[S^(2),H04]" in summaries


def test_extract_reference_octic_channels_returns_exact_summary_payloads():
    out = extract_reference_octic_channels(n_modes=1, max_order=4, max_j=8, max_v=8)
    for payload in out["summaries"].values():
        assert "term_count" in payload
        assert "octic_rot_ground_count" in payload
        assert "octic_commuting" in payload


def test_extract_reference_octic_channels_two_mode_scan_runs():
    out = extract_reference_octic_channels(n_modes=2, max_order=4, max_j=8, max_v=8)
    summaries = out["summaries"]
    assert "[S^(2),H04]" in summaries
