#!/usr/bin/env python3

from scripts.linear_quartic_scalar_basis import linear_quartic_scalar_basis


def test_linear_quartic_scalar_basis_coefficients() -> None:
    summary = linear_quartic_scalar_basis()
    assert summary["plane_coefficients"] == {"c_004": 1, "c_022": 2, "c_040": 1}
