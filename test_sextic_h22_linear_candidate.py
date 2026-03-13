#!/usr/bin/env python3
"""Check that the first sextic H22-induced linear candidate is well defined."""

from __future__ import annotations

from compare_gaussian_sextic import _aligned_model_from_gaussian, sextic_h22_linear_candidate_hz
from gaussian_vpt_parser import parse_gaussian_quartic_benchmark


def main() -> None:
    model, _rep, _order, _signs, _didq_err = _aligned_model_from_gaussian("h2o.fchk", "h2o.log")
    quart = parse_gaussian_quartic_benchmark("h2o.log")
    response = sextic_h22_linear_candidate_hz(model, quart.spectroscopic_axes)

    max_abs = 0.0
    for vals in response.values():
        max_abs = max(max_abs, abs(vals["linear_hz"]))

    print(f"max |sextic H22 linear candidate| = {max_abs:.3e} Hz")
    if max_abs <= 1.0e-12:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
