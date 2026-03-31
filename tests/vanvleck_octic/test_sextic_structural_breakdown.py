#!/usr/bin/env python3
"""Check the selected sextic geometry/anharmonic grouping."""

from __future__ import annotations

from gaussian_vpt_parser import align_gaussian_cubic_force_constants, parse_gaussian_anharmonic_force_data, parse_gaussian_quartic_benchmark
from compare_gaussian_sextic import _aligned_model_from_gaussian, sextic_structural_breakdown_hz


def main() -> None:
    model, _rep, _order, _signs, _didq_err = _aligned_model_from_gaussian("h2o.fchk", "h2o.log")
    anh = parse_gaussian_anharmonic_force_data("h2o.log")
    quart = parse_gaussian_quartic_benchmark("h2o.log")
    _, phi3_reduced_cm, _ = align_gaussian_cubic_force_constants(anh, abs(model.vib_freq_cm))
    breakdown = sextic_structural_breakdown_hz(model, phi3_reduced_cm, quart.spectroscopic_axes)

    max_err = 0.0
    for key, vals in breakdown.items():
        err = abs(vals["geometry_hz"] + vals["anharmonic_cubic_hz"] - vals["phi_hz"])
        max_err = max(max_err, err)
    print(f"max |geometry + cubic - phi| = {max_err:.3e}")
    if max_err > 1.0e-6:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
