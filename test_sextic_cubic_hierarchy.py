#!/usr/bin/env python3
"""Check the sextic cubic hierarchy split into semi-diagonal and 3-index parts."""

from __future__ import annotations

from gaussian_vpt_parser import (
    align_gaussian_cubic_force_constants,
    parse_gaussian_anharmonic_force_data,
    parse_gaussian_quartic_benchmark,
)
from compare_gaussian_sextic import _aligned_model_from_gaussian, sextic_cubic_hierarchy_hz


def main() -> None:
    model, _rep, _order, _signs, _didq_err = _aligned_model_from_gaussian("h2o.fchk", "h2o.log")
    anh = parse_gaussian_anharmonic_force_data("h2o.log")
    quart = parse_gaussian_quartic_benchmark("h2o.log")
    _mapping, phi3_reduced_cm, _phi3_raw = align_gaussian_cubic_force_constants(anh, abs(model.vib_freq_cm))

    hierarchy = sextic_cubic_hierarchy_hz(model, phi3_reduced_cm, quart.spectroscopic_axes)
    max_err = 0.0
    for vals in hierarchy.values():
        err = abs(vals["geometry_hz"] + vals["cubic_sd_hz"] + vals["cubic_3ind_hz"] - vals["total_full_hz"])
        max_err = max(max_err, err)
    print(f"max |geometry + cubic_sd + cubic_3ind - total| = {max_err:.3e}")
    if max_err > 1.0e-6:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
