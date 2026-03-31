#!/usr/bin/env python3
"""Check the quadratic/linear sextic response in the tau-only geometry sector."""

from __future__ import annotations

import numpy as np

from compare_gaussian_sextic import (
    _aligned_model_from_gaussian,
    _quartic_tau_from_model,
    _representation_axis_values,
    sextic_tau_geometry_only_hz,
    sextic_tau_linear_response_hz,
)
from gaussian_vpt_parser import parse_gaussian_quartic_benchmark


def main() -> None:
    model, _rep, _order, _signs, _didq_err = _aligned_model_from_gaussian("h2o.fchk", "h2o.log")
    quart = parse_gaussian_quartic_benchmark("h2o.log")
    tau_std, _ = _quartic_tau_from_model(model)
    _pmom, rot_cm = _representation_axis_values(model)

    # Use a small but generic perturbation in the same symmetry class to test
    # the polarization identity for the tau-only sextic geometry functional.
    delta_tau = 0.125 * tau_std

    base = sextic_tau_geometry_only_hz(tau_std, rot_cm, quart.spectroscopic_axes)
    quadratic = sextic_tau_geometry_only_hz(delta_tau, rot_cm, quart.spectroscopic_axes)
    updated = sextic_tau_geometry_only_hz(tau_std + delta_tau, rot_cm, quart.spectroscopic_axes)
    response = sextic_tau_linear_response_hz(tau_std, delta_tau, rot_cm, quart.spectroscopic_axes)

    max_err = 0.0
    for key, vals in response.items():
        err = abs(vals["linear_hz"] + vals["quadratic_hz"] - vals["delta_total_hz"])
        err = max(err, abs(vals["base_hz"] - base[key]))
        err = max(err, abs(vals["quadratic_hz"] - quadratic[key]))
        err = max(err, abs(vals["updated_hz"] - updated[key]))
        max_err = max(max_err, err)

    print(f"max |linear + quadratic - delta_total| = {max_err:.3e}")
    if max_err > 1.0e-6:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
