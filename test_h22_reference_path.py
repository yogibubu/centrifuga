#!/usr/bin/env python3

from pathlib import Path

from h22_reference import project_h22_ceditt3_reference_path


REFERENCE = {
    "h2o": {"DJ": -67.650101, "DJK": -246.335379, "DK": 287.818749, "d1": -97.130484, "d2": 65.305719},
    "h2s": {"DJ": -5.917341, "DJK": 1.331355, "DK": -3.88703, "d1": -4.153058, "d2": 1.469627},
    "h2co": {"DJ": -0.003691, "DJK": -0.17658, "DK": -8.736904, "d1": -0.001452, "d2": 0.000783},
    "h2cs": {"DJ": -0.000532, "DJK": -0.037965, "DK": -11.435701, "d1": -0.000114, "d2": 0.00005},
}

TOLERANCE_KHZ = {
    "h2o": 1.0e-5,
    "h2s": 1.0e-5,
    "h2co": 0.1,
    "h2cs": 0.1,
}


def test_h22_ceditt3_reference_path() -> None:
    for species, ref in REFERENCE.items():
        got = project_h22_ceditt3_reference_path(Path(f"{species}.fchk"), Path(f"{species}.log"))
        err = max(abs(got[key] - ref[key]) for key in ref)
        assert err <= TOLERANCE_KHZ[species], (species, err, got, ref)
