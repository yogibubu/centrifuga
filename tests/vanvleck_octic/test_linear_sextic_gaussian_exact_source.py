from __future__ import annotations

from pathlib import Path

import numpy as np

from channel_contributions import parse_model
from compare_gaussian_sextic import _classify_rotor_limit
from distortion_workflow import _linear_gaussian_exact_source_blocks, _linear_gaussian_exact_source_h_hz
from gaussian_vpt_parser import parse_gaussian_linear_rotdist_constants


def _check_case(fchk_path: str, log_path: str, tol_hz: float = 1.0e-6) -> None:
    model, _, _ = parse_model(Path(fchk_path), Path(log_path))
    rotd = parse_gaussian_linear_rotdist_constants(log_path)
    assert rotd.h_mhz is not None
    blocks = _linear_gaussian_exact_source_blocks(log_path)
    rotor_limit = _classify_rotor_limit(np.asarray(model.abc_mhz, dtype=float), np.asarray(model.moments_amu_a2, dtype=float))
    out = _linear_gaussian_exact_source_h_hz(
        model,
        gaussian_source_blocks=blocks,
        d_hz=None if rotd.d_mhz is None else rotd.d_mhz * 1.0e6,
        rotor_limit=rotor_limit,
    )
    assert out is not None
    assert out["primary_component"] == "bbb"
    assert abs(float(out["H"]) - float(rotd.h_mhz * 1.0e6)) < tol_hz


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data/gaussian/linear_cases"

def test_linear_sextic_exact_source_matches_gaussian_nno_and_hc3n() -> None:
    _check_case(str(DATA_ROOT / "nno_HPCS2.fchk"), str(REPO_ROOT / "nno_HPCS2.log"))
    _check_case(str(DATA_ROOT / "nno_DPCS3.fchk"), str(REPO_ROOT / "nno_DPCS3.log"))
    _check_case(str(DATA_ROOT / "hc3n_HPCS2.fchk"), str(REPO_ROOT / "hc3n_HPCS2.log"))
    _check_case(str(DATA_ROOT / "hc3n_DPCS3.fchk"), str(REPO_ROOT / "hc3n_DPCS3.log"))
