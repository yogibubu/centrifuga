import math

from ceditt_gui import _h22_intrinsic_share_diagnostic, _h22_scalar_diagnostic
from distortion_workflow import (
    classify_rotor_limit,
    project_special_quartic_constants,
    project_special_sextic_constants,
)


def test_diagnostic_d_matches_frobenius_ratio() -> None:
    tau_std = {
        "tau_xxxx": 10.0,
        "tau_yyyy": 6.0,
        "tau_zzzz": 4.0,
        "tau_xxyy": 2.0,
        "tau_xxzz": 1.0,
        "tau_yyzz": 3.0,
    }
    tau_h22 = {
        "tau_xxxx": 1.0,
        "tau_yyyy": 0.6,
        "tau_zzzz": 0.4,
        "tau_xxyy": 0.2,
        "tau_xxzz": 0.1,
        "tau_yyzz": 0.3,
    }
    diag = _h22_scalar_diagnostic(tau_std, tau_h22)
    assert abs(diag["D"] - 0.1) < 1.0e-12
    assert abs(diag["componentwise"]["aa"] - 0.1) < 1.0e-12
    assert abs(diag["componentwise"]["bc"] - 0.1) < 1.0e-12


def test_diagnostic_dn_matches_intrinsic_share() -> None:
    tau_h22 = {
        "tau_xxxx": 1.0,
        "tau_yyyy": 0.6,
        "tau_zzzz": 0.4,
        "tau_xxyy": 0.2,
        "tau_xxzz": 0.1,
        "tau_yyzz": 0.3,
    }
    tau_intrinsic = {
        "tau_xxxx": 0.5,
        "tau_yyyy": 0.3,
        "tau_zzzz": 0.2,
        "tau_xxyy": 0.1,
        "tau_xxzz": 0.05,
        "tau_yyzz": 0.15,
    }
    diag = _h22_intrinsic_share_diagnostic(tau_h22, tau_intrinsic)
    assert abs(diag["D_N"] - 0.5) < 1.0e-12


def test_special_limit_classifies_spherical_top() -> None:
    rotor = classify_rotor_limit((1000.0, 1000.0, 1000.0), (10.0, 10.0, 10.0))
    assert rotor["kind"] == "spherical_top"
    assert rotor["is_special_limit"] is True


def test_project_special_quartic_prolate_from_tensor() -> None:
    # DJ=2, DJK=3, DK=5 -> coefficients in the a,b,c monomial basis.
    tau = {
        "tau_xxxx": 10.0,
        "tau_yyyy": 2.0,
        "tau_zzzz": 2.0,
        "tau_xxyy": 7.0,
        "tau_xxzz": 7.0,
        "tau_yyzz": 4.0,
    }
    proj = project_special_quartic_constants(tau, abc=(10000.0, 100.0, 100.0), moments=(1.0, 2.0, 2.0))
    assert proj is not None
    got = proj["quartic_mhz"]
    assert abs(got["DJ"] - 2.0) < 1.0e-12
    assert abs(got["DJK"] - 3.0) < 1.0e-12
    assert abs(got["DK"] - 5.0) < 1.0e-12


def test_project_special_sextic_linear_scalar() -> None:
    # H = 7 in the transverse isotropic linear limit.
    phi = {
        "aaa": 0.0,
        "aab": 0.0,
        "aac": 0.0,
        "abb": 0.0,
        "abc": 0.0,
        "acc": 0.0,
        "bbb": 7.0,
        "bbc": 21.0,
        "bcc": 21.0,
        "ccc": 7.0,
    }
    proj = project_special_sextic_constants(phi, abc=(math.inf, 100.0, 100.0), moments=(0.0, 2.0, 2.0))
    assert proj is not None
    assert abs(proj["sextic_hz"]["H"] - 7.0) < 1.0e-12
