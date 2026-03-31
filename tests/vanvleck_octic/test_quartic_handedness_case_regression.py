import numpy as np

from ceditt_gui import _rotate_abc, quartic_handedness_flip_constants, transform_quartic_tensor


def test_ir_to_iiil_roundtrip_closes_on_historical_regression_case() -> None:
    A, B, C = 2019.517, 571.4362, 460.3632
    din = np.array([0.02996, -0.08096, 1.209, -0.002677, 0.1148], dtype=float)

    d_iiir = transform_quartic_tensor(din, A, B, C, "I", "III", "A")
    d_iiil = quartic_handedness_flip_constants(d_iiir, "A")
    d_iiir_back = quartic_handedness_flip_constants(d_iiil, "A")

    A3, B3, C3 = _rotate_abc(A, B, C, "I", "III")
    back = transform_quartic_tensor(d_iiir_back, A3, B3, C3, "III", "I", "A")

    np.testing.assert_allclose(back, din, atol=1.0e-12, rtol=0.0)
