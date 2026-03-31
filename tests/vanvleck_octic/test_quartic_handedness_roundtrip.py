import numpy as np

from ceditt_gui import quartic_handedness_flip_constants


def test_quartic_handedness_flip_roundtrip_a_reduction() -> None:
    din = np.array([0.02996, -0.08096, 1.209, -0.002677, 0.1148], dtype=float)
    dout = quartic_handedness_flip_constants(din, "A")
    back = quartic_handedness_flip_constants(dout, "A")
    np.testing.assert_allclose(dout, np.array([0.02996, -0.08096, 1.209, 0.002677, -0.1148]))
    np.testing.assert_allclose(back, din)


def test_quartic_handedness_flip_roundtrip_s_reduction() -> None:
    din = np.array([0.02996, -0.08096, 1.209, -0.002677, 0.1148], dtype=float)
    dout = quartic_handedness_flip_constants(din, "S")
    back = quartic_handedness_flip_constants(dout, "S")
    np.testing.assert_allclose(dout, np.array([0.02996, -0.08096, 1.209, 0.002677, 0.1148]))
    np.testing.assert_allclose(back, din)
