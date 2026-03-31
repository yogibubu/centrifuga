import numpy as np

from ceditt_gui import sextic_handedness_flip_constants


def test_sextic_handedness_flip_roundtrip_s_reduction() -> None:
    hin = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=float)
    hout = sextic_handedness_flip_constants(hin, 10000.0, 5000.0, 3000.0, "S")
    back = sextic_handedness_flip_constants(hout, 10000.0, 3000.0, 5000.0, "S")
    np.testing.assert_allclose(hout, np.array([1.0, 2.0, 3.0, 4.0, -5.0, 6.0, -7.0]))
    np.testing.assert_allclose(back, hin)


def test_sextic_handedness_flip_roundtrip_a_reduction() -> None:
    hin = np.array([1.0, 2.0, 3.0, 4.0, 0.5, 0.25, -0.75], dtype=float)
    hout = sextic_handedness_flip_constants(hin, 10000.0, 5000.0, 3000.0, "A")
    back = sextic_handedness_flip_constants(hout, 10000.0, 3000.0, 5000.0, "A")
    np.testing.assert_allclose(back, hin, atol=1.0e-12, rtol=0.0)
