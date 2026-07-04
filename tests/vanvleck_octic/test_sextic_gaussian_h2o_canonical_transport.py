from __future__ import annotations

from pathlib import Path

import numpy as np

from ceditt_gui import (
    _flip_handedness_abc,
    sextic_handedness_flip_constants,
    transform_sextic_tensor,
)
from gaussian_vpt_parser import (
    parse_gaussian_harmonic_data,
    parse_gaussian_sextic_benchmark,
)


REPO = Path(__file__).resolve().parents[2]
H2O_LOG = REPO / "gaussian" / "h2o.log"
S_KEYS = ["H J", "H JK", "H KJ", "H K", "h 1", "h 2", "h 3"]
A_KEYS = ["Phi J", "Phi JK", "Phi KJ", "Phi K", "phi j", "phi jk", "phi k"]


def _h2o_reference_data() -> tuple[tuple[float, float, float], np.ndarray, np.ndarray]:
    harmonic = parse_gaussian_harmonic_data(H2O_LOG).reordered_to_anharmonic()
    sextic = parse_gaussian_sextic_benchmark(H2O_LOG)
    abc = tuple(float(x) * 1000.0 for x in harmonic.rot_ghz)
    h_s = np.array([sextic.s_reduction_hz[k] / 1.0e6 for k in S_KEYS], dtype=float)
    h_a = np.array([sextic.a_reduction_hz[k] / 1.0e6 for k in A_KEYS], dtype=float)
    return abc, h_s, h_a


def _permuted_abc(
    abc: tuple[float, float, float],
    rep: str,
) -> tuple[float, float, float]:
    a, b, c = abc
    if rep == "I":
        return a, b, c
    if rep == "II":
        return b, c, a
    if rep == "III":
        return c, a, b
    raise ValueError(f"Unsupported representation: {rep}")


def test_sextic_h2o_native_a_s_reductions_match_gaussian() -> None:
    abc, h_s, h_a = _h2o_reference_data()

    np.testing.assert_allclose(
        transform_sextic_tensor(h_a, *abc, "I", "I", "A", "S"),
        h_s,
        atol=1.0e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        transform_sextic_tensor(h_s, *abc, "I", "I", "S", "A"),
        h_a,
        atol=1.0e-8,
        rtol=0.0,
    )


def test_sextic_h2o_canonical_transport_roundtrips() -> None:
    abc, h_s, h_a = _h2o_reference_data()

    for rep in ("II", "III"):
        abc_rep = _permuted_abc(abc, rep)
        h_s_rep = transform_sextic_tensor(h_s, *abc, "I", rep, "S", "S")
        h_s_back = transform_sextic_tensor(h_s_rep, *abc_rep, rep, "I", "S", "S")
        np.testing.assert_allclose(h_s_back, h_s, atol=1.0e-12, rtol=0.0)

        h_a_rep = transform_sextic_tensor(h_a, *abc, "I", rep, "A", "A")
        h_a_back = transform_sextic_tensor(h_a_rep, *abc_rep, rep, "I", "A", "A")
        np.testing.assert_allclose(h_a_back, h_a, atol=1.0e-9, rtol=0.0)

    flip_abc = _flip_handedness_abc(*abc)
    h_flip = sextic_handedness_flip_constants(h_s, *abc, "S")
    h_back = sextic_handedness_flip_constants(h_flip, *flip_abc, "S")
    np.testing.assert_allclose(h_back, h_s, atol=1.0e-12, rtol=0.0)
