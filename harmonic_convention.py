#!/usr/bin/env python3
"""Source-independent harmonic convention helpers.

The harmonic convention must be fixed from geometry + Hessian alone. Anharmonic
data may then be imported only after verifying how their normal-mode ordering
matches the harmonic model; they must not choose the rotational axes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gaussian_vpt_parser import GaussianAnharmonicForceData, align_gaussian_cubic_force_constants
from rovib_distortion import harmonic_inertia_model_from_geometry_hessian


DEFAULT_QUASI_SYMMETRIC_KAPPA = 0.8


def ray_asymmetry_kappa(abc_mhz: np.ndarray) -> float:
    abc = np.asarray(abc_mhz, dtype=float).reshape(3)
    a, b, c = (float(abc[0]), float(abc[1]), float(abc[2]))
    den = a - c
    if abs(den) <= 1.0e-14:
        return 0.0
    return (2.0 * b - a - c) / den


def default_harmonic_representation(
    abc_mhz: np.ndarray,
    *,
    quasi_symmetric_kappa: float = DEFAULT_QUASI_SYMMETRIC_KAPPA,
) -> tuple[str, float]:
    """Return the default right-handed harmonic representation from ``A,B,C``.

    Policy:
    - near-oblate: default to ``Ir``
    - near-prolate: default to ``IIIr``
    - otherwise keep ``Ir`` as the generic asymmetric-top default
    """
    kappa = ray_asymmetry_kappa(abc_mhz)
    if kappa <= -abs(quasi_symmetric_kappa):
        return "III", kappa
    return "I", kappa


def build_default_harmonic_model(
    masses_amu: np.ndarray,
    coords_ang: np.ndarray,
    hessian: np.ndarray,
    *,
    quasi_symmetric_kappa: float = DEFAULT_QUASI_SYMMETRIC_KAPPA,
    symbols: list[str] | None = None,
):
    probe = harmonic_inertia_model_from_geometry_hessian(
        masses_amu,
        coords_ang,
        hessian,
        representation="I",
        symbols=symbols,
    )
    rep, kappa = default_harmonic_representation(
        probe.abc_mhz,
        quasi_symmetric_kappa=quasi_symmetric_kappa,
    )
    if rep == "I":
        return probe, {"representation": "Ir", "kappa": float(kappa)}
    model = harmonic_inertia_model_from_geometry_hessian(
        masses_amu,
        coords_ang,
        hessian,
        representation=rep,
        symbols=symbols,
    )
    return model, {"representation": f"{rep}r", "kappa": float(kappa)}


@dataclass(frozen=True)
class CubicAlignmentCheck:
    mapping: tuple[int, ...]
    max_abs_freq_delta_cm: float

    @property
    def is_identity(self) -> bool:
        return self.mapping == tuple(range(len(self.mapping)))


def align_cubic_to_harmonic_model(
    anh: GaussianAnharmonicForceData,
    target_freq_cm: np.ndarray,
) -> tuple[np.ndarray, CubicAlignmentCheck]:
    """Return cubic constants reordered onto the harmonic convention.

    The check is intentionally limited to the normal-mode labelling:
    cubic force constants are reordered by frequency only. No axis selection and
    no mode-sign fitting are performed here.
    """
    mapping, _phi3_reduced_cm, phi3_raw_au = align_gaussian_cubic_force_constants(
        anh,
        np.abs(np.asarray(target_freq_cm, dtype=float)),
    )
    source = np.abs(np.asarray(anh.frequencies_cm, dtype=float))
    target = np.abs(np.asarray(target_freq_cm, dtype=float))
    aligned_source = source[np.asarray(mapping, dtype=int)]
    max_delta = float(np.max(np.abs(aligned_source - target))) if target.size else 0.0
    return phi3_raw_au, CubicAlignmentCheck(
        mapping=tuple(int(x) for x in np.asarray(mapping, dtype=int).tolist()),
        max_abs_freq_delta_cm=max_delta,
    )
