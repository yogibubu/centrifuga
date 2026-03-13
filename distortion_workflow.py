#!/usr/bin/env python3
"""Utility workflows exposing a minimal subset of order‑2 quartic utilities."""

from __future__ import annotations

import math

from typing import Iterable

import numpy as np
import sympy as sp

from quartic_channels import channel_h12h12, tau_to_watson_a

EH_TO_MHZ = 6.57968392061e9
AU_FREQ_TO_CMINV = 219474.6313705


def _watson_dict_to_float(watson: dict[str, sp.Expr]) -> dict[str, float]:
    return {key: float(EH_TO_MHZ * sp.N(value)) for key, value in watson.items()}


def compute_order2_quartic(model) -> dict[str, object]:
    """Return order-2 quartic data derived from a harmonic inertia model."""
    omega = [abs(f) / AU_FREQ_TO_CMINV for f in np.asarray(model.vib_freq_cm, dtype=float)]
    mu1 = sp.MutableDenseNDimArray(np.asarray(model.dInv_au, dtype=float))
    tau = channel_h12h12(mu1, omega)
    watson_a = _watson_dict_to_float(tau_to_watson_a(tau))
    # In the A/S reductions the numerical values coincide at this level.
    result = {
        "rotor_limit": classify_rotor_limit(np.asarray(model.abc_mhz, dtype=float), np.asarray(model.moments_amu_a2, dtype=float)),
        "watson_a_mhz": watson_a,
        "watson_s_mhz": watson_a.copy(),
        "special_quartic_projection": project_special_quartic_constants(watson_a, np.asarray(model.abc_mhz, dtype=float), np.asarray(model.moments_amu_a2, dtype=float)),
    }
    return result


def classify_rotor_limit(abc: Iterable[float], moments: Iterable[float]) -> dict[str, object]:
    """Simple rotor classification used by the GUI."""
    abc = np.asarray(abc, dtype=float)
    kind = "asymmetric"
    is_special = False
    if math.isfinite(abc[0]) and math.isfinite(abc[1]) and math.isfinite(abc[2]):
        if np.isclose(abc[1], abc[2], atol=1e-5):
            kind = "prolate"
            is_special = True
        if np.isclose(abc[0], abc[1], atol=1e-5):
            kind = "oblate"
            is_special = True
        if min(abc[1], abc[2]) < 1e-6:
            kind = "linear"
            is_special = True
    return {
        "kind": kind,
        "is_special_limit": is_special,
    }


def linear_ltype_terms(model, *, quartic_special=None, sextic_special=None):
    """Placeholder that returns no diagnostic by default."""
    return None


def project_special_quartic_constants(quartic_mhz: dict[str, float], abc: np.ndarray, moments: np.ndarray) -> dict[str, object]:
    """Return a crude symmetry-adapted projection for special rotor limits."""
    return {"quartic_mhz": quartic_mhz.copy()}


def project_special_sextic_constants(linear_cart: dict[str, float], abc: np.ndarray, moments: np.ndarray) -> dict[str, object] | None:
    """Return a minimal sextic projection used for diagnostics."""
    if not linear_cart:
        return None
    return {"sextic_hz": linear_cart.copy()}
