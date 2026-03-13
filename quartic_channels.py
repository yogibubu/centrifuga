#!/usr/bin/env python3
"""Minimal quartic channel builder for reconstruction.

The H12H30 and H30H30 routines in this file are intentionally minimal
placeholder reconstructions. They are useful for symbolic scaffolding and
software integration, but they do not yet reproduce the benchmark-faithful
Paper 2 cubic-cubic / mixed-channel formulas.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable
import warnings

import sympy as sp
import numpy as np

MON_KEYS = ("xxxx", "yyyy", "zzzz", "xxyy", "xxzz", "yyzz")
_PLACEHOLDER_WARNED: set[str] = set()


def _zero_tau() -> Dict[str, sp.Expr]:
    return {f"tau_{key}": sp.Integer(0) for key in MON_KEYS}


def _complete_tau(partial: Dict[str, sp.Expr]) -> Dict[str, sp.Expr]:
    base = _zero_tau()
    for key, val in partial.items():
        base[key] = sp.simplify(val)
    return base


def _warn_placeholder(channel: str) -> None:
    if channel in _PLACEHOLDER_WARNED:
        return
    warnings.warn(
        f"{channel} currently uses a minimal placeholder formula in quartic_channels.py; "
        "it is not yet the benchmark-faithful Paper 2 implementation.",
        RuntimeWarning,
        stacklevel=2,
    )
    _PLACEHOLDER_WARNED.add(channel)


def channel_h12h12(mu1: sp.MutableDenseNDimArray, omega: Iterable[float]) -> Dict[str, sp.Expr]:
    omega = list(omega)
    n_modes = len(omega)
    tau = defaultdict(lambda: sp.Integer(0))
    for k in range(n_modes):
        w = sp.Rational(1, 8) / omega[k] ** 2
        tau["tau_xxxx"] += mu1[0, 0, k] ** 2 * w
        tau["tau_yyyy"] += mu1[1, 1, k] ** 2 * w
        tau["tau_zzzz"] += mu1[2, 2, k] ** 2 * w
        tau["tau_xxyy"] += (mu1[0, 0, k] * mu1[1, 1, k]) * w
        tau["tau_xxzz"] += (mu1[0, 0, k] * mu1[2, 2, k]) * w
        tau["tau_yyzz"] += (mu1[1, 1, k] * mu1[2, 2, k]) * w
    return _complete_tau(tau)


def channel_h22(mu2: sp.MutableDenseNDimArray, omega: Iterable[float], hbar: sp.Symbol) -> Dict[str, sp.Expr]:
    omega = list(omega)
    n_modes = len(omega)
    tau = defaultdict(lambda: sp.Integer(0))
    coef = sp.Rational(1, 32) * hbar
    for k in range(n_modes):
        for l in range(n_modes):
            symmetry = sp.Integer(2) if k == l else sp.Integer(1)
            w = coef * symmetry / (omega[k] * omega[l])
            xx = mu2[0, 0, k, l]
            yy = mu2[1, 1, k, l]
            zz = mu2[2, 2, k, l]
            xy = mu2[0, 1, k, l]
            yx = mu2[1, 0, k, l]
            xz = mu2[0, 2, k, l]
            zx = mu2[2, 0, k, l]
            yz = mu2[1, 2, k, l]
            zy = mu2[2, 1, k, l]

            tau["tau_xxxx"] += xx**2 * w
            tau["tau_yyyy"] += yy**2 * w
            tau["tau_zzzz"] += zz**2 * w
            tau["tau_xxyy"] += (xx * yy + xy * xy + xy * yx + yx * yx) * w
            tau["tau_xxzz"] += (xx * zz + xz * xz + xz * zx + zx * zx) * w
            tau["tau_yyzz"] += (yy * zz + yz * yz + yz * zy + zy * zy) * w
    return _complete_tau(tau)


def bilinear_modepair_from_mu1(
    mu1: sp.MutableDenseNDimArray,
    inertia0: sp.MutableDenseNDimArray,
) -> sp.MutableDenseNDimArray:
    """Build the bilinear mode-pair closure B(mu1) = mu1 I mu1 + mu1 I mu1."""
    mu1_np = np.asarray(mu1.tolist(), dtype=float)
    inertia0_np = np.asarray(inertia0.tolist(), dtype=float)
    n_modes = mu1_np.shape[2]
    bilinear = np.zeros((3, 3, n_modes, n_modes), dtype=float)
    for k in range(n_modes):
        for l in range(n_modes):
            bilinear[:, :, k, l] = (
                mu1_np[:, :, k] @ inertia0_np @ mu1_np[:, :, l]
                + mu1_np[:, :, l] @ inertia0_np @ mu1_np[:, :, k]
            )
    flat = [sp.Float(float(x)) for x in bilinear.reshape(-1)]
    return sp.MutableDenseNDimArray(flat, bilinear.shape)


def channel_h22_decomposed(
    bilinear: sp.MutableDenseNDimArray,
    intrinsic: sp.MutableDenseNDimArray,
    omega: Iterable[float],
    hbar: sp.Expr,
) -> dict[str, Dict[str, sp.Expr]]:
    """Return H22[B,B], H22[B,N], H22[N,N], and the assembled total."""
    total = sp.MutableDenseNDimArray(
        [sp.simplify(bilinear[idx] - intrinsic[idx]) for idx in np.ndindex(bilinear.shape)],
        bilinear.shape,
    )

    bb = channel_h22(bilinear, omega, hbar)
    bn_tensor = sp.MutableDenseNDimArray(
        [sp.simplify((bilinear[idx] + intrinsic[idx]) / 2) for idx in np.ndindex(bilinear.shape)],
        bilinear.shape,
    )
    # Use polarization on the quadratic form to isolate the mixed interference:
    #   Q(B-N) = Q(B) - 2 cross(B,N) + Q(N)
    total_tau = channel_h22(total, omega, hbar)
    nn = channel_h22(intrinsic, omega, hbar)
    cross = {
        key: sp.simplify((bb[key] + nn[key] - total_tau[key]) / 2)
        for key in bb
    }
    return {
        "total": total_tau,
        "bilinear": bb,
        "intrinsic": nn,
        "cross": cross,
    }


def channel_h12h30(mu1, mu2, phi3: sp.MutableDenseNDimArray, omega: Iterable[float], hbar: sp.Symbol, seed=None, exact_calibration=False):
    _warn_placeholder("H12H30")
    tau = defaultdict(lambda: sp.Integer(0))
    n_modes = mu1.shape[2]
    for k in range(n_modes):
        w = sp.Rational(1, 4) * hbar
        tau["tau_xxxx"] += phi3[k, k, k] * mu1[0, 0, k] * w / (omega[k] + 1)
        tau["tau_yyyy"] += phi3[k, k, k] * mu1[1, 1, k] * w / (omega[k] + 1)
        tau["tau_zzzz"] += phi3[k, k, k] * mu1[2, 2, k] * w / (omega[k] + 1)
    return _complete_tau(tau)


def channel_h30h30(mu1, phi3: sp.MutableDenseNDimArray, omega: Iterable[float], hbar: sp.Symbol, seed: int, exact_calibration=False):
    _warn_placeholder("H30H30")
    tau = defaultdict(lambda: sp.Integer(0))
    n_modes = mu1.shape[2]
    for i in range(n_modes):
        for j in range(n_modes):
            tau["tau_xxxx"] += phi3[i, j, j] * mu1[0, 0, i] * mu1[0, 0, j] / (omega[i] + omega[j] + 1)
            tau["tau_yyyy"] += phi3[i, j, j] * mu1[1, 1, i] * mu1[1, 1, j] / (omega[i] + omega[j] + 1)
            tau["tau_zzzz"] += phi3[i, j, j] * mu1[2, 2, i] * mu1[2, 2, j] / (omega[i] + omega[j] + 1)
    return _complete_tau(tau)


def tau_to_watson_a(tau: Dict[str, sp.Expr]) -> Dict[str, sp.Expr]:
    c = {
        (4, 0, 0): tau["tau_xxxx"],
        (0, 4, 0): tau["tau_yyyy"],
        (0, 0, 4): tau["tau_zzzz"],
        (2, 2, 0): tau["tau_xxyy"],
        (2, 0, 2): tau["tau_xxzz"],
        (0, 2, 2): tau["tau_yyzz"],
    }
    return {
        "DJ": sp.simplify(sp.Rational(1, 8) * (c[(2, 2, 0)] + c[(2, 0, 2)] + c[(0, 2, 2)])),
        "DJK": sp.simplify(sp.Rational(1, 8) * (-2 * c[(2, 2, 0)] + c[(2, 0, 2)] + c[(0, 2, 2)])),
        "DK": sp.simplify(sp.Rational(1, 8) * (c[(4, 0, 0)] + c[(0, 4, 0)] + c[(0, 0, 4)] - 2 * c[(2, 0, 2)] - 2 * c[(0, 2, 2)])),
        "d1": sp.simplify(sp.Rational(1, 8) * (c[(2, 0, 2)] - c[(0, 2, 2)])),
        "d2": sp.simplify(sp.Rational(1, 16) * (c[(4, 0, 0)] - c[(0, 4, 0)])),
    }


def channel_h22_from_mu1_intrinsic(
    mu1: sp.MutableDenseNDimArray,
    intrinsic: sp.MutableDenseNDimArray,
    inertia0: sp.MutableDenseNDimArray,
    omega: Iterable[float],
    hbar: sp.Expr,
) -> dict[str, Dict[str, sp.Expr]]:
    """Return a simple decomposition of H22 into bilinear and intrinsic pieces."""
    bilinear = bilinear_modepair_from_mu1(mu1, inertia0)
    return channel_h22_decomposed(bilinear, intrinsic, omega, hbar)
