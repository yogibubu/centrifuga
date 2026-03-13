#!/usr/bin/env python3
"""Minimal quartic channel builder for reconstruction."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable

import sympy as sp
import numpy as np

MON_KEYS = ("xxxx", "yyyy", "zzzz", "xxyy", "xxzz", "yyzz")


def _zero_tau() -> Dict[str, sp.Expr]:
    return {f"tau_{key}": sp.Integer(0) for key in MON_KEYS}


def _complete_tau(partial: Dict[str, sp.Expr]) -> Dict[str, sp.Expr]:
    base = _zero_tau()
    for key, val in partial.items():
        base[key] = sp.simplify(val)
    return base


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
            w = coef / (omega[k] * omega[l])
            tau["tau_xxxx"] += mu2[0, 0, k, l] ** 2 * w
            tau["tau_yyyy"] += mu2[1, 1, k, l] ** 2 * w
            tau["tau_zzzz"] += mu2[2, 2, k, l] ** 2 * w
    return _complete_tau(tau)


def channel_h12h30(mu1, mu2, phi3: sp.MutableDenseNDimArray, omega: Iterable[float], hbar: sp.Symbol, seed=None, exact_calibration=False):
    tau = defaultdict(lambda: sp.Integer(0))
    w = sp.Rational(1, 4) * hbar
    n_modes = mu1.shape[2]
    for k in range(n_modes):
        tau["tau_xxxx"] += phi3[k, k, k] * mu1[0, 0, k] * w / (omega[k] + 1)
        tau["tau_yyyy"] += phi3[k, k, k] * mu1[1, 1, k] * w / (omega[k] + 1)
        tau["tau_zzzz"] += phi3[k, k, k] * mu1[2, 2, k] * w / (omega[k] + 1)
    return _complete_tau(tau)


def channel_h30h30(mu1, phi3: sp.MutableDenseNDimArray, omega: Iterable[float], hbar: sp.Symbol, seed: int, exact_calibration=False):
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
