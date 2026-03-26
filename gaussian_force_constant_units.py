#!/usr/bin/env python3
"""Conversions between Gaussian raw and reduced anharmonic force constants."""

from __future__ import annotations

import math

import numpy as np

from compare_gaussian_sextic import FAC3AU

HARTREE_J = 4.3597447222071e-18
PLANCK = 6.62607015e-34
CLIGHT_CM = 2.99792458e10
AMU_KG = 1.66053906660e-27
BOHR_ANG = 0.529177210903
INV_FACTG = PLANCK / (4.0 * math.pi**2 * CLIGHT_CM) / (AMU_KG * (1.0e-10) ** 2)
FACTG = 1.0 / INV_FACTG
HC_ATTOJ_CM = PLANCK * CLIGHT_CM * 1.0e18
FAC4AU = (HARTREE_J * 1.0e18) / (BOHR_ANG**4 * FACTG**2 * HC_ATTOJ_CM)


def raw_cubic_to_reduced_cm(phi3_raw_au: np.ndarray, target_freq_cm: np.ndarray) -> np.ndarray:
    """Convert Gaussian raw cubic force constants to reduced `cm^-1` form."""

    freq = np.abs(np.asarray(target_freq_cm, dtype=float))
    phi3_raw = np.asarray(phi3_raw_au, dtype=float)
    out = np.zeros_like(phi3_raw)
    n_modes = freq.size
    for i in range(n_modes):
        for j in range(n_modes):
            for k in range(n_modes):
                den = math.sqrt(freq[i] * freq[j] * freq[k])
                out[i, j, k] = 0.0 if den <= 1.0e-30 else phi3_raw[i, j, k] * FAC3AU / den
    return out


def raw_quartic_to_reduced_cm(phi4_raw_au: np.ndarray, target_freq_cm: np.ndarray) -> np.ndarray:
    """Convert Gaussian raw quartic force constants to reduced `cm^-1` form."""

    freq = np.abs(np.asarray(target_freq_cm, dtype=float))
    phi4_raw = np.asarray(phi4_raw_au, dtype=float)
    out = np.zeros_like(phi4_raw)
    n_modes = freq.size
    for i in range(n_modes):
        for j in range(n_modes):
            for k in range(n_modes):
                for l in range(n_modes):
                    den = math.sqrt(freq[i] * freq[j] * freq[k] * freq[l])
                    out[i, j, k, l] = 0.0 if den <= 1.0e-30 else phi4_raw[i, j, k, l] * FAC4AU / den
    return out


def gaussian_bxx_to_aliev(bxx_gaussian: np.ndarray, target_freq_cm: np.ndarray) -> np.ndarray:
    """Convert Gaussian-mode rotational derivatives to Aliev-coordinate form."""

    bxx = np.asarray(bxx_gaussian, dtype=float)
    freq = np.abs(np.asarray(target_freq_cm, dtype=float))
    if bxx.shape != freq.shape:
        raise ValueError(f"bxx/frequency shape mismatch: {bxx.shape} vs {freq.shape}")
    out = np.zeros_like(bxx)
    mask = freq > 1.0e-30
    out[mask] = bxx[mask] / np.sqrt(freq[mask])
    return out


def reduced_cubic_to_aliev_k3(phi3_reduced_cm: np.ndarray, target_freq_cm: np.ndarray) -> np.ndarray:
    """Convert Gaussian reduced cubic constants to the Aliev `k'` normalization."""

    freq = np.abs(np.asarray(target_freq_cm, dtype=float))
    phi3 = np.asarray(phi3_reduced_cm, dtype=float)
    out = np.zeros_like(phi3)
    n_modes = freq.size
    for i in range(n_modes):
        for j in range(n_modes):
            for k in range(n_modes):
                den = math.sqrt(freq[i] * freq[j] * freq[k])
                out[i, j, k] = 0.0 if den <= 1.0e-30 else phi3[i, j, k] / den
    return out


def reduced_quartic_to_aliev_k4(phi4_reduced_cm: np.ndarray, target_freq_cm: np.ndarray) -> np.ndarray:
    """Convert Gaussian reduced quartic constants to the Aliev `k'` normalization."""

    freq = np.abs(np.asarray(target_freq_cm, dtype=float))
    phi4 = np.asarray(phi4_reduced_cm, dtype=float)
    out = np.zeros_like(phi4)
    n_modes = freq.size
    for i in range(n_modes):
        for j in range(n_modes):
            for k in range(n_modes):
                for l in range(n_modes):
                    den = math.sqrt(freq[i] * freq[j] * freq[k] * freq[l])
                    out[i, j, k, l] = 0.0 if den <= 1.0e-30 else phi4[i, j, k, l] / den
    return out
