#!/usr/bin/env python3
"""Utilities for H30H30 resonance diagnostics and regularization.

This module does not pretend that the current cubic-cubic implementation is
complete. Instead, it provides a unit-explicit diagnostic layer that can be
reused once the full denominator families have been restored.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class ResonanceTerm:
    """Single cubic-cubic term prepared for resonance diagnostics.

    Parameters
    ----------
    family:
        Denominator family label.
    component:
        Quartic component receiving the term, for example ``tau_xxxx``.
    modes:
        Zero-based mode tuple feeding the term.
    numerator_au:
        Raw numerator in the internal atomic-unit convention.
    denominator_cm:
        Denominator converted to cm^-1.
    wilson_factor_cm_per_au:
        Conversion factor used to translate the numerator to cm^-1.
    diagonal_modes:
        Whether the term is diagonal in the modal pair sense.
    """

    family: str
    component: str
    modes: tuple[int, ...]
    numerator_au: float
    denominator_cm: float
    wilson_factor_cm_per_au: float = 1.0
    diagonal_modes: bool = False

    @property
    def coupling_cm(self) -> float:
        return self.wilson_factor_cm_per_au * self.numerator_au

    @property
    def martin_ratio(self) -> float:
        return martin_ratio(self.coupling_cm, self.denominator_cm)

    @property
    def perturbative_value_cm(self) -> float:
        return safe_divide(self.coupling_cm, self.denominator_cm)


@dataclass(frozen=True)
class DenominatorFactorDiagnostic:
    """Small diagnostic object for scaffold denominator factors.

    ``martin_ready`` is intentionally conservative: it is true only when the
    scaffold exposes at least one signed linear frequency combination that can
    plausibly become small and therefore feed a Martin test or 2x2 switch.
    """

    scaffold: str
    modes: tuple[int, ...]
    factors_cm: tuple[float, ...]
    martin_ready: bool

    @property
    def smallest_abs_factor_cm(self) -> float:
        if not self.factors_cm:
            return math.inf
        return min(abs(x) for x in self.factors_cm)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return math.copysign(math.inf, numerator if numerator != 0.0 else 1.0)
    return numerator / denominator


def martin_ratio(coupling_cm: float, denominator_cm: float) -> float:
    """Return the Martin metric |V| / |delta| in a common cm^-1 convention."""
    if denominator_cm == 0.0:
        return math.inf if coupling_cm != 0.0 else 0.0
    return abs(coupling_cm) / abs(denominator_cm)


def smooth_switch(metric: float, center: float = 1.0, width: float = 0.05, method: str = "tanh") -> float:
    """Return a smooth resonance switch in [0, 1]."""
    if width <= 0.0:
        raise ValueError("width must be positive.")
    x = (metric - center) / width
    method_norm = method.strip().lower()
    if method_norm == "tanh":
        return 0.5 * (1.0 + math.tanh(x))
    if method_norm in {"erf", "errfunc", "error", "error-function"}:
        return 0.5 * (1.0 + math.erf(x))
    raise ValueError(f"Unknown switch method '{method}'. Use 'tanh' or 'erf'.")


def regularized_two_level_term(
    coupling_cm: float,
    denominator_cm: float,
    *,
    metric_center: float = 1.0,
    metric_width: float = 0.05,
    level_shift_cm: float = 1.0,
    method: str = "tanh",
) -> float:
    """Blend perturbative and 2x2-polyad regularized terms.

    The CI-like branch follows the resolvent form used in the manuscript draft,
    while the perturbative branch uses a fixed level shift.
    """
    if level_shift_cm <= 0.0:
        raise ValueError("level_shift_cm must be positive.")

    sign = 1.0 if denominator_cm >= 0.0 else -1.0
    metric = martin_ratio(coupling_cm, denominator_cm)
    switch = smooth_switch(metric, center=metric_center, width=metric_width, method=method)

    ci_denom = denominator_cm + sign * math.sqrt(denominator_cm * denominator_cm + 4.0 * coupling_cm * coupling_cm)
    shifted_denom = denominator_cm + sign * level_shift_cm

    ci_like = safe_divide(coupling_cm, ci_denom)
    shifted = safe_divide(coupling_cm, shifted_denom)
    return switch * ci_like + (1.0 - switch) * shifted


def enumerate_current_h30h30_terms(
    mu1: np.ndarray,
    phi3: np.ndarray,
    omega_cm: Iterable[float],
    *,
    wilson_factor_cm_per_au: float = 1.0,
) -> list[ResonanceTerm]:
    """Enumerate the current simplified H30H30 terms used by the code.

    This is intentionally restricted to the currently implemented positive-sum
    kernel family ``omega_i + omega_j + 1``. It is useful for diagnosing the
    present code path and for checking units and diagonal/modal dominance, but
    it is *not* the complete BCH denominator restoration.
    """
    mu1 = np.asarray(mu1, dtype=float)
    phi3 = np.asarray(phi3, dtype=float)
    omega_cm = np.asarray(tuple(float(x) for x in omega_cm), dtype=float)

    if mu1.ndim != 3 or mu1.shape[0] != 3 or mu1.shape[1] != 3:
        raise ValueError("mu1 must have shape (3, 3, n_modes).")
    if phi3.ndim != 3:
        raise ValueError("phi3 must have shape (n_modes, n_modes, n_modes).")
    n_modes = mu1.shape[2]
    if phi3.shape != (n_modes, n_modes, n_modes):
        raise ValueError("phi3 shape must match the number of modes in mu1.")
    if omega_cm.shape != (n_modes,):
        raise ValueError("omega_cm length must match the number of modes in mu1.")

    components = (
        ("tau_xxxx", 0),
        ("tau_yyyy", 1),
        ("tau_zzzz", 2),
    )
    out: list[ResonanceTerm] = []
    for component, axis in components:
        for i in range(n_modes):
            for j in range(n_modes):
                numerator_au = float(phi3[i, j, j] * mu1[axis, axis, i] * mu1[axis, axis, j])
                denominator_cm = float(omega_cm[i] + omega_cm[j] + 1.0)
                out.append(
                    ResonanceTerm(
                        family="sum_plus_one_current",
                        component=component,
                        modes=(i, j, j),
                        numerator_au=numerator_au,
                        denominator_cm=denominator_cm,
                        wilson_factor_cm_per_au=wilson_factor_cm_per_au,
                        diagonal_modes=(i == j),
                    )
                )
    return out


def summarize_terms_by_metric(terms: Iterable[ResonanceTerm], top_n: int = 10) -> list[ResonanceTerm]:
    return sorted(terms, key=lambda term: (term.martin_ratio, abs(term.coupling_cm)), reverse=True)[:top_n]


def denominator_signature(expr: sp.Expr) -> sp.Expr:
    """Return a factored denominator signature for a symbolic term or sum."""
    _num, den = sp.fraction(sp.together(sp.simplify(expr)))
    return sp.factor(den)


def extract_denominator_families(exprs: dict[str, sp.Expr]) -> dict[sp.Expr, list[str]]:
    """Group tensor components by their symbolic denominator signature."""
    families: dict[sp.Expr, list[str]] = {}
    for name, expr in exprs.items():
        sig = denominator_signature(expr)
        families.setdefault(sig, []).append(name)
    return families


def current_h30h30_factor_diagnostics(omega_cm: Iterable[float]) -> list[DenominatorFactorDiagnostic]:
    """Return factor-level diagnostics for the current scaffold-first H30H30 model.

    This is intentionally not a Martin metric on the raw scaffold denominators.
    Instead it answers the more basic question: does the current reconstructed
    scaffold expose any signed linear combination that could meaningfully enter
    a Martin test or a perturbative->variational 2x2 switch?

    With the present scaffold set the answer is essentially "no":
    only positive powers of ``omega_i``, ``omega_j``, and positive pair sums
    ``omega_i + omega_j`` appear. Difference denominators such as
    ``2*omega_i - omega_j`` are not yet restored.
    """
    omega_cm = tuple(float(x) for x in omega_cm)
    out: list[DenominatorFactorDiagnostic] = []
    n_modes = len(omega_cm)
    for i in range(n_modes):
        out.append(
            DenominatorFactorDiagnostic(
                scaffold="diag_1_iii_iii",
                modes=(i,),
                factors_cm=(omega_cm[i],),
                martin_ready=False,
            )
        )
    for i in range(n_modes):
        for j in range(n_modes):
            if i == j:
                continue
            out.append(
                DenominatorFactorDiagnostic(
                    scaffold="diag_1_iii_iij_0",
                    modes=(i, j),
                    factors_cm=(omega_cm[i], omega_cm[j]),
                    martin_ready=False,
                )
            )
            out.append(
                DenominatorFactorDiagnostic(
                    scaffold="diag_0_iii_iij_1_candidate",
                    modes=(i, j),
                    factors_cm=(omega_cm[i], omega_cm[j], omega_cm[i] + omega_cm[j]),
                    martin_ready=False,
                )
            )
            out.append(
                DenominatorFactorDiagnostic(
                    scaffold="diag_0_iii_iij_2_resonance_candidate",
                    modes=(i, j),
                    factors_cm=(omega_cm[i], omega_cm[j], omega_cm[i] + omega_cm[j], 2.0 * omega_cm[i] - omega_cm[j], 2.0 * omega_cm[i] + omega_cm[j]),
                    martin_ready=True,
                )
            )
            out.append(
                DenominatorFactorDiagnostic(
                    scaffold="diag_0_iij_iij_1_candidate",
                    modes=(i, j),
                    factors_cm=(omega_cm[i], omega_cm[j]),
                    martin_ready=False,
                )
            )
            out.append(
                DenominatorFactorDiagnostic(
                    scaffold="diag_0_iij_iij_2_candidate",
                    modes=(i, j),
                    factors_cm=(omega_cm[i], omega_cm[j], omega_cm[i] + omega_cm[j]),
                    martin_ready=False,
                )
            )
    return out
