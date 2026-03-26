#!/usr/bin/env python3
"""General projected quartic-observable scaffolding.

This module is the extensible layer that should survive beyond the current
linear-molecule `Dv` prototype.  The central object is not a fitted scalar
constant, but a mode-resolved projected observable model

    O(v) = O_e + sum_i g_i * w_i(v)

where:

- `O` can be a scalar or a vector of quartic observables,
- `g_i` are explicit mode-resolved physical contributions,
- `w_i(v)` are state factors determined by the projection rule.

For linear molecules the current observable set has one component, `Dv`, with

    w_i(v) = v_i + 1/2   for parallel modes
    w_i(v) = v_i + 1     for perpendicular modes

The same machinery can later host non-linear projections onto Watson quartic
constants or any other operator basis once the general mapping is derived.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ArrayLike = np.ndarray | list[float] | tuple[float, ...]


def as_frequency_vector(frequencies_cm: ArrayLike) -> np.ndarray:
    freq = np.asarray(frequencies_cm, dtype=float).reshape(-1)
    if np.any(freq <= 0.0):
        raise ValueError("All harmonic frequencies must be strictly positive.")
    return freq


def validate_square(name: str, arr: np.ndarray, n_modes: int) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.shape != (n_modes, n_modes):
        raise ValueError(f"{name} must have shape {(n_modes, n_modes)}, got {out.shape}.")
    return out


def validate_cubic(phi3: np.ndarray, n_modes: int) -> np.ndarray:
    out = np.asarray(phi3, dtype=float)
    if out.shape != (n_modes, n_modes, n_modes):
        raise ValueError(f"cubic_force_constants must have shape {(n_modes, n_modes, n_modes)}, got {out.shape}.")
    return out


def validate_mode_kinds(mode_kinds: tuple[str, ...] | list[str], n_modes: int) -> tuple[str, ...]:
    kinds = tuple(str(item).strip().lower() for item in mode_kinds)
    if len(kinds) != n_modes:
        raise ValueError(f"mode_kinds must contain {n_modes} labels, got {len(kinds)}.")
    allowed = {"parallel", "perpendicular"}
    bad = [kind for kind in kinds if kind not in allowed]
    if bad:
        raise ValueError(f"Unsupported mode kind(s): {bad}. Allowed values are {sorted(allowed)}.")
    return kinds


def linear_state_factors(quanta: ArrayLike, mode_kinds: tuple[str, ...] | list[str]) -> np.ndarray:
    """Return the linear-molecule state weights from Aliev's Eq. (1)."""

    v = np.asarray(quanta, dtype=float).reshape(-1)
    kinds = validate_mode_kinds(mode_kinds, v.size)
    out = np.empty_like(v)
    for i, kind in enumerate(kinds):
        out[i] = v[i] + (0.5 if kind == "parallel" else 1.0)
    return out


def compute_phi_iijk(
    reference_hessian: np.ndarray,
    displaced_hessians: dict[int, tuple[np.ndarray, np.ndarray]],
    step_sizes: ArrayLike,
) -> np.ndarray:
    """Return the semi-diagonal reduced quartics `Phi(iiik)` from Hessian scans."""

    h0 = np.asarray(reference_hessian, dtype=float)
    if h0.ndim != 2 or h0.shape[0] != h0.shape[1]:
        raise ValueError(f"reference_hessian must be square, got {h0.shape}.")
    n_modes = h0.shape[0]
    steps = np.asarray(step_sizes, dtype=float).reshape(n_modes)
    if np.any(steps <= 0.0):
        raise ValueError("All step sizes must be strictly positive.")

    phi = np.zeros((n_modes, n_modes), dtype=float)
    for i in range(n_modes):
        if i not in displaced_hessians:
            raise ValueError(f"Missing Hessian scan for displaced mode {i}.")
        h_minus, h_plus = displaced_hessians[i]
        hm = validate_square(f"H_minus[{i}]", h_minus, n_modes)
        hp = validate_square(f"H_plus[{i}]", h_plus, n_modes)
        denom = steps[i] * steps[i]
        for k in range(n_modes):
            phi[i, k] = (hp[i, k] - 2.0 * h0[i, k] + hm[i, k]) / denom
    return phi


@dataclass(frozen=True)
class ObservableContribution:
    """One physical contribution to a projected quartic observable model."""

    label: str
    equilibrium: np.ndarray
    linear_terms: np.ndarray

    def values_for_state(self, quanta: ArrayLike, mode_kinds: tuple[str, ...] | list[str]) -> np.ndarray:
        weights = linear_state_factors(quanta, mode_kinds)
        return np.asarray(self.equilibrium, dtype=float) + weights @ np.asarray(self.linear_terms, dtype=float)


@dataclass(frozen=True)
class QuarticObservableModel:
    """Mode-resolved projected quartic-observable model."""

    observable_labels: tuple[str, ...]
    mode_kinds: tuple[str, ...]
    total: ObservableContribution
    contributions: tuple[ObservableContribution, ...]

    def values_for_state(self, quanta: ArrayLike) -> dict[str, float]:
        vals = self.total.values_for_state(quanta, self.mode_kinds)
        return {label: float(vals[i]) for i, label in enumerate(self.observable_labels)}

    def contribution_values_for_state(self, quanta: ArrayLike) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for item in self.contributions:
            vals = item.values_for_state(quanta, self.mode_kinds)
            out[item.label] = {label: float(vals[i]) for i, label in enumerate(self.observable_labels)}
        return out

    def project_scalar(self, label: str, quanta: ArrayLike) -> float:
        vals = self.values_for_state(quanta)
        if label not in vals:
            raise KeyError(f"Observable {label!r} is not present. Available labels: {sorted(vals)}.")
        return vals[label]


@dataclass(frozen=True)
class ReducedQuarticTermComponents:
    """Reduced semi-diagonal building blocks of the projected quartic model."""

    coriolis: np.ndarray
    quartic_direct: np.ndarray
    cubic_dressing: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.coriolis + self.quartic_direct + self.cubic_dressing


def compute_reduced_quartic_term_components(
    frequencies_cm: ArrayLike,
    coriolis_matrix: np.ndarray,
    cubic_force_constants: np.ndarray,
    phi_iijk: np.ndarray,
    *,
    resonance_floor_cm: float = 5.0,
    partner_weight: float = 0.5,
) -> ReducedQuarticTermComponents:
    """Return reduced semi-diagonal components of the quartic observable model."""

    freq = as_frequency_vector(frequencies_cm)
    cor = validate_square("coriolis_matrix", coriolis_matrix, freq.size)
    phi3 = validate_cubic(cubic_force_constants, freq.size)
    phi4 = validate_square("phi_iijk", phi_iijk, freq.size)
    floor = float(resonance_floor_cm)
    if floor <= 0.0:
        raise ValueError("resonance_floor_cm must be positive.")

    cor_terms = np.zeros_like(freq)
    direct_terms = np.zeros_like(freq)
    cubic_terms = np.zeros_like(freq)

    for i in range(freq.size):
        for j in range(i + 1, freq.size):
            cij = 0.5 * (float(cor[i, j]) - float(cor[j, i]))
            if abs(cij) <= 1.0e-16:
                cij = float(cor[i, j])
            if abs(cij) <= 1.0e-16:
                continue
            strength = (cij * cij) / (freq[i] + freq[j])
            cor_terms[i] -= 0.5 * strength / freq[i]
            cor_terms[j] -= 0.5 * strength / freq[j]

    for i in range(freq.size):
        w_i = freq[i]
        phi_iii = float(phi3[i, i, i])
        for k in range(freq.size):
            w_k = freq[k]
            denom_cubic = max(2.0 * w_i + w_k, floor)
            denom_proj = max(w_i * (w_i + w_k), floor)

            direct_strength = -float(phi4[i, k]) / (8.0 * denom_proj)
            cubic_strength = +(
                phi_iii * float(phi3[i, i, k]) / denom_cubic
            ) / (8.0 * denom_proj)

            direct_terms[i] += direct_strength
            cubic_terms[i] += cubic_strength
            if k != i:
                direct_terms[k] += float(partner_weight) * direct_strength
                cubic_terms[k] += float(partner_weight) * cubic_strength

    return ReducedQuarticTermComponents(
        coriolis=cor_terms,
        quartic_direct=direct_terms,
        cubic_dressing=cubic_terms,
    )


def build_linear_dv_model(
    frequencies_cm: ArrayLike,
    mode_kinds: tuple[str, ...] | list[str],
    coriolis_matrix: np.ndarray,
    cubic_force_constants: np.ndarray,
    phi_iijk: np.ndarray,
    *,
    dj_equilibrium: float = 0.0,
    resonance_floor_cm: float = 5.0,
    partner_weight: float = 0.5,
) -> tuple[QuarticObservableModel, ReducedQuarticTermComponents]:
    """Build the linear-molecule `Dv` model as one projection of the general layer."""

    freq = as_frequency_vector(frequencies_cm)
    kinds = validate_mode_kinds(mode_kinds, freq.size)
    components = compute_reduced_quartic_term_components(
        freq,
        coriolis_matrix,
        cubic_force_constants,
        phi_iijk,
        resonance_floor_cm=resonance_floor_cm,
        partner_weight=partner_weight,
    )

    eq = np.array([float(dj_equilibrium)], dtype=float)
    cor = ObservableContribution(
        label="coriolis",
        equilibrium=np.zeros(1, dtype=float),
        linear_terms=components.coriolis[:, None],
    )
    quart = ObservableContribution(
        label="quartic_direct",
        equilibrium=np.zeros(1, dtype=float),
        linear_terms=components.quartic_direct[:, None],
    )
    cubic = ObservableContribution(
        label="cubic_dressing",
        equilibrium=np.zeros(1, dtype=float),
        linear_terms=components.cubic_dressing[:, None],
    )
    total = ObservableContribution(
        label="total",
        equilibrium=eq,
        linear_terms=components.total[:, None],
    )
    model = QuarticObservableModel(
        observable_labels=("Dv",),
        mode_kinds=kinds,
        total=total,
        contributions=(cor, quart, cubic),
    )
    return model, components
