#!/usr/bin/env python3
"""Linear-molecule `Dv` wrapper over the general quartic observable backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quartic_observable_model import (
    QuarticObservableModel,
    ReducedQuarticTermComponents,
    build_linear_dv_model,
    compute_reduced_quartic_term_components,
    compute_phi_iijk,
    linear_state_factors,
    validate_mode_kinds,
)


ArrayLike = np.ndarray | list[float] | tuple[float, ...]


@dataclass(frozen=True)
class DvPerturbativeInputs:
    """Reduced perturbative ingredients for the linear-molecule `Dv` model."""

    frequencies_cm: np.ndarray
    mode_kinds: tuple[str, ...]
    coriolis_matrix: np.ndarray
    cubic_force_constants: np.ndarray
    phi_iijk: np.ndarray
    dj_equilibrium: float = 0.0


@dataclass(frozen=True)
class DvContribution:
    """One physical branch in the linear `Dv` expansion."""

    label: str
    beta: np.ndarray


@dataclass(frozen=True)
class DvResult:
    """Linear `Dv` response with branch-resolved decomposition."""

    total: DvContribution
    coriolis: DvContribution
    anharmonic: DvContribution
    reduced_terms: ReducedQuarticTermComponents
    observable_model: QuarticObservableModel
    dj_equilibrium: float
    mode_kinds: tuple[str, ...]

    def value_for_state(self, quanta: ArrayLike) -> float:
        return self.observable_model.project_scalar("Dv", quanta)


def compute_reduced_beta_terms(
    frequencies_cm: ArrayLike,
    coriolis_matrix: np.ndarray,
    cubic_force_constants: np.ndarray,
    phi_iijk: np.ndarray,
    *,
    resonance_floor_cm: float = 5.0,
    partner_weight: float = 0.5,
) -> ReducedQuarticTermComponents:
    """Compatibility wrapper exposing the reduced linear `Dv` pieces."""

    return compute_reduced_quartic_term_components(
        frequencies_cm,
        coriolis_matrix,
        cubic_force_constants,
        phi_iijk,
        resonance_floor_cm=resonance_floor_cm,
        partner_weight=partner_weight,
    )


def compute_coriolis_contribution(
    frequencies_cm: ArrayLike,
    coriolis_matrix: np.ndarray,
) -> DvContribution:
    n_modes = len(np.asarray(frequencies_cm).reshape(-1))
    components = compute_reduced_quartic_term_components(
        frequencies_cm,
        coriolis_matrix,
        np.zeros((n_modes, n_modes, n_modes), dtype=float),
        np.zeros((n_modes, n_modes), dtype=float),
    )
    return DvContribution(label="coriolis", beta=-components.coriolis)


def compute_anharmonic_contribution(
    frequencies_cm: ArrayLike,
    cubic_force_constants: np.ndarray,
    phi_iijk: np.ndarray,
    *,
    resonance_floor_cm: float = 5.0,
    partner_weight: float = 0.5,
) -> DvContribution:
    n_modes = len(np.asarray(frequencies_cm).reshape(-1))
    components = compute_reduced_quartic_term_components(
        frequencies_cm,
        np.zeros((n_modes, n_modes), dtype=float),
        cubic_force_constants,
        phi_iijk,
        resonance_floor_cm=resonance_floor_cm,
        partner_weight=partner_weight,
    )
    return DvContribution(label="anharmonic", beta=-(components.quartic_direct + components.cubic_dressing))


def split_beta_by_mode_kind(beta: ArrayLike, mode_kinds: tuple[str, ...] | list[str]) -> dict[str, np.ndarray]:
    """Split a `beta` vector into parallel and perpendicular blocks."""

    arr = np.asarray(beta, dtype=float).reshape(-1)
    kinds = validate_mode_kinds(mode_kinds, arr.size)
    parallel = np.zeros_like(arr)
    perpendicular = np.zeros_like(arr)
    for i, kind in enumerate(kinds):
        if kind == "parallel":
            parallel[i] = arr[i]
        elif kind == "perpendicular":
            perpendicular[i] = arr[i]
        else:
            raise ValueError(f"Unsupported mode kind {kind!r}.")
    return {"parallel": parallel, "perpendicular": perpendicular}


def assemble_Dv(
    inputs: DvPerturbativeInputs,
    *,
    resonance_floor_cm: float = 5.0,
    partner_weight: float = 0.5,
) -> DvResult:
    """Assemble the linear `Dv` model from the general quartic observable layer."""

    model, components = build_linear_dv_model(
        inputs.frequencies_cm,
        inputs.mode_kinds,
        inputs.coriolis_matrix,
        inputs.cubic_force_constants,
        inputs.phi_iijk,
        dj_equilibrium=inputs.dj_equilibrium,
        resonance_floor_cm=resonance_floor_cm,
        partner_weight=partner_weight,
    )

    cor = DvContribution(label="coriolis", beta=-components.coriolis)
    anh = DvContribution(label="anharmonic", beta=-(components.quartic_direct + components.cubic_dressing))
    total = DvContribution(label="total", beta=-(components.total))
    return DvResult(
        total=total,
        coriolis=cor,
        anharmonic=anh,
        reduced_terms=ReducedQuarticTermComponents(
            coriolis=-components.coriolis,
            quartic_direct=-components.quartic_direct,
            cubic_dressing=-components.cubic_dressing,
        ),
        observable_model=model,
        dj_equilibrium=float(inputs.dj_equilibrium),
        mode_kinds=tuple(inputs.mode_kinds),
    )


def compute_Dv_for_state(
    inputs: DvPerturbativeInputs,
    quanta: ArrayLike,
    *,
    resonance_floor_cm: float = 5.0,
    partner_weight: float = 0.5,
) -> DvResult:
    """Convenience wrapper returning the assembled decomposition."""

    result = assemble_Dv(
        inputs,
        resonance_floor_cm=resonance_floor_cm,
        partner_weight=partner_weight,
    )
    _ = result.value_for_state(quanta)
    return result
