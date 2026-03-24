#!/usr/bin/env python3
"""Helpers for direct/indirect anharmonic partitioning.

This module intentionally starts small. The immediate need is to provide the
algebraic bridge from the total anharmonic X matrix to the indirect dressing
matrix once the direct VPT2 contribution Y has been constructed elsewhere.

For the current project we work in Cartesian normal coordinates. In this
specialization the metric derivatives disappear, so the general reduced-VPT2
objects collapse to ordinary force constants:

- ``eta -> phi4``
- ``sigma = rho -> phi3``

The direct matrix ``Y`` is therefore built only from the quartic force field,
the pair-local cubic couplings, and the direct Coriolis block when available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


_FUNDAMENTAL_STATE_RE = re.compile(r"^\|?(\d+)\(1\)\>?$")


@dataclass
class ThreeClassModeDiagnostic:
    mode_index: int
    mode_label: str
    assigned_class: str
    y_diag_cm: float
    z_diag_cm: float
    x_diag_cm: float
    self_cubic_shift_cm: float
    quartic_diag_term_cm: float
    direct_fraction: float
    active_resonance_count: int
    low_overlap: float | None
    resonance_partner_modes: tuple[int, ...]
    omega_dvpt2_cm: float | None
    omega_gvpt2_cm: float | None
    qp_shift_cm: float | None
    reasons: tuple[str, ...]


@dataclass
class ThreeClassAnalysis:
    y_direct_cm: np.ndarray
    z_indirect_cm: np.ndarray
    diagnostics: tuple[ThreeClassModeDiagnostic, ...]
    unreliable_semidiagonal_pairs: tuple[tuple[int, int], ...]


@dataclass
class AnharmonicFilterPlan:
    warning_modes: tuple[int, ...]
    warning_pairs: tuple[tuple[int, int], ...]
    class2_modes: tuple[int, ...]
    class3_modes: tuple[int, ...]
    disable_semidiagonal_pairs: tuple[tuple[int, int], ...]
    reasons_by_mode: dict[int, tuple[str, ...]]


def effective_frequencies_from_three_class_report(harmonic_freq_cm: np.ndarray, report: ThreeClassAnalysis) -> np.ndarray:
    """Return the harmonic list with Class-II modes replaced by ``omega_GVPT2``.

    This is the current operational definition of ``omega_eff`` for the alpha
    model: only modes explicitly tagged as diagonal-problematic (Class II) are
    renormalized, while all other modes keep their harmonic reference.
    """

    freq = np.asarray(harmonic_freq_cm, dtype=float).copy()
    n_modes = freq.size
    for item in report.diagnostics:
        if item.assigned_class != "II":
            continue
        if item.omega_gvpt2_cm is None:
            continue
        idx = int(item.mode_index) - 1
        if 0 <= idx < n_modes and float(item.omega_gvpt2_cm) > 0.0:
            freq[idx] = float(item.omega_gvpt2_cm)
    return freq


def _fundamental_mode_from_state_label(label: str) -> int | None:
    text = label.strip()
    m = _FUNDAMENTAL_STATE_RE.match(text)
    if m:
        return int(m.group(1))
    text = text.strip("|")
    m = _FUNDAMENTAL_STATE_RE.match(text)
    if m:
        return int(m.group(1))
    return None


def _validate_cartesian_force_data(freq_cm: np.ndarray, phi3_cm: np.ndarray, phi4_cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freq = np.asarray(freq_cm, dtype=float)
    phi3 = np.asarray(phi3_cm, dtype=float)
    phi4 = np.asarray(phi4_cm, dtype=float)
    n_modes = freq.size
    if phi3.shape != (n_modes, n_modes, n_modes):
        raise ValueError(f"Unexpected cubic tensor shape: {phi3.shape}, expected {(n_modes, n_modes, n_modes)}.")
    if phi4.shape != (n_modes, n_modes, n_modes, n_modes):
        raise ValueError(
            f"Unexpected quartic tensor shape: {phi4.shape}, expected {(n_modes, n_modes, n_modes, n_modes)}."
        )
    return freq, phi3, phi4


def direct_y_matrix_cartesian(
    freq_cm: np.ndarray,
    phi3_reduced_cm: np.ndarray,
    phi4_reduced_cm: np.ndarray,
    *,
    coriolis_cm: np.ndarray | None = None,
    resonance_tol_cm: float = 1.0,
) -> np.ndarray:
    """Build the direct Cartesian ``Y`` matrix from reduced force constants.

    This implements the Cartesian reduction of the general formulas used in the
    three-class reduced-dimensional model:

    - diagonal terms use only ``phi_iiii`` and the self-cubic ``phi_iii``
    - off-diagonal terms use only the pair-local couplings
      ``phi_iij``/``phi_jji`` plus the direct Coriolis block

    When a pair is quasi-degenerate in the denominator ``omega_i^2-omega_j^2``,
    the pair-local cubic correction is set to zero and only the quartic and
    Coriolis direct terms are retained. This is deliberate: those cases should
    be handled later by the explicit resonance/GVPT2 logic, not by a naive
    perturbative denominator.
    """

    freq, phi3, phi4 = _validate_cartesian_force_data(freq_cm, phi3_reduced_cm, phi4_reduced_cm)
    n_modes = freq.size
    y = np.zeros((n_modes, n_modes), dtype=float)
    cor = np.zeros_like(y) if coriolis_cm is None else np.asarray(coriolis_cm, dtype=float)
    if cor.shape != (n_modes, n_modes):
        raise ValueError(f"Unexpected Coriolis X shape: {cor.shape}, expected {(n_modes, n_modes)}.")

    for i in range(n_modes):
        wi = float(freq[i])
        if wi <= 0.0:
            continue
        y[i, i] = float(phi4[i, i, i, i]) / 16.0 - 5.0 * float(phi3[i, i, i]) ** 2 / (48.0 * wi)

    for i in range(n_modes):
        wi = float(freq[i])
        if wi <= 0.0:
            continue
        for j in range(i + 1, n_modes):
            wj = float(freq[j])
            if wj <= 0.0:
                continue
            cubic_direct = 0.0
            denom = wi * wi - wj * wj
            if abs(denom) > resonance_tol_cm * max(wi, wj):
                piii = float(phi3[i, i, i])
                pjjj = float(phi3[j, j, j])
                piij = float(phi3[i, i, j])
                pjji = float(phi3[j, j, i])
                cubic_direct = (
                    -(piii * pjji / (4.0 * wi) + pjjj * piij / (4.0 * wj))
                    + 0.5
                    * (
                        piij * piij * wj / (wi * (wj * wj - wi * wi))
                        + pjji * pjji * wi / (wj * (wi * wi - wj * wj))
                    )
                )
            value = float(phi4[i, i, j, j]) / 4.0 + cubic_direct + float(cor[i, j])
            y[i, j] = value
            y[j, i] = value
    return y


def z_matrix_from_x_y(x_total_cm: np.ndarray, y_direct_cm: np.ndarray) -> np.ndarray:
    """Return the indirect anharmonic matrix ``Z = X - Y`` in cm^-1."""
    x = np.asarray(x_total_cm, dtype=float)
    y = np.asarray(y_direct_cm, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"X and Y must have the same shape, got {x.shape} and {y.shape}.")
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"Expected square X/Y matrices, got {x.shape}.")
    return x - y


def classify_three_classes_cartesian(
    force_data,
    analysis,
    *,
    low_overlap_threshold: float = 0.85,
    qp_shift_warning_cm: float = 15.0,
    qp_shift_class3_cm: float = 40.0,
    class3_low_overlap_threshold: float = 0.55,
    class3_min_active_resonances: int = 2,
    class2_direct_fraction: float = 0.6,
    class2_self_cubic_fraction: float = 0.5,
    class2_self_cubic_min_cm: float = 10.0,
    class2_max_frequency_cm: float = 1200.0,
) -> ThreeClassAnalysis:
    """Classify modes into the current I/II/III working scheme.

    The first diagnostic is the quasiparticle shift
    ``omega_GVPT2 - omega_DVPT2`` for the fundamental excitation energy
    ``E_fund - E_zpe``. Active resonances and low variational overlaps remain
    warning signals, but they are interpreted in the light of that shift.

    Class II is a diagonal-only flag used when the direct diagonal dressing is
    large and dominated by the self-cubic correction, but no explicit Class III
    evidence is present.
    """

    y = direct_y_matrix_cartesian(
        force_data.frequencies_cm,
        force_data.phi3_reduced_cm,
        force_data.phi4_reduced_cm,
        coriolis_cm=analysis.x_coriolis_cm,
    )
    z = z_matrix_from_x_y(analysis.x_total_cm, y)
    phi3 = np.asarray(force_data.phi3_reduced_cm, dtype=float)
    phi4 = np.asarray(force_data.phi4_reduced_cm, dtype=float)
    freq = np.asarray(force_data.frequencies_cm, dtype=float)
    x_total = np.asarray(analysis.x_total_cm, dtype=float)
    n_modes = freq.size

    resonance_partners: dict[int, set[int]] = {i + 1: set() for i in range(n_modes)}
    resonance_counts: dict[int, int] = {i + 1: 0 for i in range(n_modes)}
    for entry in analysis.resonances:
        if str(entry.status).lower() != "active":
            continue
        lhs = tuple(int(x) for x in entry.lhs_modes)
        rhs = tuple(int(x) for x in entry.rhs_modes)
        if len(lhs) == 1:
            mode = lhs[0]
            resonance_counts[mode] += 1
            resonance_partners[mode].update(rhs)
        if len(rhs) == 1:
            mode = rhs[0]
            resonance_counts[mode] += 1
            resonance_partners[mode].update(lhs)

    low_overlap_map: dict[int, float] = {}
    for item in analysis.variational_overlaps:
        mode = _fundamental_mode_from_state_label(item.dvpt2_state)
        if mode is None or mode < 1 or mode > n_modes:
            continue
        overlap = float(item.overlap)
        prev = low_overlap_map.get(mode)
        if prev is None or overlap < prev:
            low_overlap_map[mode] = overlap

    omega_gvpt2_map = {int(item.mode_index): float(item.anharmonic_cm) for item in getattr(analysis, "fundamental_bands", ())}
    omega_dvpt2_map = {int(item.mode_index): float(item.anharmonic_cm) for item in getattr(analysis, "fundamental_bands", ())}
    for item in analysis.variational_energies:
        labels = [lab for lab in str(item.dvpt2_state).split(";") if lab]
        if len(labels) != 1:
            continue
        mode = _fundamental_mode_from_state_label(labels[0])
        if mode is None or mode < 1 or mode > n_modes:
            continue
        omega_dvpt2_map[mode] = float(item.deperturbed_energy_cm)
        omega_gvpt2_map[mode] = float(item.after_diag_energy_cm)

    diagnostics: list[ThreeClassModeDiagnostic] = []
    unreliable_pairs: set[tuple[int, int]] = set()
    for i in range(n_modes):
        mode = i + 1
        wi = float(freq[i])
        y_diag = float(y[i, i])
        z_diag = float(z[i, i])
        x_diag = float(x_total[i, i])
        quartic_diag = float(phi4[i, i, i, i]) / 16.0
        self_cubic = 0.0 if wi <= 0.0 else -5.0 * float(phi3[i, i, i]) ** 2 / (48.0 * wi)
        direct_fraction = abs(y_diag) / max(abs(x_diag), abs(y_diag), 1.0e-12)
        low_overlap = low_overlap_map.get(mode)
        omega_dvpt2 = omega_dvpt2_map.get(mode)
        omega_gvpt2 = omega_gvpt2_map.get(mode)
        qp_shift = None if omega_dvpt2 is None or omega_gvpt2 is None else float(omega_gvpt2 - omega_dvpt2)
        reasons: list[str] = []

        assigned = "I"
        partners = tuple(sorted(int(x) for x in resonance_partners[mode] if 1 <= int(x) <= n_modes))
        has_resonance_flag = resonance_counts[mode] > 0
        has_overlap_flag = low_overlap is not None and low_overlap < low_overlap_threshold
        has_qp_warning = qp_shift is not None and abs(qp_shift) >= qp_shift_warning_cm
        if has_resonance_flag:
            reasons.append("active_fundamental_resonance_flag")
        if has_overlap_flag:
            reasons.append("variational_overlap_warning")
        if has_qp_warning:
            reasons.append("large_qp_shift_warning")
        if (
            has_qp_warning
            and qp_shift is not None
            and abs(qp_shift) >= qp_shift_class3_cm
            and has_resonance_flag
            and low_overlap is not None
            and low_overlap < class3_low_overlap_threshold
            and resonance_counts[mode] >= class3_min_active_resonances
        ):
            assigned = "III"
            reasons.append("strong_resonant_mixing")
        if assigned != "III":
            self_cubic_fraction = abs(self_cubic) / max(abs(y_diag), 1.0e-12)
            if (
                not has_resonance_flag
                and not has_overlap_flag
                and wi <= class2_max_frequency_cm
                and direct_fraction >= class2_direct_fraction
                and self_cubic_fraction >= class2_self_cubic_fraction
                and abs(self_cubic) >= class2_self_cubic_min_cm
            ):
                assigned = "II"
                reasons.append("large_direct_diagonal_self_cubic")
        if not reasons:
            reasons.append("perturbative_default")

        if assigned == "III":
            for partner in partners:
                a = min(mode, partner)
                b = max(mode, partner)
                unreliable_pairs.add((a, b))

        diagnostics.append(
            ThreeClassModeDiagnostic(
                mode_index=mode,
                mode_label=f"mode_{mode}",
                assigned_class=assigned,
                y_diag_cm=y_diag,
                z_diag_cm=z_diag,
                x_diag_cm=x_diag,
                self_cubic_shift_cm=self_cubic,
                quartic_diag_term_cm=quartic_diag,
                direct_fraction=direct_fraction,
                active_resonance_count=int(resonance_counts[mode]),
                low_overlap=low_overlap,
                resonance_partner_modes=partners,
                omega_dvpt2_cm=omega_dvpt2,
                omega_gvpt2_cm=omega_gvpt2,
                qp_shift_cm=qp_shift,
                reasons=tuple(reasons),
            )
        )

    return ThreeClassAnalysis(
        y_direct_cm=y,
        z_indirect_cm=z,
        diagnostics=tuple(diagnostics),
        unreliable_semidiagonal_pairs=tuple(sorted(unreliable_pairs)),
    )


def build_anharmonic_filter_plan(report: ThreeClassAnalysis) -> AnharmonicFilterPlan:
    """Separate diagnostic warnings from actual filtering decisions.

    Warnings collect any mode carrying resonance/mixing flags. Filtering remains
    deliberately narrower and follows only explicit Class III assignments.
    """

    warning_modes: list[int] = []
    warning_pairs: set[tuple[int, int]] = set()
    class2_modes: list[int] = []
    class3_modes: list[int] = []
    reasons_by_mode: dict[int, tuple[str, ...]] = {}

    for item in report.diagnostics:
        reasons_by_mode[item.mode_index] = item.reasons
        is_warning = any(
            reason in {
                "active_fundamental_resonance_flag",
                "variational_overlap_warning",
                "large_qp_shift_warning",
                "strong_resonant_mixing",
            }
            for reason in item.reasons
        )
        if is_warning:
            warning_modes.append(item.mode_index)
            for partner in item.resonance_partner_modes:
                a = min(item.mode_index, partner)
                b = max(item.mode_index, partner)
                warning_pairs.add((a, b))
        if item.assigned_class == "II":
            class2_modes.append(item.mode_index)
        elif item.assigned_class == "III":
            class3_modes.append(item.mode_index)

    return AnharmonicFilterPlan(
        warning_modes=tuple(sorted(warning_modes)),
        warning_pairs=tuple(sorted(warning_pairs)),
        class2_modes=tuple(sorted(class2_modes)),
        class3_modes=tuple(sorted(class3_modes)),
        disable_semidiagonal_pairs=tuple(sorted(report.unreliable_semidiagonal_pairs)),
        reasons_by_mode=reasons_by_mode,
    )


def filter_semidiagonal_phi3(phi3_reduced_cm: np.ndarray, disabled_pairs: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> np.ndarray:
    """Return a copy with selected semi-diagonal ``phi_iij`` / ``phi_jji`` pairs zeroed.

    Pair indices are 1-based mode labels, matching the Gaussian/VPT notation
    used throughout the diagnostics layer.
    """

    phi3 = np.asarray(phi3_reduced_cm, dtype=float).copy()
    if phi3.ndim != 3 or phi3.shape[0] != phi3.shape[1] or phi3.shape[0] != phi3.shape[2]:
        raise ValueError(f"Expected a cubic tensor, got shape {phi3.shape}.")
    n_modes = phi3.shape[0]
    for pair in disabled_pairs:
        if len(pair) != 2:
            raise ValueError(f"Invalid pair specification: {pair}")
        i, j = int(pair[0]), int(pair[1])
        if not (1 <= i <= n_modes and 1 <= j <= n_modes):
            raise ValueError(f"Pair {pair} outside mode range 1..{n_modes}.")
        if i == j:
            continue
        ia = i - 1
        ja = j - 1
        phi3[ia, ia, ja] = 0.0
        phi3[ia, ja, ia] = 0.0
        phi3[ja, ia, ia] = 0.0
        phi3[ja, ja, ia] = 0.0
        phi3[ja, ia, ja] = 0.0
        phi3[ia, ja, ja] = 0.0
    return phi3
