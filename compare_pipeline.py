#!/usr/bin/env python3
"""Comparison utilities for linear-Aliev C2H2 benchmarks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ExperimentalQuantities:
    modes: tuple[str, ...]
    beta_exp: np.ndarray
    dH_exp: np.ndarray


@dataclass(frozen=True)
class ComparisonTable:
    modes: tuple[str, ...]
    calc: np.ndarray
    exp: np.ndarray
    ratio: np.ndarray
    sign_match: np.ndarray
    ranking: tuple[str, ...]


@dataclass(frozen=True)
class C2H2AlignedResults:
    modes: tuple[str, ...]
    beta_calc: np.ndarray
    L_mode_calc: np.ndarray | None
    L_total_calc: float | None


def build_exp_quantities(exp_data: dict[str, Any]) -> ExperimentalQuantities:
    """Build experimental vibrational increments from working tabulated data."""

    D0 = float(exp_data["D0"])
    H0 = float(exp_data["H0"])
    Dv = np.asarray(exp_data["Dv"], dtype=float)
    Hv = np.asarray(exp_data["Hv"], dtype=float)
    return ExperimentalQuantities(
        modes=tuple(str(x) for x in exp_data["modes"]),
        beta_exp=Dv - D0,
        dH_exp=Hv - H0,
    )


def _safe_ratio(calc: np.ndarray, exp: np.ndarray) -> np.ndarray:
    out = np.zeros_like(calc, dtype=float)
    mask = exp != 0.0
    out[mask] = calc[mask] / exp[mask]
    return out


def _sign_match(calc: np.ndarray, exp: np.ndarray) -> np.ndarray:
    out = np.zeros(calc.shape, dtype=bool)
    nz = (calc != 0.0) & (exp != 0.0)
    out[nz] = np.sign(calc[nz]) == np.sign(exp[nz])
    out[(calc == 0.0) & (exp == 0.0)] = True
    return out


def build_comparison_table(
    modes: Sequence[str],
    calc: Sequence[float],
    exp: Sequence[float],
) -> ComparisonTable:
    calc_arr = np.asarray(calc, dtype=float)
    exp_arr = np.asarray(exp, dtype=float)
    mode_tuple = tuple(str(x) for x in modes)
    order = np.argsort(np.abs(calc_arr))[::-1]
    return ComparisonTable(
        modes=mode_tuple,
        calc=calc_arr,
        exp=exp_arr,
        ratio=_safe_ratio(calc_arr, exp_arr),
        sign_match=_sign_match(calc_arr, exp_arr),
        ranking=tuple(mode_tuple[i] for i in order),
    )


def _single_index_where(items: Sequence[str], target: str) -> int:
    found = [idx for idx, item in enumerate(items) if item == target]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one {target!r}, found {found}.")
    return found[0]


def align_c2h2_linear_aliev_modes(
    *,
    payload: dict[str, Any],
    beta_parallel: Sequence[float],
    beta_perpendicular: Sequence[float],
    L_mode_calc: Sequence[float] | None = None,
    L_total_calc: float | None = None,
) -> C2H2AlignedResults:
    """Align current linear-Aliev outputs to the experimental C2H2 mode order.

    Target order is:
    `v1_CH, v2_CC, v3_asym, v4_bend, v5_bend`.

    The alignment uses only metadata already emitted by the Gaussian bootstrap:
    - `parallel_mode_irreps`
    - `omega_parallel`
    - `perpendicular_pair_irreps`
    - `omega_perpendicular`
    """

    modes = ("v1_CH", "v2_CC", "v3_asym", "v4_bend", "v5_bend")
    meta = dict(payload.get("metadata", {}))
    par_irreps = [str(x) for x in meta.get("parallel_mode_irreps", ())]
    perp_irreps = [str(x) for x in meta.get("perpendicular_pair_irreps", ())]
    par_freqs = np.asarray(payload["omega_parallel"], dtype=float)
    perp_freqs = np.asarray(payload["omega_perpendicular"], dtype=float)
    beta_par = np.asarray(beta_parallel, dtype=float)
    beta_perp = np.asarray(beta_perpendicular, dtype=float)
    if len(beta_par) != len(par_irreps) or len(beta_par) != len(par_freqs):
        raise ValueError("Parallel-mode metadata are inconsistent with beta_parallel.")
    if len(beta_perp) != len(perp_irreps) or len(beta_perp) != len(perp_freqs):
        raise ValueError("Perpendicular-mode metadata are inconsistent with beta_perpendicular.")

    sigma_u_idx = _single_index_where(par_irreps, "Sigma_u+")
    sigma_g_indices = [idx for idx, ir in enumerate(par_irreps) if ir == "Sigma_g+"]
    if len(sigma_g_indices) != 2:
        raise ValueError(f"Expected two Sigma_g+ parallel modes for C2H2, found {sigma_g_indices}.")
    sigma_g_sorted = sorted(sigma_g_indices, key=lambda idx: par_freqs[idx])
    v2_idx = sigma_g_sorted[0]
    v1_idx = sigma_g_sorted[1]

    pi_g_idx = _single_index_where(perp_irreps, "Pi_g")
    pi_u_idx = _single_index_where(perp_irreps, "Pi_u")
    if not (perp_freqs[pi_g_idx] <= perp_freqs[pi_u_idx]):
        raise ValueError("Expected Pi_g bend below Pi_u bend for C2H2 bootstrap ordering.")

    # Aliev beta coefficients enter the state dependence as
    #   D_v = D_J - sum_k beta_k * f_k(v)
    # so the experimental vibrational increment
    #   Delta D_k = D(v_k=1) - D_0
    # must be compared to `-beta_k`.
    beta_calc = -np.array(
        [
            beta_par[v1_idx],
            beta_par[v2_idx],
            beta_par[sigma_u_idx],
            beta_perp[pi_g_idx],
            beta_perp[pi_u_idx],
        ],
        dtype=float,
    )

    if L_mode_calc is None:
        l_mode = None
    else:
        L_mode_arr = np.asarray(L_mode_calc, dtype=float)
        if len(L_mode_arr) != 5:
            raise ValueError(f"Expected 5 modewise optical increments, got {len(L_mode_arr)}.")
        perp_offset = len(beta_par)
        l_mode = np.array(
            [
                L_mode_arr[v1_idx],
                L_mode_arr[v2_idx],
                L_mode_arr[sigma_u_idx],
                L_mode_arr[perp_offset + pi_g_idx],
                L_mode_arr[perp_offset + pi_u_idx],
            ],
            dtype=float,
        )

    return C2H2AlignedResults(
        modes=modes,
        beta_calc=beta_calc,
        L_mode_calc=l_mode,
        L_total_calc=None if L_total_calc is None else float(L_total_calc),
    )


def format_comparison_report(
    beta_table: ComparisonTable,
    *,
    beta_note: str | None = None,
    dH_table: ComparisonTable | None = None,
    L_total_calc: float | None = None,
    dH_note: str | None = None,
) -> str:
    lines: list[str] = []
    lines.append("=== QUARTIC CONSTANTS (beta_k) ===")
    if beta_note:
        lines.append(beta_note)
    lines.append("mode        beta_calc      beta_exp      ratio      sign")
    for idx, mode in enumerate(beta_table.modes):
        sign = "ok" if beta_table.sign_match[idx] else "flip"
        lines.append(
            f"{mode:10s} {beta_table.calc[idx]:12.3e} {beta_table.exp[idx]:12.3e} "
            f"{beta_table.ratio[idx]:10.3f} {sign:>8s}"
        )
    lines.append("")
    lines.append("beta ranking: " + ", ".join(beta_table.ranking))

    if dH_table is not None:
        lines.append("")
        lines.append("=== OPTICAL CONSTANTS (Delta H_k) ===")
        if dH_note:
            lines.append(dH_note)
        lines.append("mode        calc          exp           ratio      sign")
        for idx, mode in enumerate(dH_table.modes):
            sign = "ok" if dH_table.sign_match[idx] else "flip"
            lines.append(
                f"{mode:10s} {dH_table.calc[idx]:12.3e} {dH_table.exp[idx]:12.3e} "
                f"{dH_table.ratio[idx]:10.3f} {sign:>8s}"
            )
        lines.append("")
        lines.append("Delta H ranking: " + ", ".join(dH_table.ranking))
    elif L_total_calc is not None:
        lines.append("")
        lines.append("=== OPTICAL CONSTANT ===")
        lines.append(
            "Modewise Delta H_k comparison not available: current linear-Aliev "
            f"implementation exposes only the total L = {L_total_calc:.6e} cm^-1."
        )

    return "\n".join(lines)
