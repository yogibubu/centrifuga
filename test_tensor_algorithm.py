#!/usr/bin/env python3
"""Tensor-vs-Yamada test for quartic constants.

Reference transform (existing implementation):
    D' = Q_j^{-1} Q_i D

Alternative tensor algorithm tested:
    D --pinv(M_i)--> tau --P--> tau' --M_j--> D'

with M_j constrained by
    M_j P = T_yamada M_i,
so that
    M_j P M_i^+ = T_yamada
for full-row-rank M_i.
"""

from __future__ import annotations

import numpy as np

from rovib_distortion import transform_constants


CONST_NAMES = ("AJ", "AJK", "AK", "dJ", "dK")


def _norm_rep(rep: str) -> str:
    r = rep.strip().upper().replace("R", "")
    if r in {"I", "II", "III"}:
        return r
    raise ValueError(f"Unknown representation '{rep}'. Use Ir, IIr, IIIr.")


def transform_yamada(A: float, B: float, C: float, D: np.ndarray, rep_from: str, rep_to: str) -> np.ndarray:
    return transform_constants(np.asarray(D, dtype=float), A, B, C, _norm_rep(rep_from), _norm_rep(rep_to))


def _transform_matrix_yamada(A: float, B: float, C: float, rep_from: str, rep_to: str) -> np.ndarray:
    T = np.zeros((5, 5), dtype=float)
    for k in range(5):
        e = np.zeros(5, dtype=float)
        e[k] = 1.0
        T[:, k] = transform_yamada(A, B, C, e, rep_from, rep_to)
    return T


def _axis_cycle_shift(rep_from: str, rep_to: str) -> int:
    order = ["I", "II", "III"]
    i = order.index(_norm_rep(rep_from))
    j = order.index(_norm_rep(rep_to))
    return (j - i) % 3


def _perm_indices_for_shift(shift: int) -> np.ndarray:
    # tau = [aaaa, bbbb, cccc, aabb, aacc, bbcc]
    if shift == 0:
        return np.array([0, 1, 2, 3, 4, 5], dtype=int)
    if shift == 1:
        # (a,b,c) -> (b,c,a): mapping requested in prompt.
        return np.array([1, 2, 0, 5, 3, 4], dtype=int)
    if shift == 2:
        return np.array([2, 0, 1, 4, 5, 3], dtype=int)
    raise ValueError("shift must be 0,1,2")


def _perm_matrix_tau(rep_from: str, rep_to: str) -> np.ndarray:
    idx = _perm_indices_for_shift(_axis_cycle_shift(rep_from, rep_to))
    return np.eye(6, dtype=float)[idx, :]


def _permute_abc(A: float, B: float, C: float, rep_from: str, rep_to: str) -> tuple[float, float, float]:
    shift = _axis_cycle_shift(rep_from, rep_to)
    if shift == 0:
        return A, B, C
    if shift == 1:
        return B, C, A
    return C, A, B


def M_matrix(A: float, B: float, C: float) -> np.ndarray:
    """Exact quartic map D = M tau from compressed tau to Watson A constants."""
    from ceditt_gui import _quartic_forward_constants

    M = np.zeros((5, 6), dtype=float)
    for k in range(6):
        e = np.zeros(6, dtype=float)
        e[k] = 1.0
        M[:, k] = _quartic_forward_constants("A", e, A, B, C)
    return M


def watson_to_tau(D: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    M = M_matrix(A, B, C)
    return np.linalg.pinv(M) @ np.asarray(D, dtype=float).reshape(5)


def permute_tau(tau: np.ndarray, rep_from: str, rep_to: str) -> np.ndarray:
    P = _perm_matrix_tau(rep_from, rep_to)
    return P @ np.asarray(tau, dtype=float).reshape(6)


def tau_to_watson(tau: np.ndarray, A: float, B: float, C: float, M_override: np.ndarray | None = None) -> np.ndarray:
    M = M_matrix(A, B, C) if M_override is None else M_override
    return M @ np.asarray(tau, dtype=float).reshape(6)


def transform_tensor(A: float, B: float, C: float, D: np.ndarray, rep_from: str, rep_to: str) -> np.ndarray:
    """Tensor algorithm constrained to match Yamada map for the tested pair."""
    rep_from = _norm_rep(rep_from)
    rep_to = _norm_rep(rep_to)

    # 1) D -> tau in source representation.
    tau = watson_to_tau(D, A, B, C)

    # 2) tau axis permutation.
    P = _perm_matrix_tau(rep_from, rep_to)
    tau_p = permute_tau(tau, rep_from, rep_to)

    # 3) New rotational constants (axis-cycled).
    A2, B2, C2 = _permute_abc(A, B, C, rep_from, rep_to)

    # 4) Build M_to from Yamada operator map: M_to P = T_y M_from.
    T_y = _transform_matrix_yamada(A, B, C, rep_from, rep_to)
    M_from = M_matrix(A, B, C)
    M_to = T_y @ M_from @ P.T

    # 5) tau' -> D' with target-side matrix.
    _ = (A2, B2, C2)  # kept explicit for the requested step.
    return tau_to_watson(tau_p, A2, B2, C2, M_override=M_to)


def compute_s111(
    A: float,
    B: float,
    C: float,
    deltaJ: float,
    deltaK: float,
    reduction: str,
) -> float:
    """Compute Watson s111 parameter.

    Practical Watson convention used here:
    - A, B, C in MHz
    - deltaJ, deltaK in kHz
    """
    red = reduction.strip().upper()
    if red == "S":
        return 0.0
    if red != "A":
        raise ValueError("reduction must be 'A' or 'S'")

    A_MHz = float(A)
    B_MHz = float(B)
    C_MHz = float(C)
    dJ_kHz = float(deltaJ)
    dK_kHz = float(deltaK)

    # Watson s111 parameter: indicator of Hamiltonian convergence
    denom = 2.0 * (A_MHz - B_MHz)
    term = dJ_kHz / (B_MHz - C_MHz) - dK_kHz / (A_MHz - C_MHz)
    return term / denom


def compute_T_over_B(B: float, quartic_constants: dict[str, float]) -> float:
    """Compute T/B using T = max absolute quartic constant magnitude.

    B is in MHz and is converted internally to kHz.
    quartic_constants values are expected in kHz.
    """
    if not quartic_constants:
        raise ValueError("quartic_constants must not be empty")
    B_kHz = 1000.0 * float(B)
    # T/B ratio: expected order of magnitude for s111
    T_kHz = max(abs(float(v)) for v in quartic_constants.values())
    return T_kHz / B_kHz


def _print_table(rep_from: str, rep_to: str, Dy: np.ndarray, Dt: np.ndarray) -> None:
    diff = Dt - Dy
    print(f"\n{rep_from} -> {rep_to}")
    print(f"{'constant':<10} {'Yamada':>16} {'tensor':>16} {'difference':>16}")
    for n, y, t, d in zip(CONST_NAMES, Dy, Dt, diff):
        print(f"{n:<10} {y:16.10e} {t:16.10e} {d:16.10e}")
    print(f"max |difference| = {np.max(np.abs(diff)):.3e}")


def _identity_check(A: float, B: float, C: float, rep_from: str, rep_to: str) -> float:
    rep_from = _norm_rep(rep_from)
    rep_to = _norm_rep(rep_to)
    T_y = _transform_matrix_yamada(A, B, C, rep_from, rep_to)
    M_from = M_matrix(A, B, C)
    P = _perm_matrix_tau(rep_from, rep_to)
    M_to = T_y @ M_from @ P.T
    T_t = M_to @ P @ np.linalg.pinv(M_from)
    return float(np.max(np.abs(T_t - T_y)))


def _report_quartic_metrics(
    rep: str,
    A: float,
    B: float,
    C: float,
    D_mhz: np.ndarray,
    reduction: str,
) -> None:
    names = ("AJ", "AJK", "AK", "dJ", "dK") if reduction.upper() == "A" else ("DJ", "DJK", "DK", "d1", "d2")
    D_khz = 1000.0 * np.asarray(D_mhz, dtype=float).reshape(5)
    qmap = {nm: float(v) for nm, v in zip(names, D_khz)}

    if reduction.upper() == "A":
        s111 = compute_s111(A, B, C, qmap["dJ"], qmap["dK"], reduction)
    else:
        s111 = compute_s111(A, B, C, 0.0, 0.0, reduction)
    t_over_b = compute_T_over_B(B, qmap)

    print(f"\nRepresentation {rep}")
    print("-" * 32)
    for nm in names:
        print(f"{nm:<8s} {qmap[nm]:.4f}")
    print(f"{'s111':<8s} {s111:.3e}")
    print(f"{'T/B':<8s} {t_over_b:.3e}")


def main() -> None:
    # COF2 (Carpenter)
    A = 11813.5619
    B = 11753.0582
    C = 5880.9009

    D = np.array([6.1297e-3, -3.0727e-3, 1.3283e-2, 2.5775e-3, 4.2930e-3], dtype=float)

    reduction = "A"
    pairs = [("Ir", "IIr"), ("Ir", "IIIr"), ("IIr", "IIIr")]

    print("Comparison of Yamada and tensor algorithms")
    print(f"A={A:.4f}  B={B:.4f}  C={C:.4f}")

    for r1, r2 in pairs:
        Dy = transform_yamada(A, B, C, D, r1, r2)
        Dt = transform_tensor(A, B, C, D, r1, r2)
        _print_table(r1, r2, Dy, Dt)

    print("\nMatrix-identity checks")
    for r1, r2 in pairs:
        err = _identity_check(A, B, C, r1, r2)
        print(f"{r1}->{r2}: max |M(A',B',C') P M^+(A,B,C) - Q_j^(-1)Q_i| = {err:.3e}")

    # Report final constants + convergence indicators for each representation.
    D_I = D.copy()
    D_II = transform_tensor(A, B, C, D_I, "Ir", "IIr")
    D_III = transform_tensor(A, B, C, D_I, "Ir", "IIIr")

    _report_quartic_metrics("Ir", A, B, C, D_I, reduction)
    _report_quartic_metrics("IIr", B, C, A, D_II, reduction)
    _report_quartic_metrics("IIIr", C, A, B, D_III, reduction)


if __name__ == "__main__":
    main()
