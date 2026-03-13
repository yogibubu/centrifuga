#!/usr/bin/env python3
"""Quartic Watson transforms through tau tensor only.

Pipeline:
    Watson constants -> tau reconstruction -> representation permutation
    -> Watson constants in target reduction/representation

No direct analytical A<->S formulas are used.
"""

from __future__ import annotations

import numpy as np

from ceditt_gui import get_Q_A, get_Q_S, _quartic_forward_constants

REPRESENTATIONS = ("I", "II", "III")
REDUCTIONS = ("A", "S")

NAMES_A = ("dJ", "dJK", "dK", "deltaJ", "deltaK")
NAMES_S = ("DJ", "DJK", "DK", "d1", "d2")


def _norm_rep(rep: str) -> str:
    r = rep.strip().upper().replace("R", "")
    if r not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation '{rep}'")
    return r


def _norm_red(red: str) -> str:
    r = red.strip().upper()
    if r not in REDUCTIONS:
        raise ValueError(f"Unknown reduction '{red}'")
    return r


def _cycle_shift(rep_from: str, rep_to: str) -> int:
    order = ["I", "II", "III"]
    i = order.index(_norm_rep(rep_from))
    j = order.index(_norm_rep(rep_to))
    return (j - i) % 3


def rotate_abc(A: float, B: float, C: float, rep_from: str, rep_to: str) -> tuple[float, float, float]:
    s = _cycle_shift(rep_from, rep_to)
    if s == 0:
        return A, B, C
    if s == 1:
        return B, C, A
    return C, A, B


def tau_perm(rep_from: str, rep_to: str) -> np.ndarray:
    """Permutation matrix on tau = [aaaa, bbbb, cccc, aabb, aacc, bbcc]."""
    shift = _cycle_shift(rep_from, rep_to)
    if shift == 0:
        idx = np.array([0, 1, 2, 3, 4, 5], dtype=int)
    elif shift == 1:
        idx = np.array([1, 2, 0, 5, 3, 4], dtype=int)
    else:
        idx = np.array([2, 0, 1, 4, 5, 3], dtype=int)
    return np.eye(6, dtype=float)[idx, :]


def _Q(red: str, rep: str, A: float, B: float, C: float) -> np.ndarray:
    return get_Q_A(rep, A, B, C) if _norm_red(red) == "A" else get_Q_S(rep, A, B, C)


def T_yamada(red: str, rep_from: str, rep_to: str, A: float, B: float, C: float) -> np.ndarray:
    qf = _Q(red, rep_from, A, B, C)
    qt = _Q(red, rep_to, A, B, C)
    return np.linalg.solve(qt, qf)


def K_tau_to_watson(red: str, rep: str, A: float, B: float, C: float) -> np.ndarray:
    """Build K(red,rep) with D = K tau using the exact quartic projection equations."""
    red = _norm_red(red)
    rep = _norm_rep(rep)
    K = np.zeros((5, 6), dtype=float)
    for j in range(6):
        e = np.zeros(6, dtype=float)
        e[j] = 1.0
        K[:, j] = _quartic_forward_constants(red, tau_perm("I", rep) @ e, *rotate_abc(A, B, C, "I", rep))
    return K


def watson_to_tau(D: np.ndarray, red: str, rep: str, A: float, B: float, C: float) -> np.ndarray:
    Dv = np.asarray(D, dtype=float).reshape(5)
    K = K_tau_to_watson(red, rep, A, B, C)
    return np.linalg.pinv(K) @ Dv


def tau_to_watson(tau: np.ndarray, red: str, rep: str, A: float, B: float, C: float) -> np.ndarray:
    tauv = np.asarray(tau, dtype=float).reshape(6)
    K = K_tau_to_watson(red, rep, A, B, C)
    return K @ tauv


def transform_watson(
    A: float,
    B: float,
    C: float,
    D: np.ndarray,
    rep_from: str,
    red_from: str,
    rep_to: str,
    red_to: str,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Full transform via tau tensor only."""
    rep_from = _norm_rep(rep_from)
    red_from = _norm_red(red_from)
    rep_to = _norm_rep(rep_to)
    red_to = _norm_red(red_to)

    tau = watson_to_tau(D, red_from, rep_from, A, B, C)
    P = tau_perm(rep_from, rep_to)
    tau_p = P @ tau

    A2, B2, C2 = rotate_abc(A, B, C, rep_from, rep_to)
    D2 = tau_to_watson(tau_p, red_to, rep_to, A2, B2, C2)
    return D2, (A2, B2, C2)


def compute_s111(A: float, B: float, C: float, D_khz: np.ndarray, reduction: str) -> float:
    """Compute s111.

    Rotational constants are in MHz; quartic constants are in kHz.
    kHz -> MHz conversion is applied before evaluation.
    """
    red = _norm_red(reduction)
    d = np.asarray(D_khz, dtype=float).reshape(5) / 1000.0
    if red == "A":
        xj, xk = d[3], d[4]
    else:
        xj, xk = d[3], d[4]
    return (1.0 / (2.0 * (A - B))) * (xj / (B - C) - xk / (A - C))


def _fmt_block(title: str, names: tuple[str, ...], vals: np.ndarray, s111: float) -> None:
    print(f"\n{title}")
    print("-" * 36)
    for n, v in zip(names, vals):
        print(f"{n:<8s} {float(v): .6f}")
    print(f"{'s111':<8s} {s111:.3e}")


def _validate(A: float, B: float, C: float, D_A: np.ndarray) -> None:
    y12 = T_yamada("A", "I", "II", A, B, C) @ D_A
    y13 = T_yamada("A", "I", "III", A, B, C) @ D_A

    t12, _ = transform_watson(A, B, C, D_A, "I", "A", "II", "A")
    t13, _ = transform_watson(A, B, C, D_A, "I", "A", "III", "A")

    print("\nValidation vs Yamada (representation only, A reduction)")
    print(f"max|tau - Yamada| I->II = {np.max(np.abs(t12 - y12)):.3e}")
    print(f"max|tau - Yamada| I->III = {np.max(np.abs(t13 - y13)):.3e}")

    dS, _ = transform_watson(A, B, C, D_A, "I", "A", "I", "S")
    dA_back, _ = transform_watson(A, B, C, dS, "I", "S", "I", "A")
    print("\nValidation A->S->A")
    print(f"max|D_back - D_in| = {np.max(np.abs(dA_back - D_A)):.3e}")


def main() -> None:
    # Example input (kHz)
    A, B, C = 7036.579634, 6910.830104, 4218.779498
    D_A_I = np.array([4.0509, -4.4528, 6.7065, 1.4549, 1.2856], dtype=float)

    D_IIIA, (A3, B3, C3) = transform_watson(A, B, C, D_A_I, "I", "A", "III", "A")
    D_IIIS, (A3s, B3s, C3s) = transform_watson(A, B, C, D_A_I, "I", "A", "III", "S")

    _fmt_block("IIIr, A (from Ir, A via tau)", NAMES_A, D_IIIA, compute_s111(A3, B3, C3, D_IIIA, "A"))
    _fmt_block("IIIr, S (from Ir, A via tau)", NAMES_S, D_IIIS, compute_s111(A3s, B3s, C3s, D_IIIS, "S"))

    _validate(A, B, C, D_A_I)


if __name__ == "__main__":
    main()
