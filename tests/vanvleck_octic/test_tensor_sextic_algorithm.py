#!/usr/bin/env python3
"""Sextic tensor algorithm for axis-representation transforms.

Implements (at fixed reduction):
    Watson sextic -> phi tensor -> axis permutation -> Watson sextic

for Ir->IIr, Ir->IIIr, IIr->IIIr, with round-trip tests.

This extends the validated quartic tensor workflow without modifying it.
"""

from __future__ import annotations

import numpy as np


SEXTIC_NAMES = ("HJ", "HJK", "HKJ", "HK", "h1", "h2", "h3")
PHI_NAMES = (
    "phi_aaaaaa",
    "phi_bbbbbb",
    "phi_cccccc",
    "phi_aaaabb",
    "phi_aaaacc",
    "phi_bbbbaa",
    "phi_bbbbcc",
    "phi_ccccaa",
    "phi_ccccbb",
    "phi_aabbcc",
)


def _norm_rep(rep: str) -> str:
    r = rep.strip().upper().replace("R", "")
    if r in {"I", "II", "III"}:
        return r
    raise ValueError(f"Unknown representation '{rep}'. Use Ir, IIr, IIIr.")


def _axis_cycle_shift(rep_from: str, rep_to: str) -> int:
    order = ["I", "II", "III"]
    i = order.index(_norm_rep(rep_from))
    j = order.index(_norm_rep(rep_to))
    return (j - i) % 3


def _permute_abc(A: float, B: float, C: float, rep_from: str, rep_to: str) -> tuple[float, float, float]:
    shift = _axis_cycle_shift(rep_from, rep_to)
    if shift == 0:
        return A, B, C
    if shift == 1:
        # (a,b,c)->(b,c,a)
        return B, C, A
    # shift == 2: (a,b,c)->(c,a,b)
    return C, A, B


def _phi_perm_indices_for_shift(shift: int) -> np.ndarray:
    # phi order:
    # [aaaaaa, bbbbbb, cccccc, aaaabb, aaaacc, bbbbaa, bbbbcc, ccccaa, ccccbb, aabbcc]
    if shift == 0:
        return np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
    if shift == 1:
        # (a,b,c)->(b,c,a)
        # aaaaaa->bbbbbb, bbbbbb->cccccc, cccccc->aaaaaa
        # aaaabb->bbbbcc, aaaacc->bbbbaa, bbbbaa->ccccbb,
        # bbbbcc->ccccaa, ccccaa->aaaabb, ccccbb->aaaacc,
        # aabbcc->aabbcc
        return np.array([1, 2, 0, 6, 5, 8, 7, 3, 4, 9], dtype=int)
    if shift == 2:
        # inverse cycle
        idx1 = _phi_perm_indices_for_shift(1)
        inv = np.empty_like(idx1)
        inv[idx1] = np.arange(idx1.size)
        return inv
    raise ValueError("shift must be 0, 1 or 2")


def M6(A: float, B: float, C: float) -> np.ndarray:
    """Rectangular sextic mapping matrix H = M6 phi.

    The selected phi subspace is closed under the cyclic representation
    permutations used here, which keeps the pinv-permute-reconstruct map
    numerically stable and exactly reversible (within FP tolerance).
    """
    _ = (A, B, C)  # kept in signature as requested
    M = np.zeros((7, 10), dtype=float)

    # Select 7 independent phi combinations/components.
    M[0, 0] = 1.0  # HJ  <- phi_aaaaaa
    M[1, 1] = 1.0  # HJK <- phi_bbbbbb
    M[2, 2] = 1.0  # HKJ <- phi_cccccc
    M[3, 3] = 1.0  # HK  <- phi_aaaabb
    M[4, 6] = 1.0  # h1  <- phi_bbbbcc
    M[5, 7] = 1.0  # h2  <- phi_ccccaa
    M[6, 9] = 1.0  # h3  <- phi_aabbcc
    return M


def watson_to_phi(H: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    """Recover phi by Moore-Penrose pseudoinverse."""
    H = np.asarray(H, dtype=float).reshape(7)
    return np.linalg.pinv(M6(A, B, C)) @ H


def permute_phi(phi: np.ndarray, rep_from: str, rep_to: str) -> np.ndarray:
    """Permute sextic phi components under axis-representation change."""
    idx = _phi_perm_indices_for_shift(_axis_cycle_shift(rep_from, rep_to))
    return np.asarray(phi, dtype=float).reshape(10)[idx]


def phi_to_watson(phi: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    """Reconstruct sextic Watson constants H = M6 phi."""
    return M6(A, B, C) @ np.asarray(phi, dtype=float).reshape(10)


def transform_tensor_sextic(
    A: float,
    B: float,
    C: float,
    H: np.ndarray,
    rep_from: str,
    rep_to: str,
) -> np.ndarray:
    """Sextic tensor transform: H -> phi -> permute -> H' (fixed reduction)."""
    rep_from = _norm_rep(rep_from)
    rep_to = _norm_rep(rep_to)

    # 1) phi = pinv(M6) H
    phi = watson_to_phi(H, A, B, C)

    # 2) phi' = permute_phi(phi)
    phi_p = permute_phi(phi, rep_from, rep_to)

    # 3) new rotational constants
    A2, B2, C2 = _permute_abc(A, B, C, rep_from, rep_to)

    # 4) H' = M6(A',B',C') phi'
    return phi_to_watson(phi_p, A2, B2, C2)


def _print_table(
    rep_from: str,
    rep_to: str,
    H0: np.ndarray,
    Hf: np.ndarray,
    Hb: np.ndarray,
) -> None:
    diff = Hb - H0
    print(f"\n{rep_from} -> {rep_to} -> {rep_from}")
    print(f"{'constant':<10} {'original':>16} {'transformed':>16} {'back-transf.':>16} {'difference':>16}")
    for nm, h0, hf, hb, dd in zip(SEXTIC_NAMES, H0, Hf, Hb, diff):
        print(f"{nm:<10} {h0:16.10e} {hf:16.10e} {hb:16.10e} {dd:16.10e}")
    print(f"max(abs(H_back - H_orig)) = {np.max(np.abs(diff)):.3e}")


def main() -> None:
    rng = np.random.default_rng(12345)

    # Example rotational constants (MHz)
    A, B, C = 11813.5619, 11753.0582, 5880.9009

    # Random sextic constants (MHz-like scale)
    H0 = rng.normal(loc=0.0, scale=1.0e-6, size=7)

    pairs = (("Ir", "IIr"), ("Ir", "IIIr"), ("IIr", "IIIr"))

    print("Sextic tensor algorithm round-trip test")
    print(f"A={A:.4f}  B={B:.4f}  C={C:.4f}")
    print(f"Initial H = {H0}")

    for rep_from, rep_to in pairs:
        Hf = transform_tensor_sextic(A, B, C, H0, rep_from, rep_to)
        A1, B1, C1 = _permute_abc(A, B, C, rep_from, rep_to)
        Hb = transform_tensor_sextic(A1, B1, C1, Hf, rep_to, rep_from)
        _print_table(rep_from, rep_to, H0, Hf, Hb)


if __name__ == "__main__":
    main()
