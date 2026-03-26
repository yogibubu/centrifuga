#!/usr/bin/env python3
"""Auxiliary symbolic families for the future Aliev-equivalent linear `Dv` derivation.

This module makes explicit the intermediate coefficient families that appear in
the linear-molecule literature derivation.  At the current stage it does not
claim to reproduce Aliev's full closed formulas.  Instead it provides a stable
symbolic interface for those families so the operator-level derivation can be
grown term by term without redesigning the public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import sympy as sp


@dataclass(frozen=True)
class LinearDvAuxiliaryFamilies:
    """Symbolic placeholder families used by the full linear derivation."""

    x_nt: dict[tuple[int, int], sp.Expr]
    xn_t: dict[tuple[int, int], sp.Expr]
    r_n: dict[int, sp.Expr]
    f_nn: dict[tuple[int, int], sp.Expr]
    f_nt: dict[tuple[int, int], sp.Expr]
    u_tt: dict[tuple[int, int], sp.Expr]
    v_tt: dict[tuple[int, int], sp.Expr]


@dataclass(frozen=True)
class LinearDvReadableFamilies:
    """Auxiliary families whose formulas are already readable from the source scan.

    At the current stage only the `X`-type objects and `r_n` are promoted from
    bare symbolic placeholders to explicit symbolic formulas.  The remaining
    families stay symbolic until the operator derivation is completed.
    """

    x_nt_upper: dict[tuple[int, int], sp.Expr]
    x_nt_lower: dict[tuple[int, int], sp.Expr]
    r_n: dict[int, sp.Expr]
    f_nn: dict[tuple[int, int], sp.Expr]


@dataclass(frozen=True)
class LinearDvSkeletonFamilies:
    """Structured skeletons for auxiliary families not yet fully derived.

    These objects preserve index structure and summation topology without
    pretending that the unreadable scanned denominators have already been
    reconstructed exactly.
    """

    f_nn: dict[tuple[int, int], sp.Expr]
    f_nt: dict[tuple[int, int], sp.Expr]
    u_tt: dict[tuple[int, int], sp.Expr]
    v_tt: dict[tuple[int, int], sp.Expr]


@dataclass(frozen=True)
class LinearDvPartialFamilies:
    """Families whose outer additive structure is readable but not fully closed.

    These objects are more informative than pure skeletons: they expose the
    readable leading terms and summation topology while keeping unreadable
    denominator details inside explicit kernel symbols.
    """

    r_nt: dict[tuple[int, int], sp.Expr]
    r_tt: dict[tuple[int, int], sp.Expr]
    f_nt: dict[tuple[int, int], sp.Expr]


@dataclass(frozen=True)
class LinearDvBetaSkeleton:
    """Structured symbolic `beta` equations in Aliev-style auxiliary notation.

    These expressions are intentionally hybrid:

    - the families already readable from the scan (`X`, `r`) appear explicitly;
    - the families not yet fully derived (`F`, `U`, `V`) enter through the
      structured skeleton expressions built elsewhere in this module.
    """

    beta_parallel: dict[int, sp.Expr]
    beta_perpendicular: dict[int, sp.Expr]


@dataclass(frozen=True)
class LinearDvModePartition:
    """Partition of normal modes into parallel and perpendicular subsets."""

    parallel: tuple[int, ...]
    perpendicular: tuple[int, ...]


@dataclass(frozen=True)
class LinearAlievExplicitInputs:
    """Explicit readable inputs for the linear Aliev equations."""

    B: sp.Expr
    D_J: sp.Expr
    omega_parallel: tuple[sp.Expr, ...]
    omega_perpendicular: tuple[sp.Expr, ...]
    coriolis_nt: tuple[tuple[sp.Expr, ...], ...]
    bxx_parallel: tuple[sp.Expr, ...]
    k3_parallel: tuple[tuple[tuple[sp.Expr, ...], ...], ...]
    k3_perp_pair: tuple[tuple[tuple[sp.Expr, ...], ...], ...]
    pair_seed_perpendicular: tuple[sp.Expr, ...] | None = None
    coriolis_pair_blocks: tuple[tuple[tuple[tuple[sp.Expr, ...], ...], ...], ...] | None = None
    beta_t_xf_cross_sign: sp.Expr = sp.Integer(1)
    k4_reduced: tuple[tuple[tuple[sp.Expr, ...], ...], ...] | None = None
    k4_parallel: tuple[tuple[tuple[tuple[sp.Expr, ...], ...], ...], ...] | None = None


@dataclass(frozen=True)
class LinearAlievExplicitFamilies:
    """Readable explicit families reconstructed from Eqs. (4)-(12)."""

    C_n: dict[int, sp.Expr]
    X_upper_nt: dict[tuple[int, int], sp.Expr]
    X_lower_nt: dict[tuple[int, int], sp.Expr]
    r_n: dict[int, sp.Expr]
    r_nn: dict[tuple[int, int], sp.Expr]
    r_tt: dict[tuple[int, int], sp.Expr]
    F_upper_nn: dict[tuple[int, int], sp.Expr]
    F_lower_nn: dict[tuple[int, int], sp.Expr]
    U_nn: dict[tuple[int, int], sp.Expr]
    V_nn: dict[tuple[int, int], sp.Expr]
    U_tt: dict[tuple[int, int], sp.Expr]
    V_tt: dict[tuple[int, int], sp.Expr]
    L: sp.Expr | None


@dataclass(frozen=True)
class LinearAlievExplicitBetas:
    """Explicit `beta` families using readable Aliev blocks plus named kernels."""

    beta_parallel: dict[int, sp.Expr]
    beta_perpendicular: dict[int, sp.Expr]


@dataclass(frozen=True)
class LinearAlievExplicitBetaBreakdown:
    """Term-by-term decomposition of the explicit `beta` families."""

    parallel: dict[int, dict[str, sp.Expr]]
    perpendicular: dict[int, dict[str, sp.Expr]]


@dataclass(frozen=True)
class LinearAlievExplicitUvBreakdown:
    """Detailed decomposition of the `U/V` contribution to perpendicular betas."""

    parallel: dict[int, dict[str, sp.Expr]]
    perpendicular: dict[int, dict[str, sp.Expr]]


@dataclass(frozen=True)
class LinearAlievExplicitDvModel:
    """Operational linear-`Dv` model built from the explicit Aliev beta families.

    This stays separate from the reduced `Phi(iiik)` builder used elsewhere in
    the repo.  The purpose is to make the scanned linear formulas immediately
    usable without forcing them into the reduced semi-diagonal approximation.
    """

    dj_equilibrium: sp.Expr
    mode_kinds: tuple[str, ...]
    beta_by_mode: tuple[sp.Expr, ...]
    beta_parallel: dict[int, sp.Expr]
    beta_perpendicular: dict[int, sp.Expr]

    def state_factors(self, quanta: Sequence[object]) -> tuple[sp.Expr, ...]:
        if len(quanta) != len(self.mode_kinds):
            raise ValueError(f"Expected {len(self.mode_kinds)} quanta, got {len(quanta)}.")
        return tuple(
            sp.sympify(v) + (sp.Rational(1, 2) if kind == "parallel" else sp.Integer(1))
            for v, kind in zip(quanta, self.mode_kinds, strict=True)
        )

    def value_for_state(self, quanta: Sequence[object]) -> sp.Expr:
        factors = self.state_factors(quanta)
        return self.dj_equilibrium - sum(beta * factor for beta, factor in zip(self.beta_by_mode, factors, strict=True))


@dataclass(frozen=True)
class LinearAlievExplicitLModel:
    """Public optical-constant wrapper for the explicit linear Aliev `L` scalar."""

    value: sp.Expr
    shared_offset: sp.Expr
    mode_kinds: tuple[str, ...]
    mode_contributions: tuple[sp.Expr, ...]
    mode_contributions_with_shared_offset: tuple[sp.Expr, ...]


def _pair_seed_formula(
    inputs: LinearAlievExplicitInputs,
    *,
    t: int,
) -> sp.Expr:
    raise ValueError(
        "Pure Aliev bending seed evaluation is disabled: the Gaussian Coriolis tensor is not treated as the physical "
        "rotational-derivative zeta object required for P_t. Supply pair_seed_perpendicular explicitly."
    )


def _pair_seed_value(
    inputs: LinearAlievExplicitInputs,
    *,
    t: int,
) -> sp.Expr:
    if inputs.pair_seed_perpendicular is not None:
        return sp.sympify(inputs.pair_seed_perpendicular[t])
    return _pair_seed_formula(inputs, t=t)


def resolve_linear_aliev_quartic_mode(
    *,
    k4_parallel: object | None = None,
    k4_reduced: object | None = None,
    quartic_mode: str = "auto",
) -> str:
    """Resolve which quartic branch the linear-Aliev builder should use."""

    mode = str(quartic_mode).strip().lower()
    valid = {"auto", "full", "reduced", "none"}
    if mode not in valid:
        raise ValueError(f"Unsupported quartic mode {quartic_mode!r}. Expected one of {sorted(valid)}.")
    has_full = k4_parallel is not None
    has_reduced = k4_reduced is not None
    if mode == "auto":
        if has_full:
            return "full"
        if has_reduced:
            return "reduced"
        return "none"
    if mode == "full":
        if not has_full:
            raise ValueError("quartic_mode='full' requires k4_parallel input.")
        return "full"
    if mode == "reduced":
        if not has_reduced:
            raise ValueError("quartic_mode='reduced' requires k4_reduced input.")
        return "reduced"
    return "none"


def _quartic_term_supported(*indices: int) -> bool:
    """Return whether a quartic contribution is kept in the current partial model.

    For now we explicitly neglect only the terms whose four indices are all
    different, because those constants are not available in the present data
    flow. All repeated-index patterns are kept.
    """

    return len(set(indices)) < 4


def _reduced_quartic_key(*indices: int) -> tuple[int, int, int] | None:
    """Return the canonical `(i,j,k)` key for a reduced quartic `k_(i i j k)`.

    The reduced three-index representation is defined only for quartic patterns
    with at least one repeated index. The trailing pair is sorted to make the
    lookup canonical.
    """

    if not _quartic_term_supported(*indices):
        return None
    counts: dict[int, int] = {}
    for idx in indices:
        counts[idx] = counts.get(idx, 0) + 1
    repeated = max(counts, key=lambda idx: (counts[idx], -idx))
    if counts[repeated] < 2:
        return None
    tail = list(indices)
    tail.remove(repeated)
    tail.remove(repeated)
    if len(tail) == 1:
        tail = [tail[0], tail[0]]
    elif len(tail) == 0:
        tail = [repeated, repeated]
    tail_a, tail_b = sorted(tail)
    return repeated, tail_a, tail_b


def _quartic_value_if_supported(
    tensor4: tuple[tuple[tuple[tuple[sp.Expr, ...], ...], ...], ...] | None,
    tensor3: tuple[tuple[tuple[sp.Expr, ...], ...], ...] | None,
    *indices: int,
) -> sp.Expr:
    """Return a quartic entry from either the full or reduced representation."""

    if not _quartic_term_supported(*indices):
        return sp.Integer(0)
    if tensor4 is not None:
        i, j, k, l = indices
        return tensor4[i][j][k][l]
    if tensor3 is None:
        return sp.Integer(0)
    key = _reduced_quartic_key(*indices)
    if key is None:
        return sp.Integer(0)
    i, j, k = key
    return tensor3[i][j][k]


def _tupleize_expr_grid_2(data: Sequence[Sequence[object]]) -> tuple[tuple[sp.Expr, ...], ...]:
    return tuple(tuple(sp.sympify(item) for item in row) for row in data)


def _tupleize_expr_grid_3(
    data: Sequence[Sequence[Sequence[object]]],
) -> tuple[tuple[tuple[sp.Expr, ...], ...], ...]:
    return tuple(_tupleize_expr_grid_2(block) for block in data)


def _tupleize_expr_grid_4(
    data: Sequence[Sequence[Sequence[Sequence[object]]]],
) -> tuple[tuple[tuple[tuple[sp.Expr, ...], ...], ...], ...]:
    return tuple(_tupleize_expr_grid_3(block) for block in data)


def make_linear_aliev_explicit_inputs(
    *,
    B: object,
    D_J: object,
    omega_parallel: Sequence[object],
    omega_perpendicular: Sequence[object],
    zeta_nt: Sequence[Sequence[object]] | None = None,
    coriolis_nt: Sequence[Sequence[object]] | None = None,
    zeta_pair_blocks: Sequence[Sequence[Sequence[Sequence[object]]]] | None = None,
    coriolis_pair_blocks: Sequence[Sequence[Sequence[Sequence[object]]]] | None = None,
    bxx_parallel: Sequence[object],
    pair_seed_perpendicular: Sequence[object] | None = None,
    k3_parallel: Sequence[Sequence[Sequence[object]]],
    k3_perp_pair: Sequence[Sequence[Sequence[object]]],
    beta_t_xf_cross_sign: object = 1,
    k4_reduced: Sequence[Sequence[Sequence[object]]] | None = None,
    k4_parallel: Sequence[Sequence[Sequence[Sequence[object]]]] | None = None,
) -> LinearAlievExplicitInputs:
    """Normalize explicit readable inputs to SymPy expressions."""

    if coriolis_nt is None and zeta_nt is None:
        raise ValueError("Need coriolis_nt input.")
    if coriolis_nt is not None and zeta_nt is not None:
        raise ValueError("Pass only one of coriolis_nt or zeta_nt.")
    if coriolis_pair_blocks is not None and zeta_pair_blocks is not None:
        raise ValueError("Pass only one of coriolis_pair_blocks or zeta_pair_blocks.")
    coriolis_nt_data = coriolis_nt if coriolis_nt is not None else zeta_nt
    coriolis_pair_block_data = coriolis_pair_blocks if coriolis_pair_blocks is not None else zeta_pair_blocks

    return LinearAlievExplicitInputs(
        B=sp.sympify(B),
        D_J=sp.sympify(D_J),
        omega_parallel=tuple(sp.sympify(x) for x in omega_parallel),
        omega_perpendicular=tuple(sp.sympify(x) for x in omega_perpendicular),
        coriolis_nt=_tupleize_expr_grid_2(coriolis_nt_data),
        bxx_parallel=tuple(sp.sympify(x) for x in bxx_parallel),
        pair_seed_perpendicular=None
        if pair_seed_perpendicular is None
        else tuple(sp.sympify(x) for x in pair_seed_perpendicular),
        k3_parallel=_tupleize_expr_grid_3(k3_parallel),
        k3_perp_pair=_tupleize_expr_grid_3(k3_perp_pair),
        coriolis_pair_blocks=None if coriolis_pair_block_data is None else _tupleize_expr_grid_4(coriolis_pair_block_data),
        beta_t_xf_cross_sign=sp.sympify(beta_t_xf_cross_sign),
        k4_reduced=None if k4_reduced is None else _tupleize_expr_grid_3(k4_reduced),
        k4_parallel=None if k4_parallel is None else _tupleize_expr_grid_4(k4_parallel),
    )


def make_linear_aliev_explicit_inputs_from_mapping(
    payload: Mapping[str, object],
    *,
    quartic_mode: str = "auto",
) -> LinearAlievExplicitInputs:
    """Build explicit inputs from a mapping payload used by external programs."""

    mode = resolve_linear_aliev_quartic_mode(
        k4_parallel=payload.get("k4_parallel"),
        k4_reduced=payload.get("k4_reduced"),
        quartic_mode=quartic_mode,
    )
    return make_linear_aliev_explicit_inputs(
        B=payload["B"],
        D_J=payload["D_J"],
        omega_parallel=payload["omega_parallel"],
        omega_perpendicular=payload["omega_perpendicular"],
        coriolis_nt=payload.get("coriolis_nt", payload.get("zeta_nt")),
        coriolis_pair_blocks=payload.get("coriolis_pair_blocks", payload.get("zeta_pair_blocks")),
        bxx_parallel=payload["bxx_parallel"],
        pair_seed_perpendicular=payload.get("pair_seed_perpendicular"),
        k3_parallel=payload["k3_parallel"],
        k3_perp_pair=payload["k3_perp_pair"],
        beta_t_xf_cross_sign=payload.get("beta_t_xf_cross_sign", 1),
        k4_reduced=payload.get("k4_reduced") if mode == "reduced" else None,
        k4_parallel=payload.get("k4_parallel") if mode == "full" else None,
    )


def build_explicit_aliev_families(inputs: LinearAlievExplicitInputs) -> LinearAlievExplicitFamilies:
    """Build the explicit readable families from the scanned equations."""

    B = inputs.B
    D_J = inputs.D_J
    omega_n = inputs.omega_parallel
    omega_t = inputs.omega_perpendicular
    coriolis = inputs.coriolis_nt
    n_parallel = len(omega_n)
    n_perp = len(omega_t)
    if len(coriolis) != n_parallel or any(len(row) != n_perp for row in coriolis):
        raise ValueError("coriolis_nt shape must be (n_parallel, n_perpendicular).")
    if len(inputs.bxx_parallel) != n_parallel:
        raise ValueError("bxx_parallel length must match omega_parallel.")

    C_n = {n: sp.simplify(-B * inputs.bxx_parallel[n] / omega_n[n]) for n in range(n_parallel)}

    X_upper_nt: dict[tuple[int, int], sp.Expr] = {}
    X_lower_nt: dict[tuple[int, int], sp.Expr] = {}
    for n in range(n_parallel):
        for t in range(n_perp):
            wn = omega_n[n]
            wt = omega_t[t]
            z = coriolis[n][t]
            denom = wn**2 - wt**2
            X_upper_nt[(n, t)] = sp.simplify(2 * B * z * sp.sqrt(wn * wt) / denom)
            X_lower_nt[(n, t)] = sp.simplify(B * z * (wn**2 + wt**2) / (sp.sqrt(wn * wt) * denom))

    r_n: dict[int, sp.Expr] = {}
    for n in range(n_parallel):
        cubic_sum = sp.Rational(1, 2) * sum(
            inputs.k3_parallel[n][np][npp] * C_n[np] * C_n[npp]
            for np in range(n_parallel)
            for npp in range(n_parallel)
        )
        r_n[n] = sp.simplify(4 * omega_n[n] * C_n[n] * (D_J / B - B**2 / omega_n[n] ** 2) + cubic_sum)

    r_nn: dict[tuple[int, int], sp.Expr] = {}
    for n in range(n_parallel):
        for np_ in range(n_parallel):
            cubic_sum = sp.Rational(1, 2) * sum(inputs.k3_parallel[n][np_][npp] * C_n[npp] for npp in range(n_parallel))
            r_nn[(n, np_)] = sp.simplify(
                sp.Rational(3, 4) * omega_n[n] * omega_n[np_] * C_n[n] * C_n[np_] / B + cubic_sum
            )

    r_tt: dict[tuple[int, int], sp.Expr] = {}
    for t in range(n_perp):
        for tp in range(n_perp):
            cubic_sum = sp.Rational(1, 2) * sum(inputs.k3_perp_pair[t][tp][n] * C_n[n] for n in range(n_parallel))
            r_tt[(t, tp)] = sp.simplify(cubic_sum)

    F_upper_nn: dict[tuple[int, int], sp.Expr] = {}
    F_lower_nn: dict[tuple[int, int], sp.Expr] = {}
    for n in range(n_parallel):
        for np_ in range(n_parallel):
            wn = omega_n[n]
            wnp = omega_n[np_]
            acc_upper = sp.Integer(0)
            acc_lower = sp.Integer(0)
            for t in range(n_perp):
                wt = omega_t[t]
                zprod = coriolis[n][t] * coriolis[np_][t]
                acc_upper += zprod * sp.sqrt(wn * wnp) * (wn**2 + wnp**2 - 2 * wt**2) / (
                    (wn**2 - wt**2) * (wnp**2 - wt**2)
                )
                acc_lower += zprod * (wn**2 * wnp**2 - wt**4) / (
                    wt * sp.sqrt(wn * wnp) * (wn**2 - wt**2) * (wnp**2 - wt**2)
                )
            F_upper_nn[(n, np_)] = sp.simplify(B**2 * acc_upper)
            F_lower_nn[(n, np_)] = sp.simplify(r_nn[(n, np_)] + B**2 * acc_lower)

    U_nn: dict[tuple[int, int], sp.Expr] = {}
    V_nn: dict[tuple[int, int], sp.Expr] = {}
    for n in range(n_parallel):
        for np_ in range(n_parallel):
            wn = omega_n[n]
            wnp = omega_n[np_]
            u_sum = sp.Integer(0)
            v_sum = sp.Integer(0)
            for t in range(n_perp):
                wt = omega_t[t]
                zprod = coriolis[n][t] * coriolis[np_][t]
                common = zprod / (wt * sp.sqrt(wn * wnp))
                u_num = (wn + wt) ** 2 * (wnp - wt) ** 2 + (wn - wt) ** 2 * (wnp + wt) ** 2
                u_den = (wn**2 - wt**2) * (wnp**2 - wt**2)
                u_sum += common * u_num / u_den
                if n != np_:
                    v_num = (
                        (wn + wt) ** 2 * (wnp + wt) ** 2 * (wn + wnp - 2 * wt)
                        - (wn - wt) ** 2 * (wnp - wt) ** 2 * (wn + wnp + 2 * wt)
                    )
                    v_den = (wn**2 - wt**2) * (wnp**2 - wt**2) * (wn - wnp)
                    v_sum += common * v_num / v_den
            if n == np_:
                # In the Gaussian-coupled degenerate representation, the diagonal
                # B^2 zeta^2 self-term is not treated as an observable invariant.
                # Keep only the r_nn contribution on the diagonal.
                U_nn[(n, np_)] = sp.simplify(r_nn[(n, np_)] / (4 * (wn + wnp)))
                V_nn[(n, np_)] = sp.Integer(0)
            else:
                U_nn[(n, np_)] = sp.simplify(r_nn[(n, np_)] / (4 * (wn + wnp)) + B**2 * u_sum / 8)
                V_nn[(n, np_)] = sp.simplify(r_nn[(n, np_)] / (4 * (wn - wnp)) + B**2 * v_sum / 8)

    U_tt: dict[tuple[int, int], sp.Expr] = {}
    V_tt: dict[tuple[int, int], sp.Expr] = {}
    for t in range(n_perp):
        for tp in range(n_perp):
            wt = omega_t[t]
            wtp = omega_t[tp]
            u_sum = sp.Integer(0)
            v_sum = sp.Integer(0)
            for n in range(n_parallel):
                wn = omega_n[n]
                zprod = coriolis[n][t] * coriolis[n][tp]
                common = zprod / (wn * sp.sqrt(wt * wtp))
                u_num = (wt + wn) ** 2 * (wtp - wn) ** 2 + (wt - wn) ** 2 * (wtp + wn) ** 2
                u_den = (wt**2 - wn**2) * (wtp**2 - wn**2)
                u_sum += common * u_num / u_den
                if t != tp:
                    v_num = (
                        (wt + wtp) ** 2 * (wtp + wn) ** 2 * (wt + wtp - 2 * wn)
                        - (wt - wtp) ** 2 * (wtp - wn) ** 2 * (wt + wtp + 2 * wn)
                    )
                    v_den = (wt**2 - wn**2) * (wtp**2 - wn**2) * (wt - wtp)
                    v_sum += common * v_num / v_den
            if t == tp:
                # As in the parallel-sector correction, exclude the diagonal
                # self-coupling B^2 zeta^2 contribution on the degenerate-pair
                # diagonal and keep only the r_tt term.
                U_tt[(t, tp)] = sp.simplify(r_tt[(t, tp)] / (4 * (wt + wtp)))
                V_tt[(t, tp)] = sp.Integer(0)
            else:
                U_tt[(t, tp)] = sp.simplify(r_tt[(t, tp)] / (4 * (wt + wtp)) + B**2 * u_sum / 8)
                V_tt[(t, tp)] = sp.simplify(r_tt[(t, tp)] / (4 * (wt - wtp)) + B**2 * v_sum / 8)

    L = None
    if inputs.k4_parallel is not None or inputs.k4_reduced is not None:
        quartic_sum = sp.Rational(1, 24) * sum(
            _quartic_value_if_supported(inputs.k4_parallel, inputs.k4_reduced, n, np_, npp, nppp)
            * C_n[n]
            * C_n[np_]
            * C_n[npp]
            * C_n[nppp]
            for n in range(n_parallel)
            for np_ in range(n_parallel)
            for npp in range(n_parallel)
            for nppp in range(n_parallel)
        )
        L = sp.simplify(
            -8 * D_J**3 / B**2
            + quartic_sum
            + 8 * B * D_J * sum(C_n[n] ** 2 / omega_n[n] for n in range(n_parallel))
            - sum(r_n[n] ** 2 / (2 * omega_n[n]) for n in range(n_parallel))
        )

    return LinearAlievExplicitFamilies(
        C_n=C_n,
        X_upper_nt=X_upper_nt,
        X_lower_nt=X_lower_nt,
        r_n=r_n,
        r_nn=r_nn,
        r_tt=r_tt,
        F_upper_nn=F_upper_nn,
        F_lower_nn=F_lower_nn,
        U_nn=U_nn,
        V_nn=V_nn,
        U_tt=U_tt,
        V_tt=V_tt,
        L=L,
    )


def build_explicit_aliev_betas(inputs: LinearAlievExplicitInputs) -> LinearAlievExplicitBetas:
    """Build `beta_n` and `beta_t` from the readable scan structure.

    Current approximation policy for quartics:

    - accept either a full four-index field `k4_parallel` or a reduced
      three-index field `k4_reduced`, interpreted as `k_(i i j k)`;
    - keep repeated-index quartic patterns that are representable in the
      available input;
    - drop only the terms whose four indices are all different.
    """

    fam = build_explicit_aliev_families(inputs)
    B = inputs.B
    omega_n = inputs.omega_parallel
    omega_t = inputs.omega_perpendicular
    n_parallel = len(omega_n)
    n_perp = len(omega_t)

    beta_n: dict[int, sp.Expr] = {}
    for n in range(n_parallel):
        direct = sp.simplify(4 * omega_n[n] ** 2 * fam.C_n[n] ** 2 * (B / omega_n[n] ** 2 - inputs.D_J / B**2))
        quartic_seed = sp.Rational(1, 4) * sum(
            _quartic_value_if_supported(inputs.k4_parallel, inputs.k4_reduced, n, n, np_, npp)
            * fam.C_n[np_]
            * fam.C_n[npp]
            for np_ in range(n_parallel)
            for npp in range(n_parallel)
        )
        quartic_r_block = -sum(
            _quartic_value_if_supported(inputs.k4_parallel, inputs.k4_reduced, n, n, np_, np_)
            * fam.r_n[np_]
            / (2 * omega_n[np_])
            for np_ in range(n_parallel)
        )
        mixed_prefactor_block = -4 * B * sum(
            fam.C_n[np_]
            * inputs.coriolis_nt[np_][t]
            * (
                (3 * omega_n[n] ** 2 + omega_t[t] ** 2)
                * omega_n[n] ** sp.Rational(3, 2)
                * fam.C_n[n]
                * inputs.coriolis_nt[n][t]
                + (omega_n[n] ** 2 + omega_t[t] ** 2)
                * omega_n[n] ** sp.Rational(3, 2)
                * fam.C_n[n]
                * fam.r_nn[(n, np_)]
            )
            / (sp.sqrt(omega_n[n] * omega_t[t]) * (omega_n[n] ** 2 - omega_t[t] ** 2))
            for np_ in range(n_parallel)
            for t in range(n_perp)
        )
        xf_block = 4 * sum(
            fam.X_lower_nt[(n, t)] * fam.X_lower_nt[(np_, t)] * fam.F_upper_nn[(n, np_)]
            + fam.X_upper_nt[(n, t)] * fam.X_upper_nt[(np_, t)] * fam.F_lower_nn[(n, np_)]
            for np_ in range(n_parallel)
            for t in range(n_perp)
        )
        cross_block = -4 * sum(
            fam.X_lower_nt[(n, t)] * fam.X_upper_nt[(np_, t)] * fam.F_lower_nn[(n, np_)]
            + fam.X_upper_nt[(n, t)] * fam.X_lower_nt[(np_, t)] * fam.F_upper_nn[(n, np_)]
            for np_ in range(n_parallel)
            for t in range(n_perp)
        )
        uv_block = sp.Integer(0)
        beta_n[n] = direct + quartic_seed + quartic_r_block + mixed_prefactor_block + xf_block + cross_block + uv_block

    beta_t: dict[int, sp.Expr] = {}
    for t in range(n_perp):
        xf_cross_sign = inputs.beta_t_xf_cross_sign
        quartic_seed = sp.Rational(1, 4) * sum(
            _quartic_value_if_supported(inputs.k4_parallel, inputs.k4_reduced, n, np_, n, np_)
            * fam.C_n[n]
            * fam.C_n[np_]
            for n in range(n_parallel)
            for np_ in range(n_parallel)
        )
        pair_seed = _pair_seed_value(inputs, t=t)
        linear_r_block = -sum(
            inputs.k3_perp_pair[t][t][n] * fam.r_n[n] / (2 * omega_n[n])
            for n in range(n_parallel)
        )
        mixed_prefactor_block = -4 * B * sum(
            fam.C_n[n]
            * inputs.coriolis_nt[n][t]
            * (
                2
                * omega_t[t] ** 2
                * omega_n[np_] ** sp.Rational(3, 2)
                * fam.C_n[np_]
                * inputs.coriolis_nt[np_][t]
                + (3 * omega_t[t] ** 2 + omega_n[np_] ** 2)
                * omega_n[np_] ** sp.Rational(3, 2)
                * fam.C_n[np_]
                * fam.r_nn[(n, np_)]
            )
            / (omega_t[t] * sp.sqrt(omega_n[np_]) * (omega_t[t] ** 2 - omega_n[np_] ** 2))
            for n in range(n_parallel)
            for np_ in range(n_parallel)
        )
        xf_block_raw = 4 * sum(
            fam.X_lower_nt[(n, t)] * fam.X_lower_nt[(np_, t)] * fam.F_upper_nn[(n, np_)]
            + fam.X_upper_nt[(n, t)] * fam.X_upper_nt[(np_, t)] * fam.F_lower_nn[(n, np_)]
            for n in range(n_parallel)
            for np_ in range(n_parallel)
        )
        cross_block_raw = -4 * sum(
            fam.X_lower_nt[(n, t)] * fam.X_upper_nt[(np_, t)] * fam.F_lower_nn[(n, np_)]
            + fam.X_upper_nt[(n, t)] * fam.X_lower_nt[(np_, t)] * fam.F_upper_nn[(n, np_)]
            for n in range(n_parallel)
            for np_ in range(n_parallel)
        )
        xf_block = sp.simplify(xf_cross_sign * xf_block_raw)
        cross_block = sp.simplify(xf_cross_sign * cross_block_raw)
        uv_block = -16 * sum(
            fam.U_tt[(t, tp)] * (omega_t[t] + omega_t[tp])
            + fam.V_tt[(t, tp)] * (omega_t[tp] - omega_t[t])
            for tp in range(n_perp)
        )
        beta_t[t] = quartic_seed + pair_seed + linear_r_block + mixed_prefactor_block + xf_block + cross_block + uv_block

    return LinearAlievExplicitBetas(beta_parallel=beta_n, beta_perpendicular=beta_t)


def build_explicit_aliev_beta_breakdown(
    inputs: LinearAlievExplicitInputs,
) -> LinearAlievExplicitBetaBreakdown:
    """Return a term-by-term decomposition of `beta_n` and `beta_t`."""

    fam = build_explicit_aliev_families(inputs)
    B = inputs.B
    omega_n = inputs.omega_parallel
    omega_t = inputs.omega_perpendicular
    n_parallel = len(omega_n)
    n_perp = len(omega_t)

    beta_n: dict[int, dict[str, sp.Expr]] = {}
    for n in range(n_parallel):
        direct = sp.simplify(4 * omega_n[n] ** 2 * fam.C_n[n] ** 2 * (B / omega_n[n] ** 2 - inputs.D_J / B**2))
        quartic_seed = sp.Rational(1, 4) * sum(
            _quartic_value_if_supported(inputs.k4_parallel, inputs.k4_reduced, n, n, np_, npp)
            * fam.C_n[np_]
            * fam.C_n[npp]
            for np_ in range(n_parallel)
            for npp in range(n_parallel)
        )
        quartic_r_block = -sum(
            _quartic_value_if_supported(inputs.k4_parallel, inputs.k4_reduced, n, n, np_, np_)
            * fam.r_n[np_]
            / (2 * omega_n[np_])
            for np_ in range(n_parallel)
        )
        mixed_prefactor_block = -4 * B * sum(
            fam.C_n[np_]
            * inputs.coriolis_nt[np_][t]
            * (
                (3 * omega_n[n] ** 2 + omega_t[t] ** 2)
                * omega_n[n] ** sp.Rational(3, 2)
                * fam.C_n[n]
                * inputs.coriolis_nt[n][t]
                + (omega_n[n] ** 2 + omega_t[t] ** 2)
                * omega_n[n] ** sp.Rational(3, 2)
                * fam.C_n[n]
                * fam.r_nn[(n, np_)]
            )
            / (sp.sqrt(omega_n[n] * omega_t[t]) * (omega_n[n] ** 2 - omega_t[t] ** 2))
            for np_ in range(n_parallel)
            for t in range(n_perp)
        )
        xf_block = 4 * sum(
            fam.X_lower_nt[(n, t)] * fam.X_lower_nt[(np_, t)] * fam.F_upper_nn[(n, np_)]
            + fam.X_upper_nt[(n, t)] * fam.X_upper_nt[(np_, t)] * fam.F_lower_nn[(n, np_)]
            for np_ in range(n_parallel)
            for t in range(n_perp)
        )
        cross_block = -4 * sum(
            fam.X_lower_nt[(n, t)] * fam.X_upper_nt[(np_, t)] * fam.F_lower_nn[(n, np_)]
            + fam.X_upper_nt[(n, t)] * fam.X_lower_nt[(np_, t)] * fam.F_upper_nn[(n, np_)]
            for np_ in range(n_parallel)
            for t in range(n_perp)
        )
        uv_block = sp.Integer(0)
        total = sp.simplify(direct + quartic_seed + quartic_r_block + mixed_prefactor_block + xf_block + cross_block + uv_block)
        beta_n[n] = {
            "direct": sp.simplify(direct),
            "quartic_seed": sp.simplify(quartic_seed),
            "quartic_r_block": sp.simplify(quartic_r_block),
            "mixed_prefactor_block": sp.simplify(mixed_prefactor_block),
            "xf_block": sp.simplify(xf_block),
            "cross_block": sp.simplify(cross_block),
            "uv_block": sp.simplify(uv_block),
            "total": total,
        }

    beta_t: dict[int, dict[str, sp.Expr]] = {}
    for t in range(n_perp):
        xf_cross_sign = inputs.beta_t_xf_cross_sign
        quartic_seed = sp.Rational(1, 4) * sum(
            _quartic_value_if_supported(inputs.k4_parallel, inputs.k4_reduced, n, np_, n, np_)
            * fam.C_n[n]
            * fam.C_n[np_]
            for n in range(n_parallel)
            for np_ in range(n_parallel)
        )
        pair_seed = _pair_seed_value(inputs, t=t)
        linear_r_block = -sum(
            inputs.k3_perp_pair[t][t][n] * fam.r_n[n] / (2 * omega_n[n])
            for n in range(n_parallel)
        )
        mixed_prefactor_block = -4 * B * sum(
            fam.C_n[n]
            * inputs.coriolis_nt[n][t]
            * (
                2
                * omega_t[t] ** 2
                * omega_n[np_] ** sp.Rational(3, 2)
                * fam.C_n[np_]
                * inputs.coriolis_nt[np_][t]
                + (3 * omega_t[t] ** 2 + omega_n[np_] ** 2)
                * omega_n[np_] ** sp.Rational(3, 2)
                * fam.C_n[np_]
                * fam.r_nn[(n, np_)]
            )
            / (omega_t[t] * sp.sqrt(omega_n[np_]) * (omega_t[t] ** 2 - omega_n[np_] ** 2))
            for n in range(n_parallel)
            for np_ in range(n_parallel)
        )
        xf_block_raw = 4 * sum(
            fam.X_lower_nt[(n, t)] * fam.X_lower_nt[(np_, t)] * fam.F_upper_nn[(n, np_)]
            + fam.X_upper_nt[(n, t)] * fam.X_upper_nt[(np_, t)] * fam.F_lower_nn[(n, np_)]
            for n in range(n_parallel)
            for np_ in range(n_parallel)
        )
        cross_block_raw = -4 * sum(
            fam.X_lower_nt[(n, t)] * fam.X_upper_nt[(np_, t)] * fam.F_lower_nn[(n, np_)]
            + fam.X_upper_nt[(n, t)] * fam.X_lower_nt[(np_, t)] * fam.F_upper_nn[(n, np_)]
            for n in range(n_parallel)
            for np_ in range(n_parallel)
        )
        xf_block = sp.simplify(xf_cross_sign * xf_block_raw)
        cross_block = sp.simplify(xf_cross_sign * cross_block_raw)
        uv_block = -16 * sum(
            fam.U_tt[(t, tp)] * (omega_t[t] + omega_t[tp])
            + fam.V_tt[(t, tp)] * (omega_t[tp] - omega_t[t])
            for tp in range(n_perp)
        )
        total = sp.simplify(quartic_seed + pair_seed + linear_r_block + mixed_prefactor_block + xf_block + cross_block + uv_block)
        beta_t[t] = {
            "quartic_seed": sp.simplify(quartic_seed),
            "pair_seed": sp.simplify(pair_seed),
            "linear_r_block": sp.simplify(linear_r_block),
            "mixed_prefactor_block": sp.simplify(mixed_prefactor_block),
            "xf_block": sp.simplify(xf_block),
            "cross_block": sp.simplify(cross_block),
            "uv_block": sp.simplify(uv_block),
            "total": total,
        }

    return LinearAlievExplicitBetaBreakdown(parallel=beta_n, perpendicular=beta_t)


def build_explicit_aliev_uv_breakdown(
    inputs: LinearAlievExplicitInputs,
) -> LinearAlievExplicitUvBreakdown:
    """Return a detailed decomposition of the perpendicular `U/V` block.

    This is a diagnostic helper for the current linear prototype. It isolates
    the diagonal and off-diagonal pieces of the final `-16` block after the
    Eq. (8)-(10) correction. Under the present physical interpretation of the
    Gaussian-coupled degenerate subspace, the diagonal `B^2 zeta^2` term in
    `U_(n,n)` is excluded and therefore reported as zero.
    """

    fam = build_explicit_aliev_families(inputs)
    omega_n = inputs.omega_parallel
    omega_t = inputs.omega_perpendicular
    n_parallel = len(omega_n)

    parallel: dict[int, dict[str, sp.Expr]] = {}
    perpendicular: dict[int, dict[str, sp.Expr]] = {}

    for n in range(n_parallel):
        wn = omega_n[n]
        u_diag_r = sp.simplify(-32 * wn * (fam.r_nn[(n, n)] / (8 * wn)))
        u_diag_b = sp.Integer(0)
        u_offdiag = sp.Integer(0)
        v_offdiag = sp.Integer(0)
        for np_ in range(n_parallel):
            if np_ == n:
                continue
            u_offdiag += sp.simplify(-16 * fam.U_nn[(n, np_)] * (omega_n[n] + omega_n[np_]))
            v_offdiag += sp.simplify(-16 * fam.V_nn[(n, np_)] * (omega_n[np_] - omega_n[n]))
        uv_total = sp.simplify(u_diag_r + u_diag_b + u_offdiag + v_offdiag)
        parallel[n] = {
            "u_diag_rterm": sp.simplify(u_diag_r),
            "u_diag_bterm": sp.simplify(u_diag_b),
            "u_diag_total": sp.simplify(u_diag_r + u_diag_b),
            "u_offdiag": sp.simplify(u_offdiag),
            "v_offdiag": sp.simplify(v_offdiag),
            "uv_total": uv_total,
        }
    for t in range(len(omega_t)):
        wt = omega_t[t]
        u_diag_r = sp.simplify(-16 * fam.U_tt[(t, t)] * (wt + wt))
        u_diag_b = sp.Integer(0)
        u_offdiag = sp.Integer(0)
        v_offdiag = sp.Integer(0)
        for tp in range(len(omega_t)):
            if tp == t:
                continue
            u_offdiag += sp.simplify(-16 * fam.U_tt[(t, tp)] * (wt + omega_t[tp]))
            v_offdiag += sp.simplify(-16 * fam.V_tt[(t, tp)] * (omega_t[tp] - wt))
        uv_total = sp.simplify(u_diag_r + u_diag_b + u_offdiag + v_offdiag)
        perpendicular[t] = {
            "u_diag_rterm": sp.simplify(u_diag_r),
            "u_diag_bterm": sp.simplify(u_diag_b),
            "u_diag_total": sp.simplify(u_diag_r + u_diag_b),
            "u_offdiag": sp.simplify(u_offdiag),
            "v_offdiag": sp.simplify(v_offdiag),
            "uv_total": uv_total,
        }
    return LinearAlievExplicitUvBreakdown(parallel=parallel, perpendicular=perpendicular)


def build_explicit_aliev_dv_model(
    inputs: LinearAlievExplicitInputs,
) -> LinearAlievExplicitDvModel:
    """Return the explicit linear-molecule `Dv` model from the Aliev beta families."""

    betas = build_explicit_aliev_betas(inputs)
    mode_kinds = ("parallel",) * len(inputs.omega_parallel) + ("perpendicular",) * len(inputs.omega_perpendicular)
    beta_by_mode = tuple(betas.beta_parallel[n] for n in range(len(inputs.omega_parallel))) + tuple(
        betas.beta_perpendicular[t] for t in range(len(inputs.omega_perpendicular))
    )
    return LinearAlievExplicitDvModel(
        dj_equilibrium=inputs.D_J,
        mode_kinds=mode_kinds,
        beta_by_mode=beta_by_mode,
        beta_parallel=betas.beta_parallel,
        beta_perpendicular=betas.beta_perpendicular,
    )


def build_explicit_aliev_L_model(
    inputs: LinearAlievExplicitInputs,
) -> LinearAlievExplicitLModel:
    """Return the explicit linear-molecule optical constant `L`.

    Eq. (12) requires quartic input. The helper accepts either the full
    four-index tensor `k4_parallel` or the reduced three-index representation
    `k4_reduced(i,j,k) = k_(i i j k)`.
    """

    fam = build_explicit_aliev_families(inputs)
    if fam.L is None:
        raise ValueError("Eq. (12) for L requires quartic input: k4_parallel or k4_reduced.")
    B = inputs.B
    D_J = inputs.D_J
    omega_n = inputs.omega_parallel
    n_parallel = len(omega_n)
    n_perp = len(inputs.omega_perpendicular)
    shared_offset = sp.simplify(-8 * D_J**3 / B**2)

    quartic_mode_terms = [sp.Integer(0) for _ in range(n_parallel)]
    for n in range(n_parallel):
        for np_ in range(n_parallel):
            for npp in range(n_parallel):
                for nppp in range(n_parallel):
                    q = _quartic_value_if_supported(inputs.k4_parallel, inputs.k4_reduced, n, np_, npp, nppp)
                    if q == 0:
                        continue
                    factors = (
                        fam.C_n[n],
                        fam.C_n[np_],
                        fam.C_n[npp],
                        fam.C_n[nppp],
                    )
                    term = sp.simplify(sp.Rational(1, 24) * q * factors[0] * factors[1] * factors[2] * factors[3])
                    counts = {
                        idx: (1 if idx == n else 0)
                        + (1 if idx == np_ else 0)
                        + (1 if idx == npp else 0)
                        + (1 if idx == nppp else 0)
                        for idx in {n, np_, npp, nppp}
                    }
                    multiplicity = sp.Integer(sum(counts.values()))
                    for idx, count in counts.items():
                        quartic_mode_terms[idx] += sp.simplify(term * sp.Rational(count, multiplicity))

    mode_contributions_parallel = []
    for n in range(n_parallel):
        geom = sp.simplify(8 * B * D_J * fam.C_n[n] ** 2 / omega_n[n])
        rterm = sp.simplify(-fam.r_n[n] ** 2 / (2 * omega_n[n]))
        total_n = sp.simplify(quartic_mode_terms[n] + geom + rterm)
        mode_contributions_parallel.append(total_n)

    mode_kinds = ("parallel",) * n_parallel + ("perpendicular",) * n_perp
    mode_contributions = tuple(mode_contributions_parallel) + (sp.Integer(0),) * n_perp
    shared_piece = sp.simplify(shared_offset / sp.Integer(n_parallel + n_perp))
    mode_contributions_with_shared_offset = tuple(sp.simplify(x + shared_piece) for x in mode_contributions)

    return LinearAlievExplicitLModel(
        value=fam.L,
        shared_offset=shared_offset,
        mode_kinds=mode_kinds,
        mode_contributions=mode_contributions,
        mode_contributions_with_shared_offset=mode_contributions_with_shared_offset,
    )


def partition_linear_modes(mode_kinds: tuple[str, ...] | list[str]) -> LinearDvModePartition:
    """Return explicit parallel/perpendicular mode index sets."""

    kinds = tuple(str(item).strip().lower() for item in mode_kinds)
    bad = [kind for kind in kinds if kind not in {"parallel", "perpendicular"}]
    if bad:
        raise ValueError(f"Unsupported mode kind(s): {bad}.")
    parallel = tuple(i for i, kind in enumerate(kinds) if kind == "parallel")
    perpendicular = tuple(i for i, kind in enumerate(kinds) if kind == "perpendicular")
    return LinearDvModePartition(parallel=parallel, perpendicular=perpendicular)


def build_auxiliary_symbol_families(n_parallel: int, n_perpendicular: int) -> LinearDvAuxiliaryFamilies:
    """Create symbolic auxiliary families in the notation of the linear derivation."""

    x_nt = {(n, t): sp.Symbol(f"X_{n}_{t}", real=True) for n in range(n_parallel) for t in range(n_perpendicular)}
    xn_t = {(n, t): sp.Symbol(f"Xbar_{n}_{t}", real=True) for n in range(n_parallel) for t in range(n_perpendicular)}
    r_n = {n: sp.Symbol(f"r_{n}", real=True) for n in range(n_parallel)}
    f_nn = {(n, np): sp.Symbol(f"Fnn_{n}_{np}", real=True) for n in range(n_parallel) for np in range(n_parallel)}
    f_nt = {(n, t): sp.Symbol(f"Fnt_{n}_{t}", real=True) for n in range(n_parallel) for t in range(n_perpendicular)}
    u_tt = {(t, tp): sp.Symbol(f"U_{t}_{tp}", real=True) for t in range(n_perpendicular) for tp in range(n_perpendicular)}
    v_tt = {(t, tp): sp.Symbol(f"V_{t}_{tp}", real=True) for t in range(n_perpendicular) for tp in range(n_perpendicular)}
    return LinearDvAuxiliaryFamilies(
        x_nt=x_nt,
        xn_t=xn_t,
        r_n=r_n,
        f_nn=f_nn,
        f_nt=f_nt,
        u_tt=u_tt,
        v_tt=v_tt,
    )


def build_reduced_auxiliary_bridge(mode_kinds: tuple[str, ...] | list[str]) -> dict[str, object]:
    """Return the mode partition plus symbolic auxiliary families.

    This is the bridge object the operator-level derivation should populate as
    the full Aliev-equivalent calculation is unfolded.
    """

    part = partition_linear_modes(mode_kinds)
    aux = build_auxiliary_symbol_families(len(part.parallel), len(part.perpendicular))
    return {
        "partition": part,
        "auxiliaries": aux,
        "status": "symbolic_scaffold_only",
    }


def build_readable_auxiliary_families(mode_kinds: tuple[str, ...] | list[str]) -> LinearDvReadableFamilies:
    """Return the auxiliary families that are already readable from the scan.

    The notation follows the partially legible equations in the Aliev scan:

    - `X^{nt} = 2 B zeta_nt sqrt(omega_n omega_t) / (omega_n^2 - omega_t^2)`
    - `X_{nt} = B zeta_nt (omega_n^2 + omega_t^2) / [sqrt(omega_n omega_t) (omega_n^2 - omega_t^2)]`
    - `r_n = 4 omega_n C_n (D_J / B - B^2 / omega_n^2) + 1/2 sum k_{n n' n''} C_{n'} C_{n''}`

    In addition, the scan of Eq. (6) is readable enough to promote the
    `F^{nn'}` family to an explicit symbolic formula:

    - `F^{nn'} = B^2 sum_t zeta_{nt} zeta_{n't} sqrt(omega_n omega_n')
       (omega_n^2 + omega_n'^2 - 2 omega_t^2) /
       [(omega_n^2 - omega_t^2)(omega_n'^2 - omega_t^2)]`

    These are the families that can be introduced safely without guessing the
    unreadable parts of the scanned equations.
    """

    part = partition_linear_modes(mode_kinds)
    n_parallel = len(part.parallel)
    n_perpendicular = len(part.perpendicular)

    B = sp.Symbol("B", positive=True)
    DJ = sp.Symbol("D_J", real=True)

    omega_n = sp.symbols(f"omega_n0:{n_parallel}", positive=True)
    omega_t = sp.symbols(f"omega_t0:{n_perpendicular}", positive=True)
    zeta_nt = {(n, t): sp.Symbol(f"zeta_{n}_{t}", real=True) for n in range(n_parallel) for t in range(n_perpendicular)}
    Cn = {n: sp.Symbol(f"C_{n}", real=True) for n in range(n_parallel)}
    k_npp = {
        (n, np, npp): sp.Symbol(f"k_{n}_{np}_{npp}", real=True)
        for n in range(n_parallel)
        for np in range(n_parallel)
        for npp in range(n_parallel)
    }

    x_upper: dict[tuple[int, int], sp.Expr] = {}
    x_lower: dict[tuple[int, int], sp.Expr] = {}
    for n in range(n_parallel):
        for t in range(n_perpendicular):
            wn = omega_n[n]
            wt = omega_t[t]
            z = zeta_nt[(n, t)]
            x_upper[(n, t)] = sp.simplify(2 * B * z * sp.sqrt(wn * wt) / (wn**2 - wt**2))
            x_lower[(n, t)] = sp.simplify(B * z * (wn**2 + wt**2) / (sp.sqrt(wn * wt) * (wn**2 - wt**2)))

    r_n: dict[int, sp.Expr] = {}
    for n in range(n_parallel):
        wn = omega_n[n]
        cubic_sum = sp.Rational(1, 2) * sum(
            k_npp[(n, np, npp)] * Cn[np] * Cn[npp]
            for np in range(n_parallel)
            for npp in range(n_parallel)
        )
        r_n[n] = sp.simplify(4 * wn * Cn[n] * (DJ / B - B**2 / wn**2) + cubic_sum)

    f_nn: dict[tuple[int, int], sp.Expr] = {}
    for n in range(n_parallel):
        for np in range(n_parallel):
            wn = omega_n[n]
            wnp = omega_n[np]
            acc = sp.Integer(0)
            for t in range(n_perpendicular):
                wt = omega_t[t]
                z_left = zeta_nt[(n, t)]
                z_right = zeta_nt[(np, t)]
                acc += (
                    z_left
                    * z_right
                    * sp.sqrt(wn * wnp)
                    * (wn**2 + wnp**2 - 2 * wt**2)
                    / ((wn**2 - wt**2) * (wnp**2 - wt**2))
                )
            f_nn[(n, np)] = sp.simplify(B**2 * acc)

    return LinearDvReadableFamilies(
        x_nt_upper=x_upper,
        x_nt_lower=x_lower,
        r_n=r_n,
        f_nn=f_nn,
    )


def build_structured_family_skeletons(mode_kinds: tuple[str, ...] | list[str]) -> LinearDvSkeletonFamilies:
    """Return symbolic skeletons for the `F`, `U`, and `V` families.

    The scan makes the existence and index pattern of these families clear, but
    not every denominator is readable enough to encode as a trustworthy closed
    formula.  This builder therefore records:

    - which sums belong to which family,
    - which indices are coupled,
    - which lower-level readable families feed them.

    The resulting expressions are symbolic skeletons, intentionally marked by
    dedicated `K_*` kernels that must later be replaced by derived formulas.
    """

    part = partition_linear_modes(mode_kinds)
    n_parallel = len(part.parallel)
    n_perpendicular = len(part.perpendicular)

    B = sp.Symbol("B", positive=True)
    readable = build_readable_auxiliary_families(mode_kinds)

    f_nn: dict[tuple[int, int], sp.Expr] = {}
    for n in range(n_parallel):
        for np in range(n_parallel):
            delta = sp.Integer(1) if n == np else sp.Integer(0)
            kernel_sum = sum(
                sp.Symbol(f"K_Fnn_{n}_{np}_{t}", real=True)
                for t in range(n_perpendicular)
            )
            base = readable.f_nn[(n, np)] + delta * readable.r_n[n]
            f_nn[(n, np)] = sp.simplify(base + B**2 * kernel_sum)

    f_nt: dict[tuple[int, int], sp.Expr] = {}
    for n in range(n_parallel):
        for t in range(n_perpendicular):
            base = sp.Symbol(f"r_{n}_{t}", real=True)
            kernel_sum = sum(
                sp.Symbol(f"K_Fnt_{n}_{t}_{tp}", real=True)
                for tp in range(n_perpendicular)
            )
            f_nt[(n, t)] = sp.simplify(base + B**2 * kernel_sum)

    u_tt: dict[tuple[int, int], sp.Expr] = {}
    v_tt: dict[tuple[int, int], sp.Expr] = {}
    for t in range(n_perpendicular):
        for tp in range(n_perpendicular):
            u_sum = sum(sp.Symbol(f"K_U_{t}_{tp}_{n}", real=True) for n in range(n_parallel))
            v_sum = sum(sp.Symbol(f"K_V_{t}_{tp}_{n}", real=True) for n in range(n_parallel))
            u_tt[(t, tp)] = sp.simplify(u_sum)
            v_tt[(t, tp)] = sp.simplify(v_sum)

    return LinearDvSkeletonFamilies(
        f_nn=f_nn,
        f_nt=f_nt,
        u_tt=u_tt,
        v_tt=v_tt,
    )


def build_partially_readable_families(mode_kinds: tuple[str, ...] | list[str]) -> LinearDvPartialFamilies:
    """Return families with readable outer structure but kernelized inner detail.

    From the scan, the additive pattern of the following families is clear:

    - `r_nt = leading readable term + cubic sum`
    - `r_tt = cubic sum`
    - `F_nt = r_nt + B^2 * sum(...)`

    The exact denominators inside the final sums are still left as kernels.
    """

    part = partition_linear_modes(mode_kinds)
    n_parallel = len(part.parallel)
    n_perpendicular = len(part.perpendicular)

    B = sp.Symbol("B", positive=True)
    omega_n = sp.symbols(f"omega_n0:{n_parallel}", positive=True)
    omega_t = sp.symbols(f"omega_t0:{n_perpendicular}", positive=True)
    Cn = {n: sp.Symbol(f"C_{n}", real=True) for n in range(n_parallel)}
    Ct = {t: sp.Symbol(f"Ct_{t}", real=True) for t in range(n_perpendicular)}

    r_nt: dict[tuple[int, int], sp.Expr] = {}
    for n in range(n_parallel):
        for t in range(n_perpendicular):
            leading = sp.Rational(3, 4) * omega_n[n] * omega_t[t] * Cn[n] * Ct[t] / B
            cubic = sp.Rational(1, 2) * sum(
                sp.Symbol(f"k_rnt_{n}_{t}_{a}", real=True) * Cn[a]
                for a in range(n_parallel)
            )
            r_nt[(n, t)] = sp.simplify(leading + cubic)

    r_tt: dict[tuple[int, int], sp.Expr] = {}
    for t in range(n_perpendicular):
        for tp in range(n_perpendicular):
            cubic = sp.Rational(1, 2) * sum(
                sp.Symbol(f"k_rtt_{t}_{tp}_{n}", real=True) * Cn[n]
                for n in range(n_parallel)
            )
            r_tt[(t, tp)] = sp.simplify(cubic)

    f_nt: dict[tuple[int, int], sp.Expr] = {}
    for n in range(n_parallel):
        for t in range(n_perpendicular):
            kernel_sum = sum(
                sp.Symbol(f"K_Fnt_{n}_{t}_{tp}", real=True)
                for tp in range(n_perpendicular)
            )
            f_nt[(n, t)] = sp.simplify(r_nt[(n, t)] + B**2 * kernel_sum)

    return LinearDvPartialFamilies(r_nt=r_nt, r_tt=r_tt, f_nt=f_nt)


def build_beta_equation_skeletons(mode_kinds: tuple[str, ...] | list[str]) -> LinearDvBetaSkeleton:
    """Return Aliev-style symbolic skeletons for `beta_n` and `beta_t`.

    This is the first place in the codebase where the auxiliary families are
    reassembled into the target spectroscopic objects.  The intent is:

    - preserve the literature-level block structure,
    - avoid inventing unreadable denominators,
    - make the remaining derivation a local replacement of symbolic kernels.
    """

    part = partition_linear_modes(mode_kinds)
    readable = build_readable_auxiliary_families(mode_kinds)
    skeleton = build_structured_family_skeletons(mode_kinds)
    partial = build_partially_readable_families(mode_kinds)

    beta_n: dict[int, sp.Expr] = {}
    for n in range(len(part.parallel)):
        quartic_seed = sp.Symbol(f"Qpar_{n}", real=True)
        cubic_seed = sp.Symbol(f"Cpar_{n}", real=True)
        x_block = sum(readable.x_nt_upper[(n, t)] * readable.x_nt_lower[(n, t)] for t in range(len(part.perpendicular)))
        f_block = sum(skeleton.f_nn[(n, np)] for np in range(len(part.parallel)))
        mix_block = sum(partial.f_nt[(n, t)] for t in range(len(part.perpendicular)))
        beta_n[n] = sp.simplify(quartic_seed + cubic_seed + readable.r_n[n] + x_block + f_block + mix_block)

    beta_t: dict[int, sp.Expr] = {}
    for t in range(len(part.perpendicular)):
        quartic_seed = sp.Symbol(f"Qperp_{t}", real=True)
        cubic_seed = sp.Symbol(f"Cperp_{t}", real=True)
        x_block = sum(readable.x_nt_upper[(n, t)] * readable.x_nt_lower[(n, t)] for n in range(len(part.parallel)))
        u_block = sum(skeleton.u_tt[(t, tp)] for tp in range(len(part.perpendicular)))
        v_block = sum(skeleton.v_tt[(t, tp)] for tp in range(len(part.perpendicular)))
        beta_t[t] = sp.simplify(quartic_seed + cubic_seed + x_block + u_block + v_block)

    return LinearDvBetaSkeleton(beta_parallel=beta_n, beta_perpendicular=beta_t)
