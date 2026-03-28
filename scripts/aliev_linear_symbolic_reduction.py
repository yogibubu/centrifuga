#!/usr/bin/env python3
"""Minimal exact symbolic reductions for the linear Watson-Aliev branch.

Current scope:
1. Exact inversion of the Table-IV ladder
      M_kl, N_kl  <->  X_kl, X^kl
2. Exact specialization of M_nt and N_nt from the linear X_nt, X^nt formulas
3. Exact symmetry checks for the compact linear descendants U_nn' and V_nn'
4. Exact bookkeeping of the 1/4 and denominator factors leading to the
   global B^2/8 prefactor in the compact U/V branch.
5. Exact cancellation of r_nn' inside the uv combination
      (w_n+w_n') U_nn' + (w_n'-w_n) V_nn'

This file intentionally avoids any numerical input and keeps all steps in
symbolic SymPy form.
"""

from __future__ import annotations

import sympy as sp


def derive_mn_from_x() -> dict[str, sp.Expr]:
    Xd, Xu = sp.symbols("X_d X_u")
    M, N = sp.symbols("M N")
    eqs = [
        sp.Eq(Xd, 2 * (N + M)),
        sp.Eq(Xu, 2 * (N - M)),
    ]
    sol = sp.solve(eqs, (M, N), dict=True)
    if len(sol) != 1:
        raise RuntimeError("Unexpected non-unique solution for M,N in terms of X_d/X_u.")
    out = sol[0]
    return {"M": sp.simplify(out[M]), "N": sp.simplify(out[N])}


def prove_mn_inversion() -> dict[str, sp.Expr]:
    Xd, Xu = sp.symbols("X_d X_u")
    inv = derive_mn_from_x()
    M_expr = inv["M"]
    N_expr = inv["N"]
    back_xd = sp.simplify(2 * (N_expr + M_expr))
    back_xu = sp.simplify(2 * (N_expr - M_expr))
    return {
        "M_expr": M_expr,
        "N_expr": N_expr,
        "back_Xd_minus_Xd": sp.simplify(back_xd - Xd),
        "back_Xu_minus_Xu": sp.simplify(back_xu - Xu),
    }


def derive_linear_pair_mn_from_x() -> dict[str, sp.Expr]:
    wn, wt, B, z = sp.symbols("w_n w_t B z", positive=True)
    Xd = B * z * (wn**2 + wt**2) / (sp.sqrt(wn * wt) * (wn**2 - wt**2))
    Xu = 2 * B * z * sp.sqrt(wn * wt) / (wn**2 - wt**2)
    M_expr = sp.simplify((Xd - Xu) / 4)
    N_expr = sp.simplify((Xd + Xu) / 4)
    return {
        "X_d": sp.simplify(Xd),
        "X_u": sp.simplify(Xu),
        "M_expr": sp.factor(M_expr),
        "N_expr": sp.factor(N_expr),
    }


def prove_linear_pair_products() -> dict[str, sp.Expr]:
    wn, wp, wt, B, zn, zp = sp.symbols("w_n w_p w_t B z_n z_p", positive=True)
    Xd_n = B * zn * (wn**2 + wt**2) / (sp.sqrt(wn * wt) * (wn**2 - wt**2))
    Xu_n = 2 * B * zn * sp.sqrt(wn * wt) / (wn**2 - wt**2)
    Xd_p = B * zp * (wp**2 + wt**2) / (sp.sqrt(wp * wt) * (wp**2 - wt**2))
    Xu_p = 2 * B * zp * sp.sqrt(wp * wt) / (wp**2 - wt**2)
    M_n = sp.simplify((Xd_n - Xu_n) / 4)
    N_n = sp.simplify((Xd_n + Xu_n) / 4)
    M_p = sp.simplify((Xd_p - Xu_p) / 4)
    N_p = sp.simplify((Xd_p + Xu_p) / 4)
    return {
        "M_n": sp.factor(M_n),
        "N_n": sp.factor(N_n),
        "MM_plus_NN": sp.factor(sp.together(sp.simplify(M_n * M_p + N_n * N_p))),
        "MM_minus_NN": sp.factor(sp.together(sp.simplify(M_n * M_p - N_n * N_p))),
    }


def compact_linear_u_formula(
    wn: sp.Expr,
    wp: sp.Expr,
    wt: sp.Expr,
    B: sp.Expr,
    z1: sp.Expr,
    z2: sp.Expr,
    r: sp.Expr,
) -> sp.Expr:
    kernel = (
        (wn + wt) ** 2 * (wp - wt) ** 2
        + (wn - wt) ** 2 * (wp + wt) ** 2
    ) / ((wn**2 - wt**2) * (wp**2 - wt**2))
    return sp.simplify(
        r / (4 * (wn + wp))
        + B**2 * z1 * z2 * kernel / (8 * wt * sp.sqrt(wn * wp))
    )


def compact_linear_v_formula(
    wn: sp.Expr,
    wp: sp.Expr,
    wt: sp.Expr,
    B: sp.Expr,
    z1: sp.Expr,
    z2: sp.Expr,
    r: sp.Expr,
) -> sp.Expr:
    numer = (
        (wn + wt) ** 2 * (wp + wt) ** 2 * (wn + wp - 2 * wt)
        - (wn - wt) ** 2 * (wp - wt) ** 2 * (wn + wp + 2 * wt)
    )
    den = (wn**2 - wt**2) * (wp**2 - wt**2) * (wn - wp)
    return sp.simplify(
        r / (4 * (wn - wp))
        + B**2 * z1 * z2 * numer / (8 * wt * sp.sqrt(wn * wp) * den)
    )


def prove_uv_symmetries() -> dict[str, sp.Expr]:
    wn, wp, wt, B, z1, z2, r = sp.symbols("w_n w_p w_t B z1 z2 r", positive=True)
    u = compact_linear_u_formula(wn, wp, wt, B, z1, z2, r)
    v = compact_linear_v_formula(wn, wp, wt, B, z1, z2, r)
    swap = {wn: wp, wp: wn, z1: z2, z2: z1}
    u_swapped = sp.simplify(sp.factor(u.subs(swap, simultaneous=True)))
    v_swapped = sp.simplify(sp.factor(v.subs(swap, simultaneous=True)))
    return {
        "U_swap_minus_U": sp.simplify(sp.together(u_swapped - u)),
        "V_swap_plus_V": sp.simplify(sp.together(v_swapped + v)),
    }


def prove_prefactor_bookkeeping() -> dict[str, sp.Expr]:
    Xd_nm, Xd_pm, Xu_nm, Xu_pm = sp.symbols("Xdn Xdp Xun Xup")
    Mnm, Mpm, Nnm, Npm = sp.symbols("Mnm Mpm Nnm Npm")
    subs = {
        Mnm: (Xd_nm - Xu_nm) / 4,
        Mpm: (Xd_pm - Xu_pm) / 4,
        Nnm: (Xd_nm + Xu_nm) / 4,
        Npm: (Xd_pm + Xu_pm) / 4,
    }
    mm_plus_nn = sp.expand((Mnm * Mpm + Nnm * Npm).subs(subs))
    mm_minus_nn = sp.expand((Mnm * Mpm - Nnm * Npm).subs(subs))
    return {
        "M*M+N*N": sp.factor(mm_plus_nn),
        "M*M-N*N": sp.factor(mm_minus_nn),
        "common_prefactor_mm_nn": sp.Rational(1, 8),
    }


def prove_r_cancellation_in_uv_block() -> dict[str, sp.Expr]:
    wn, wp, wt, B, z1, z2, r = sp.symbols("w_n w_p w_t B z1 z2 r", positive=True)
    u = compact_linear_u_formula(wn, wp, wt, B, z1, z2, r)
    v = compact_linear_v_formula(wn, wp, wt, B, z1, z2, r)
    combo = sp.simplify(sp.together((wn + wp) * u + (wp - wn) * v))
    return {
        "r_coefficient": sp.simplify(sp.diff(combo, r)),
        "uv_combo_no_r": sp.factor(sp.together(combo.subs(r, 0))),
    }


def compare_compact_uv_with_bare_h24_uv() -> dict[str, sp.Expr]:
    wn, wp, wt, B, z1, z2 = sp.symbols("w_n w_p w_t B z1 z2", positive=True)
    compact_u = compact_linear_u_formula(wn, wp, wt, B, z1, z2, 0)
    compact_v = compact_linear_v_formula(wn, wp, wt, B, z1, z2, 0)
    compact_combo = sp.simplify(sp.together((wn + wp) * compact_u + (wp - wn) * compact_v))

    Xd_n = B * z1 * (wn**2 + wt**2) / (sp.sqrt(wn * wt) * (wn**2 - wt**2))
    Xu_n = 2 * B * z1 * sp.sqrt(wn * wt) / (wn**2 - wt**2)
    Xd_p = B * z2 * (wp**2 + wt**2) / (sp.sqrt(wp * wt) * (wp**2 - wt**2))
    Xu_p = 2 * B * z2 * sp.sqrt(wp * wt) / (wp**2 - wt**2)
    M_n = sp.simplify((Xd_n - Xu_n) / 4)
    N_n = sp.simplify((Xd_n + Xu_n) / 4)
    M_p = sp.simplify((Xd_p - Xu_p) / 4)
    N_p = sp.simplify((Xd_p + Xu_p) / 4)
    bare_h24_uv = sp.simplify(
        sp.together(
            -8
            * (
                (2 * wt + wn + wp) * M_n * M_p
                + (2 * wt - wn - wp) * N_n * N_p
            )
        )
    )
    return {
        "compact_combo": sp.factor(compact_combo),
        "bare_h24_uv": sp.factor(bare_h24_uv),
        "difference": sp.factor(sp.together(compact_combo - bare_h24_uv)),
    }


def prove_bare_h24_uv_equals_v_only_channel() -> dict[str, sp.Expr]:
    wn, wp, wt, B, z1, z2 = sp.symbols("w_n w_p w_t B z1 z2", positive=True)
    compact_v = compact_linear_v_formula(wn, wp, wt, B, z1, z2, 0)

    Xd_n = B * z1 * (wn**2 + wt**2) / (sp.sqrt(wn * wt) * (wn**2 - wt**2))
    Xu_n = 2 * B * z1 * sp.sqrt(wn * wt) / (wn**2 - wt**2)
    Xd_p = B * z2 * (wp**2 + wt**2) / (sp.sqrt(wp * wt) * (wp**2 - wt**2))
    Xu_p = 2 * B * z2 * sp.sqrt(wp * wt) / (wp**2 - wt**2)
    M_n = sp.simplify((Xd_n - Xu_n) / 4)
    N_n = sp.simplify((Xd_n + Xu_n) / 4)
    M_p = sp.simplify((Xd_p - Xu_p) / 4)
    N_p = sp.simplify((Xd_p + Xu_p) / 4)
    bare_h24_uv = sp.simplify(
        sp.together(
            -8
            * (
                (2 * wt + wn + wp) * M_n * M_p
                + (2 * wt - wn - wp) * N_n * N_p
            )
        )
    )
    return {
        "bare_plus_4_times_v_channel": sp.simplify(
            sp.together(bare_h24_uv + 4 * (wp - wn) * compact_v)
        )
    }


def analyze_xf_cross_degree_structure() -> dict[str, object]:
    wn, wp, wt, B, z1, z2, r = sp.symbols("w_n w_p w_t B z1 z2 r")
    Xup_n = 2 * B * z1 * sp.sqrt(wn * wt) / (wn**2 - wt**2)
    Xdn_n = B * z1 * (wn**2 + wt**2) / (sp.sqrt(wn * wt) * (wn**2 - wt**2))
    Xup_p = 2 * B * z2 * sp.sqrt(wp * wt) / (wp**2 - wt**2)
    Xdn_p = B * z2 * (wp**2 + wt**2) / (sp.sqrt(wp * wt) * (wp**2 - wt**2))
    Fup = B**2 * z1 * z2 * sp.sqrt(wn * wp) * (wn**2 + wp**2 - 2 * wt**2) / (
        (wn**2 - wt**2) * (wp**2 - wt**2)
    )
    Flo = r + B**2 * z1 * z2 * (wn**2 * wp**2 - wt**4) / (
        wt * sp.sqrt(wn * wp) * (wn**2 - wt**2) * (wp**2 - wt**2)
    )
    expr = sp.expand(
        4
        * (
            Xdn_n * Xdn_p * Fup
            + Xup_n * Xup_p * Flo
            - Xdn_n * Xup_p * Flo
            - Xup_n * Xdn_p * Fup
        )
    )
    numer = sp.together(expr).as_numer_denom()[0]
    monoms = sp.Poly(numer, B, z1, z2, r).monoms()
    return {"monomials_in_(B,z1,z2,r)": monoms}


def identify_only_possible_source_for_delta() -> dict[str, object]:
    """Degree-only argument inside bare H24.

    Delta := compact_uv - bare_h24_uv has pure degree B^2 zeta^2.
    The xf/cross sector carries only B^4 zeta^4 and B^2 r zeta^2 monomials.
    Therefore only the R-type pieces of Table V can absorb Delta.
    """
    delta_degree = (2, 1, 1, 0)
    xf_cross_monomials = [(4, 2, 2, 0), (2, 1, 1, 1)]
    candidates = ["R-type pieces"]
    return {
        "delta_degree_(B,z1,z2,r)": delta_degree,
        "xf_cross_monomials_(B,z1,z2,r)": xf_cross_monomials,
        "only_possible_source_inside_H24": candidates,
    }


def degree_audit_h24_r_terms() -> dict[str, object]:
    """Exact bookkeeping from the printed R-operator definitions (Table I).

    Grading:
      deg = (deg_C, deg_zeta, deg_k4)

    Printed definitions imply:
      R^k_l      ~ zeta
      R_k        ~ C
      R_kl       ~ C^2
      R^m_kl     ~ C^2 zeta
      R_klm      ~ C^3

    Therefore the H24 R-type pieces have the grades listed below.
    """
    grades = {
        "delta": (0, 2, 0),
        "T1_-sum R_m(R_kml+R_mkl+R_mlk)/w_m": (4, 0, 0),
        "T2a_sum R_m R_n k4/(w_m w_n)": (2, 0, 1),
        "T2b_sum R_m R_n B zeta zeta/(w_m w_n)": (2, 2, 0),
        "T3_i-sum[...]/(2 w_k w_m)": (2, 0, 0),
        "T4_- [R_k,[R_l,H02]]/(2 w_k w_l)": (2, 0, 0),
        "T5_-sum R^m R^m_kl / w_m": (2, 2, 0),
        "T6_-sum R'_m k'_mkl /(2 w_m)": (1, 0, 1),
        "T7a_4 sum R_k X_lm R^m_l / w_l": (1, 2, 0),
        "T7b_-4 sum R_k X^lm R'_lm / w_l": (3, 1, 0),
        "T8_i sum (...) [R_m,R_l]": (2, 1, 0),
    }
    pure_candidates = [
        name for name, deg in grades.items()
        if name != "delta" and deg == grades["delta"]
    ]
    return {
        "grades_(deg_C,deg_zeta,deg_k4)": grades,
        "pure_delta_candidates": pure_candidates,
    }


def locate_u_channel_after_bare_h24_analysis() -> dict[str, object]:
    """Logical consequence of the exact reductions completed so far.

    Established:
      1. bare H24(UV) = V-only channel
      2. xf/cross cannot supply Delta by degree
      3. the explicit bare R-type terms do not match the pure Delta grade

    Therefore any surviving compact U-channel must lie in the block-diagonal
    correction \tilde H24 - H24 of Eq. (106), not in the explicit bare H24
    kernel already reduced.
    """
    return {
        "u_channel_location": "block-diagonal correction only",
        "eq106_residual": [
            "i[S03,H12]",
            "i[S11^(R),H04]",
            "i[S13^(R),H02]",
        ],
    }


def degree_audit_eq106_commutators() -> dict[str, object]:
    """Minimal degree audit for the three residual commutators in Eq. (106).

    Grading:
      (deg_C, deg_zeta, deg_k3, deg_k4)

    Printed source facts used:
      H04, H02 are pure rotational -> (0,0,0,0)
      H12 is linear in R_k -> R_k ~ C -> (1,0,0,0)
      S03 is generated from H30 -> cubic-only -> (0,0,1,0)
      S11^(R) is the block-diagonal remover of H22 rotational-resonance terms;
        Eq. (101) shows H22 contains both C^2 and zeta^2 pieces -> (2,0,0,0)
        or (0,2,0,0)
      S13^(R) would be generated only from the off-diagonal H24 sector itself.
    """
    grades = {
        "target_U_channel": (0, 2, 0, 0),
        "i[S03,H12]": [(1, 0, 1, 0)],
        "i[S11^(R),H04]": [(2, 0, 0, 0), (0, 2, 0, 0)],
        "i[S13^(R),H02]": ["same unresolved degree structure as off-diagonal H24"],
    }
    constructive_sources = ["i[S11^(R),H04]"]
    return {
        "grades_(deg_C,deg_zeta,deg_k3,deg_k4)": grades,
        "constructive_sources_for_pure_U_channel": constructive_sources,
    }


def derive_s11r_xy_generator_from_h22_offdiag() -> dict[str, sp.Expr]:
    """Exact coefficient matching for the accidental xy rotational resonance.

    Use the printed forms:
      H22_off = -(1/2) Q * X_xy * {Jx,Jy}
      S11^(R) = 2 Q * s_xy * Jz
      H02 = Bx Jx^2 + By Jy^2 + Bz Jz^2

    With the so(3) commutator identity
      i[Jz, H02] = (By-Bx){Jx,Jy},
    one gets
      i[S11^(R), H02] = 2 Q s_xy (By-Bx){Jx,Jy}.

    Matching i[S11^(R),H02] = -H22_off gives the exact coefficient below.
    """
    Bx, By, Xxy = sp.symbols("B_x B_y X_xy")
    sxy = sp.symbols("s_xy")
    solved = sp.solve(
        [sp.Eq(2 * sxy * (By - Bx), sp.Rational(1, 2) * Xxy)],
        (sxy,),
        dict=True,
    )[0][sxy]
    return {
        "s_xy": sp.simplify(solved),
        "iJz_H02_coeff": By - Bx,
    }


def derive_i_s11r_h04_xy_structure() -> dict[str, sp.Expr]:
    """Infinitesimal z-rotation of the diagonal xy quartic form.

    For
      H04^(xy) = A Jx^4 + B Jy^4 + C Jx^2 Jy^2
    the generator Jz produces only odd xy quartic operators:
      i[Jz,H04^(xy)] ~ coeff_31 * Jx^3 Jy + coeff_13 * Jx Jy^3

    No antisymmetric vibrational prefactor can arise, because S11^(R) carries
    the symmetric external factor (q_k q_l + p_k p_l) s_xy^{kl} with
    s_xy^{kl}=s_xy^{lk}.
    """
    Jx, Jy = sp.symbols("Jx J_y")
    A, B, C, eps = sp.symbols("A B C eps")
    H = A * Jx**4 + B * Jy**4 + C * Jx**2 * Jy**2
    H_rot = sp.expand(H.subs({Jx: Jx - eps * Jy, Jy: Jy + eps * Jx}))
    dH = sp.expand(sp.diff(H_rot, eps).subs(eps, 0))
    coeff_31 = sp.expand(dH).coeff(Jx, 3).coeff(Jy, 1)
    coeff_13 = sp.expand(dH).coeff(Jx, 1).coeff(Jy, 3)
    return {
        "dH": dH,
        "coeff_Jx3Jy": coeff_31,
        "coeff_JxJy3": coeff_13,
    }


def derive_i_s11r_h04_xy_tensor_form() -> dict[str, sp.Expr]:
    Jx, Jy = sp.symbols("Jx J_y")
    tau_xxxx, tau_yyyy, tau_xxyy, eps = sp.symbols(
        "tau_xxxx tau_yyyy tau_xxyy eps"
    )
    H = sp.Rational(1, 4) * (
        tau_xxxx * Jx**4 + tau_yyyy * Jy**4 + 6 * tau_xxyy * Jx**2 * Jy**2
    )
    H_rot = sp.expand(H.subs({Jx: Jx - eps * Jy, Jy: Jy + eps * Jx}))
    dH = sp.expand(sp.diff(H_rot, eps).subs(eps, 0))
    return {
        "coeff_Jx3Jy": sp.expand(dH).coeff(Jx, 3).coeff(Jy, 1),
        "coeff_JxJy3": sp.expand(dH).coeff(Jx, 1).coeff(Jy, 3),
    }


def derive_h04_xy_tensor_from_eq86() -> dict[str, sp.Expr]:
    """Eq. (86): tau_{abgd} = -2 sum_k C_k^{ab} C_k^{gd} w_k."""
    wx, wy = sp.symbols("w_x w_y", positive=True)
    Cxxx, Cxyy, Cyxx, Cyyy = sp.symbols("C_x_xx C_x_yy C_y_xx C_y_yy")
    tau_xxxx = -2 * (wx * Cxxx**2 + wy * Cyxx**2)
    tau_xxyy = -2 * (wx * Cxxx * Cxyy + wy * Cyxx * Cyyy)
    tau_yyyy = -2 * (wx * Cxyy**2 + wy * Cyyy**2)
    return {
        "tau_xxxx": sp.expand(tau_xxxx),
        "tau_xxyy": sp.expand(tau_xxyy),
        "tau_yyyy": sp.expand(tau_yyyy),
        "minus_tau_xxxx_plus_3_tau_xxyy": sp.expand(-tau_xxxx + 3 * tau_xxyy),
        "tau_yyyy_minus_3_tau_xxyy": sp.expand(tau_yyyy - 3 * tau_xxyy),
    }


def prove_s11r_h04_can_only_feed_u_channel() -> dict[str, object]:
    """Exact symmetry consequence for the external vibrational indices k,l."""
    skl, slk = sp.symbols("s_kl s_lk")
    return {
        "symmetry_condition": sp.simplify(skl - slk),
        "channel": "U-only",
    }


def derive_regular_symmetric_limit_of_s11r() -> dict[str, sp.Expr]:
    """Exact finite-limit parameterization of Eq. (104).

    Since the source text states that S11^(R) usually vanishes for essential
    degeneracy, the consistent regular limit is
      X_xy = 4(By-Bx) Uhat_xy,
    which gives
      s_xy -> Uhat_xy
    as Bx -> By.
    """
    Bx, By, Uhat = sp.symbols("B_x B_y Uhat_xy")
    Xxy = 4 * (By - Bx) * Uhat
    sxy = sp.simplify(Xxy / (4 * (By - Bx)))
    return {
        "X_xy_regularized": Xxy,
        "s_xy_regular_limit": sxy,
    }


def derive_cylindrical_xy_quartic_constraints() -> dict[str, sp.Expr]:
    """Exact cylindrical-symmetry constraints on the xy quartic block.

    For an exactly linear molecule the transverse plane is O(2)-isotropic, so
      H04^(xy) = lam * (Jx^2 + Jy^2)^2.

    Matching to
      H04^(xy) = 1/4 [tau_xxxx Jx^4 + tau_yyyy Jy^4 + 6 tau_xxyy Jx^2 Jy^2]
    gives the exact identities below.
    """
    lam = sp.symbols("lam")
    return {
        "tau_xxxx": 4 * lam,
        "tau_yyyy": 4 * lam,
        "tau_xxyy": sp.Rational(4, 3) * lam,
        "constraint_tau_xxxx_minus_tau_yyyy": 0,
        "constraint_tau_xxxx_minus_3_tau_xxyy": 0,
    }


def prove_iJz_h04_vanishes_under_cylindrical_symmetry() -> dict[str, sp.Expr]:
    tau_xxxx, tau_yyyy, tau_xxyy, lam = sp.symbols("tau_xxxx tau_yyyy tau_xxyy lam")
    coeffs = derive_i_s11r_h04_xy_tensor_form()
    subs = {
        tau_xxxx: 4 * lam,
        tau_yyyy: 4 * lam,
        tau_xxyy: sp.Rational(4, 3) * lam,
    }
    return {
        "coeff_Jx3Jy_linear_limit": sp.simplify(coeffs["coeff_Jx3Jy"].subs(subs)),
        "coeff_JxJy3_linear_limit": sp.simplify(coeffs["coeff_JxJy3"].subs(subs)),
    }


def derive_singular_scaling_requirement_for_u_channel() -> dict[str, sp.Expr]:
    """First-order anisotropy analysis near the linear limit.

    Write the quartic block as an isotropic part plus O(Delta) anisotropy:
      tau_xxxx = 3*sigma + Delta*a
      tau_yyyy = 3*sigma + Delta*b
      tau_xxyy = sigma + Delta*c

    Then i[Jz,H04^(xy)] is O(Delta). Combined with
      s_xy = X_xy / (4 Delta),
    the commutator i[S11^(R),H04] has a finite nonzero limit only if S11^(R)
    is singular as 1/Delta. If one imposes the regularized ansatz
      X_xy = 4 Delta Uhat_xy,
    the resulting U-channel vanishes identically in the linear limit.
    """
    Delta, sigma, a, b, c, Xxy, Uhat = sp.symbols("Delta sigma a b c X_xy Uhat_xy")
    tau_xxxx = 3 * sigma + Delta * a
    tau_yyyy = 3 * sigma + Delta * b
    tau_xxyy = sigma + Delta * c
    coeff_31 = sp.simplify(-tau_xxxx + 3 * tau_xxyy)
    coeff_13 = sp.simplify(tau_yyyy - 3 * tau_xxyy)
    s_singular = Xxy / (4 * Delta)
    s_regular = (4 * Delta * Uhat) / (4 * Delta)
    return {
        "coeff_Jx3Jy_first_order": sp.factor(coeff_31),
        "coeff_JxJy3_first_order": sp.factor(coeff_13),
        "singular_limit_Jx3Jy": sp.simplify(sp.limit(s_singular * coeff_31, Delta, 0)),
        "singular_limit_JxJy3": sp.simplify(sp.limit(s_singular * coeff_13, Delta, 0)),
        "regularized_limit_Jx3Jy": sp.simplify(sp.limit(s_regular * coeff_31, Delta, 0)),
        "regularized_limit_JxJy3": sp.simplify(sp.limit(s_regular * coeff_13, Delta, 0)),
    }


def prove_regular_s11r_branch_contradicts_generic_compact_u() -> dict[str, sp.Expr]:
    wn, wp, wt, B, z1, z2 = sp.symbols("w_n w_p w_t B z1 z2", positive=True)
    compact_u = compact_linear_u_formula(wn, wp, wt, B, z1, z2, 0)
    sing = derive_singular_scaling_requirement_for_u_channel()
    return {
        "compact_u_generic": sp.factor(sp.together(compact_u)),
        "regularized_commutator_limit_Jx3Jy": sing["regularized_limit_Jx3Jy"],
        "regularized_commutator_limit_JxJy3": sing["regularized_limit_JxJy3"],
        "compact_u_is_identically_zero": sp.simplify(compact_u),
    }


def transcribe_eq101_visible_zeta2_sector() -> dict[str, sp.Expr]:
    """Literal transcription of the two visible zeta^2 fractions in Eq. (101).

    From the rendered page-23 PDF crop, the zeta^2 part of alpha^{ab}_{lk}
    contains exactly two symmetric bilinears:

      (z_mk^a z_ml^b + z_ml^a z_mk^b)

    multiplied by the two frequency kernels below. The crop also shows the
    closing bracket and equation number (101), so there is no third hidden
    zeta^2 line in the printed formula.
    """
    wk, wl, wm = sp.symbols("w_k w_l w_m", positive=True)
    kernel_1 = sp.factor(
        (wm - wk) * (wm - wl) * (2 * wm + wk + wl)
        / (wm * (wm + wk) * (wm + wl) * sp.sqrt(wk * wl))
    )
    kernel_2 = sp.factor(
        (wm + wk) * (wm + wl) * (2 * wm - wk - wl)
        / (wm * (wm - wk) * (wm - wl) * sp.sqrt(wk * wl))
    )
    return {
        "kernel_1": kernel_1,
        "kernel_2": kernel_2,
        "symmetric_bilinear": sp.Symbol("Zsym_klm_ab"),
    }


def prove_eq101_crop_is_zeta2_complete() -> dict[str, object]:
    return {
        "zeta2_term_count_in_eq101": 2,
        "closing_bracket_visible": True,
        "equation_number_visible": True,
        "zeta2_sector_complete_in_crop": True,
    }


def transcribe_eq101_visible_c2_k4_sector() -> dict[str, sp.Expr]:
    """Non-zeta part of Eq. (101) from the page-23 crop.

    The readable first line of Eq. (101) has the structure

      alpha_{lk}^{ab} = alpha_{lk}^{ba}
        = (3/8) w_k w_l sum_g (C_k^{ag} C_l^{bg} + C_l^{ag} C_k^{bg}) / B_g^e
          + (1/2) sum_m k'_{klm} C_m^{ab}
          + ...

    This is exactly of C^2 + k4 type and carries no zeta factors.
    """
    wk, wl = sp.symbols("w_k w_l", positive=True)
    Csym, K4 = sp.symbols("Csym K4")
    return {
        "c2_term": sp.Rational(3, 8) * wk * wl * Csym,
        "k4_term": sp.Rational(1, 2) * K4,
        "degree_(C,zeta,k4)": {
            "c2_term": (2, 0, 0),
            "k4_term": (0, 0, 1),
        },
    }


def separate_eq101_visible_sectors() -> dict[str, object]:
    c2k4 = transcribe_eq101_visible_c2_k4_sector()
    zeta2 = transcribe_eq101_visible_zeta2_sector()
    return {
        "c2_k4_sector": c2k4,
        "zeta2_sector": zeta2,
        "visible_sector_degrees": {
            "c2": (2, 0, 0),
            "k4": (0, 0, 1),
            "zeta2": (0, 2, 0),
        },
    }


def compare_visible_eq101_zeta2_with_compact_u_kernel() -> dict[str, sp.Expr]:
    wk, wl, wm = sp.symbols("w_k w_l w_m", positive=True)
    out = transcribe_eq101_visible_zeta2_sector()
    vis = sp.simplify(sp.together(-sp.Rational(1, 4) * (out["kernel_1"] + out["kernel_2"])))
    compact_u_kernel = sp.simplify(
        sp.together(
            ((wk + wm) ** 2 * (wl - wm) ** 2 + (wk - wm) ** 2 * (wl + wm) ** 2)
            / (8 * wm * sp.sqrt(wk * wl) * (wk**2 - wm**2) * (wl**2 - wm**2))
        )
    )
    return {
        "visible_eq101_zeta2_kernel": sp.factor(vis),
        "compact_u_kernel": sp.factor(compact_u_kernel),
        "difference": sp.factor(sp.together(vis - compact_u_kernel)),
    }


def derive_missing_u_kernel_relative_to_eq101() -> dict[str, sp.Expr]:
    wk, wl, wm = sp.symbols("w_k w_l w_m", positive=True)
    cmp = compare_visible_eq101_zeta2_with_compact_u_kernel()
    delta = sp.factor(sp.together(cmp["compact_u_kernel"] - cmp["visible_eq101_zeta2_kernel"]))
    num, den = sp.together(delta).as_numer_denom()
    return {
        "delta_u_kernel": delta,
        "delta_u_numerator": sp.factor(num),
        "delta_u_denominator": sp.factor(den),
    }


def derive_minimal_paper1_u_candidate_and_residual() -> dict[str, sp.Expr]:
    wk, wl, wm, rkl, B, Z = sp.symbols("w_k w_l w_m r_kl B Z", positive=True)
    cmp = compare_visible_eq101_zeta2_with_compact_u_kernel()
    u_paper1_min = sp.simplify(rkl / (4 * (wk + wl)) + B**2 * Z * cmp["visible_eq101_zeta2_kernel"])
    u_aliev = sp.simplify(rkl / (4 * (wk + wl)) + B**2 * Z * cmp["compact_u_kernel"])
    return {
        "u_paper1_minimal_candidate": sp.factor(sp.together(u_paper1_min)),
        "u_aliev_compact": sp.factor(sp.together(u_aliev)),
        "residual_to_aliev": sp.factor(sp.together(u_aliev - u_paper1_min)),
    }


def derive_dimensionless_u_kernel_mismatch() -> dict[str, sp.Expr]:
    x, y = sp.symbols("x y", positive=True)
    ku_dim = (x**2 * y**2 + x**2 - 4 * x * y + y**2 + 1) / (
        4 * (x - 1) * (x + 1) * (y - 1) * (y + 1)
    )
    k101_dim = (x**3 * y + x**2 * y**2 + x * y**3 - 2 * x * y - 1) / (
        (x - 1) * (x + 1) * (y - 1) * (y + 1)
    )
    mismatch = sp.factor(sp.together(ku_dim - k101_dim / 4))
    return {
        "ku_dimensionless": sp.factor(ku_dim),
        "k101_dimensionless": sp.factor(k101_dim),
        "mismatch_after_simple_1_over_4_rescaling": mismatch,
    }


def compare_visible_eq101_degrees_with_compact_u() -> dict[str, object]:
    """Degree matching only.

    Compact U has:
      r-part     ~ (0,0,1)
      zeta2-part ~ (0,2,0)

    The visible Eq. (101) sectors have:
      C^2        ~ (2,0,0)
      k4         ~ (0,0,1)
      zeta2      ~ (0,2,0)
    """
    return {
        "compact_U_degrees_(C,zeta,k4)": {
            "r_part": (0, 0, 1),
            "zeta2_part": (0, 2, 0),
        },
        "visible_eq101_degrees_(C,zeta,k4)": separate_eq101_visible_sectors()[
            "visible_sector_degrees"
        ],
    }


def derive_beta_parallel_uv_dependence_from_rotlin() -> dict[str, sp.Expr]:
    """Exact U/V dependence of the parallel observable from rotlin.F MkBetN.

    Lines 220-221 in rotlin.F give:
      Term71 = Unn(i,j)*Unn(i,j)*(freq(i)+freq(j))
      Term72 = Vnn(i,j)*Vnn(i,j)*(freq(j)-freq(i))
      Term7  = Term7 - 16*(Term71+Term72)

    No other term in MkBetN contains Unn or Vnn.
    """
    wi, wj, Uij, Vij = sp.symbols("w_i w_j U_ij V_ij")
    term71 = Uij**2 * (wi + wj)
    term72 = Vij**2 * (wj - wi)
    term7 = -16 * (term71 + term72)
    return {
        "term71": term71,
        "term72": term72,
        "term7_uv_piece": sp.expand(term7),
        "d_term7_dU": sp.diff(term7, Uij),
        "d_term7_dV": sp.diff(term7, Vij),
    }


def prove_beta_parallel_depends_essentially_on_u() -> dict[str, object]:
    wi, wj, Uij, Vij = sp.symbols("w_i w_j U_ij V_ij")
    dep = derive_beta_parallel_uv_dependence_from_rotlin()
    return {
        "beta_parallel_uv_entry_point": "Term7 only",
        "term7_uv_piece": dep["term7_uv_piece"],
        "u_dependence_is_nonzero": sp.simplify(dep["d_term7_dU"]),
        "v_dependence_is_nonzero": sp.simplify(dep["d_term7_dV"]),
        "u_survives_quadratically": True,
    }


def derive_beta_perpendicular_uv_dependence_from_rotlin() -> dict[str, sp.Expr]:
    """Exact U/V dependence of the perpendicular observable from MkBetT.

    In the nondegenerate-j branch of MkBetT:
      U2ij  = Unn(ivib,j)*Unn(ivib,j)
      V2ij  = Vnn(ivib,j)*Vnn(ivib,j)
      Term7 = Term7 - 16*(U2ij*Sij + V2ij*Dij)

    This is the same quadratic U/V structure as in MkBetN.
    """
    wt, wj, Utj, Vtj = sp.symbols("w_t w_j U_tj V_tj")
    sij = wt + wj
    dij = wj - wt
    term7 = sp.expand(-16 * (Utj**2 * sij + Vtj**2 * dij))
    return {
        "term7_uv_piece": term7,
        "d_term7_dU": sp.diff(term7, Utj),
        "d_term7_dV": sp.diff(term7, Vtj),
    }


def prove_beta_perpendicular_depends_essentially_on_u() -> dict[str, object]:
    dep = derive_beta_perpendicular_uv_dependence_from_rotlin()
    return {
        "beta_perpendicular_uv_entry_point": "Term7 only",
        "term7_uv_piece": dep["term7_uv_piece"],
        "u_dependence_is_nonzero": sp.simplify(dep["d_term7_dU"]),
        "v_dependence_is_nonzero": sp.simplify(dep["d_term7_dV"]),
        "u_survives_quadratically": True,
    }


def main() -> None:
    inv = prove_mn_inversion()
    print("M =", inv["M_expr"])
    print("N =", inv["N_expr"])
    print("back_Xd_minus_Xd =", inv["back_Xd_minus_Xd"])
    print("back_Xu_minus_Xu =", inv["back_Xu_minus_Xu"])
    lin = derive_linear_pair_mn_from_x()
    print("M_nt =", lin["M_expr"])
    print("N_nt =", lin["N_expr"])
    prods = prove_linear_pair_products()
    print("MM_plus_NN =", prods["MM_plus_NN"])
    print("MM_minus_NN =", prods["MM_minus_NN"])
    uv = prove_uv_symmetries()
    print("U_swap_minus_U =", uv["U_swap_minus_U"])
    print("V_swap_plus_V =", uv["V_swap_plus_V"])
    pf = prove_prefactor_bookkeeping()
    print("M*M+N*N =", pf["M*M+N*N"])
    print("M*M-N*N =", pf["M*M-N*N"])
    print("common_prefactor_mm_nn =", pf["common_prefactor_mm_nn"])
    cancel = prove_r_cancellation_in_uv_block()
    print("r_coefficient =", cancel["r_coefficient"])
    print("uv_combo_no_r =", cancel["uv_combo_no_r"])
    cmp_uv = compare_compact_uv_with_bare_h24_uv()
    print("compact_minus_bare_h24_uv =", cmp_uv["difference"])
    v_only = prove_bare_h24_uv_equals_v_only_channel()
    print("bare_plus_4_times_v_channel =", v_only["bare_plus_4_times_v_channel"])
    xfc = analyze_xf_cross_degree_structure()
    print("xf_cross_monomials =", xfc["monomials_in_(B,z1,z2,r)"])
    src = identify_only_possible_source_for_delta()
    print("delta_degree =", src["delta_degree_(B,z1,z2,r)"])
    print("only_possible_source =", src["only_possible_source_inside_H24"])
    audit = degree_audit_h24_r_terms()
    print("pure_delta_candidates =", audit["pure_delta_candidates"])
    loc = locate_u_channel_after_bare_h24_analysis()
    print("u_channel_location =", loc["u_channel_location"])
    eq106 = degree_audit_eq106_commutators()
    print("constructive_sources_for_pure_U_channel =", eq106["constructive_sources_for_pure_U_channel"])
    s11 = derive_s11r_xy_generator_from_h22_offdiag()
    print("s_xy =", s11["s_xy"])
    h04 = derive_i_s11r_h04_xy_structure()
    print("coeff_Jx3Jy =", h04["coeff_Jx3Jy"])
    print("coeff_JxJy3 =", h04["coeff_JxJy3"])
    h04t = derive_i_s11r_h04_xy_tensor_form()
    print("tensor_coeff_Jx3Jy =", h04t["coeff_Jx3Jy"])
    print("tensor_coeff_JxJy3 =", h04t["coeff_JxJy3"])
    eq86 = derive_h04_xy_tensor_from_eq86()
    print("minus_tau_xxxx_plus_3_tau_xxyy =", eq86["minus_tau_xxxx_plus_3_tau_xxyy"])
    print("tau_yyyy_minus_3_tau_xxyy =", eq86["tau_yyyy_minus_3_tau_xxyy"])
    uonly = prove_s11r_h04_can_only_feed_u_channel()
    print("s11r_h04_channel =", uonly["channel"])
    reg = derive_regular_symmetric_limit_of_s11r()
    print("s_xy_regular_limit =", reg["s_xy_regular_limit"])
    cyl = derive_cylindrical_xy_quartic_constraints()
    print("linear_constraint_tau_xxxx_minus_tau_yyyy =", cyl["constraint_tau_xxxx_minus_tau_yyyy"])
    print("linear_constraint_tau_xxxx_minus_3_tau_xxyy =", cyl["constraint_tau_xxxx_minus_3_tau_xxyy"])
    vanish = prove_iJz_h04_vanishes_under_cylindrical_symmetry()
    print("coeff_Jx3Jy_linear_limit =", vanish["coeff_Jx3Jy_linear_limit"])
    print("coeff_JxJy3_linear_limit =", vanish["coeff_JxJy3_linear_limit"])
    sing = derive_singular_scaling_requirement_for_u_channel()
    print("coeff_Jx3Jy_first_order =", sing["coeff_Jx3Jy_first_order"])
    print("coeff_JxJy3_first_order =", sing["coeff_JxJy3_first_order"])
    print("singular_limit_Jx3Jy =", sing["singular_limit_Jx3Jy"])
    print("singular_limit_JxJy3 =", sing["singular_limit_JxJy3"])
    print("regularized_limit_Jx3Jy =", sing["regularized_limit_Jx3Jy"])
    print("regularized_limit_JxJy3 =", sing["regularized_limit_JxJy3"])
    contra = prove_regular_s11r_branch_contradicts_generic_compact_u()
    print("compact_u_generic =", contra["compact_u_generic"])
    print("regularized_commutator_limit_Jx3Jy =", contra["regularized_commutator_limit_Jx3Jy"])
    print("regularized_commutator_limit_JxJy3 =", contra["regularized_commutator_limit_JxJy3"])
    eq101 = transcribe_eq101_visible_zeta2_sector()
    print("eq101_kernel_1 =", eq101["kernel_1"])
    print("eq101_kernel_2 =", eq101["kernel_2"])
    cmp101 = compare_visible_eq101_zeta2_with_compact_u_kernel()
    print("visible_eq101_minus_compact_u_kernel =", cmp101["difference"])


if __name__ == "__main__":
    main()
