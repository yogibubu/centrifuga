#!/usr/bin/env python3
"""Tests for the explicit readable Aliev linear families."""

from __future__ import annotations

import itertools

import sympy as sp

from linear_dv_aliev_terms import (
    build_explicit_aliev_betas,
    build_explicit_aliev_beta_breakdown,
    build_explicit_aliev_dv_model,
    build_explicit_aliev_families,
    build_explicit_aliev_L_model,
    build_explicit_aliev_uv_breakdown,
    make_linear_aliev_explicit_inputs,
    make_linear_aliev_explicit_inputs_from_mapping,
    resolve_linear_aliev_quartic_mode,
)


def test_explicit_aliev_families_expose_readable_equation_blocks() -> None:
    inputs = make_linear_aliev_explicit_inputs(
        B=sp.Symbol("B", positive=True),
        D_J=sp.Symbol("D_J", real=True),
        omega_parallel=sp.symbols("wn0:2", positive=True),
        omega_perpendicular=sp.symbols("wt0:2", positive=True),
        zeta_nt=[
            [sp.Symbol("z00", real=True), sp.Symbol("z01", real=True)],
            [sp.Symbol("z10", real=True), sp.Symbol("z11", real=True)],
        ],
        bxx_parallel=sp.symbols("Bxx0:2", real=True),
        k3_parallel=[
            [
                [sp.Symbol("k000", real=True), sp.Symbol("k001", real=True)],
                [sp.Symbol("k010", real=True), sp.Symbol("k011", real=True)],
            ],
            [
                [sp.Symbol("k100", real=True), sp.Symbol("k101", real=True)],
                [sp.Symbol("k110", real=True), sp.Symbol("k111", real=True)],
            ],
        ],
        k3_perp_pair=[
            [
                [sp.Symbol("ktt000", real=True), sp.Symbol("ktt001", real=True)],
                [sp.Symbol("ktt010", real=True), sp.Symbol("ktt011", real=True)],
            ],
            [
                [sp.Symbol("ktt100", real=True), sp.Symbol("ktt101", real=True)],
                [sp.Symbol("ktt110", real=True), sp.Symbol("ktt111", real=True)],
            ],
        ],
        k4_parallel=[
            [
                [
                    [sp.Symbol(f"k4_{a}{b}{c}{d}", real=True) for d in range(2)]
                    for c in range(2)
                ]
                for b in range(2)
            ]
            for a in range(2)
        ],
    )

    fam = build_explicit_aliev_families(inputs)

    assert "Bxx0" in str(fam.C_n[0])
    assert "sqrt(wn0)*sqrt(wt0)" in str(fam.X_upper_nt[(0, 0)]) or "sqrt(wn0*wt0)" in str(fam.X_upper_nt[(0, 0)])
    assert "wn0**2 + wt0**2" in str(fam.X_lower_nt[(0, 0)])
    assert "D_J" in str(fam.r_n[0])
    assert "k000" in str(fam.r_n[0])
    assert "z00*z10" in str(fam.F_upper_nn[(0, 1)])
    assert "wt0" in str(fam.F_lower_nn[(0, 1)])
    assert "ktt000" in str(fam.r_tt[(0, 0)])
    assert fam.L is not None
    assert "D_J**3" in str(fam.L)


def test_explicit_aliev_families_numeric_case_stays_finite_off_resonance() -> None:
    inputs = make_linear_aliev_explicit_inputs(
        B=0.39,
        D_J=2.1e-7,
        omega_parallel=[1330.0, 2349.0],
        omega_perpendicular=[667.0, 720.0],
        zeta_nt=[[0.22, 0.05], [0.03, 0.17]],
        bxx_parallel=[-0.015, 0.021],
        k3_parallel=[
            [[0.10, 0.02], [0.02, 0.03]],
            [[0.04, 0.01], [0.01, 0.05]],
        ],
        k3_perp_pair=[
            [[0.03, 0.02], [0.01, 0.04]],
            [[0.02, 0.01], [0.05, 0.03]],
        ],
        k4_parallel=[
            [[[0.01, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.02]]],
        ],
    )

    fam = build_explicit_aliev_families(inputs)
    vals = [
        fam.C_n[0],
        fam.X_upper_nt[(0, 0)],
        fam.X_lower_nt[(1, 1)],
        fam.r_n[1],
        fam.F_upper_nn[(0, 1)],
        fam.F_lower_nn[(0, 1)],
        fam.U_nn[(0, 1)],
        fam.V_nn[(0, 1)],
        fam.r_tt[(0, 1)],
        fam.L,
    ]
    for value in vals:
        assert value is not None
        assert bool(sp.sympify(value).is_finite)


def test_explicit_aliev_betas_include_readable_blocks_and_named_kernels() -> None:
    inputs = make_linear_aliev_explicit_inputs(
        B=sp.Symbol("B", positive=True),
        D_J=sp.Symbol("D_J", real=True),
        omega_parallel=sp.symbols("wn0:2", positive=True),
        omega_perpendicular=sp.symbols("wt0:2", positive=True),
        zeta_nt=[
            [sp.Symbol("z00", real=True), sp.Symbol("z01", real=True)],
            [sp.Symbol("z10", real=True), sp.Symbol("z11", real=True)],
        ],
        bxx_parallel=sp.symbols("Bxx0:2", real=True),
        k3_parallel=[
            [
                [sp.Symbol("k000", real=True), sp.Symbol("k001", real=True)],
                [sp.Symbol("k010", real=True), sp.Symbol("k011", real=True)],
            ],
            [
                [sp.Symbol("k100", real=True), sp.Symbol("k101", real=True)],
                [sp.Symbol("k110", real=True), sp.Symbol("k111", real=True)],
            ],
        ],
        k3_perp_pair=[
            [
                [sp.Symbol("ktt000", real=True), sp.Symbol("ktt001", real=True)],
                [sp.Symbol("ktt010", real=True), sp.Symbol("ktt011", real=True)],
            ],
            [
                [sp.Symbol("ktt100", real=True), sp.Symbol("ktt101", real=True)],
                [sp.Symbol("ktt110", real=True), sp.Symbol("ktt111", real=True)],
            ],
        ],
        pair_seed_perpendicular=[sp.Symbol("pt0", real=True), sp.Symbol("pt1", real=True)],
    )

    betas = build_explicit_aliev_betas(inputs)
    text_par = str(betas.beta_parallel[0])
    text_perp = str(betas.beta_perpendicular[0])
    assert "D_J" in text_par
    assert "3*wn0**2 + wt0**2" in text_par or "3*wn0**2 + wt1**2" in text_par
    assert "K_beta_parallel_mix_tail" not in text_par
    assert "k000" in text_par or "k001" in text_par
    assert "z00**2" in text_par or "z01**2" in text_par or "wt0 + wn0" in text_par
    assert "K_beta_perpendicular_mid_0" not in text_perp
    assert "K_beta_perpendicular_mix" not in text_perp
    assert "wt0**2" in text_perp
    assert "z00" in text_perp or "z10" in text_perp
    assert "B**2*z00**2" in text_perp or "B**2*z10**2" in text_perp or "wn0*wt0" in text_perp
    assert "ktt000" in text_perp or "ktt001" in text_perp or "D_J" in text_perp
    assert "r_nn" not in text_perp
    assert "k4_beta_" not in text_par
    assert "k4_beta_" not in text_perp


def test_explicit_aliev_beta_breakdown_sums_to_total() -> None:
    inputs = make_linear_aliev_explicit_inputs(
        B=0.39,
        D_J=2.1e-7,
        omega_parallel=[1330.0, 2349.0],
        omega_perpendicular=[667.0, 720.0],
        zeta_nt=[[0.22, 0.05], [0.03, 0.17]],
        bxx_parallel=[-0.015, 0.021],
        k3_parallel=[
            [[0.10, 0.02], [0.02, 0.03]],
            [[0.04, 0.01], [0.01, 0.05]],
        ],
        k3_perp_pair=[
            [[0.03, 0.02], [0.01, 0.04]],
            [[0.02, 0.01], [0.05, 0.03]],
        ],
        pair_seed_perpendicular=[0.0, 0.0],
        k4_parallel=[
            [[[0.01, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.02]]],
        ],
    )
    pieces = build_explicit_aliev_beta_breakdown(inputs)
    for idx, block in pieces.parallel.items():
        total = sum(v for k, v in block.items() if k != "total")
        assert sp.Abs(sp.simplify(total - block["total"])) == 0
    for idx, block in pieces.perpendicular.items():
        total = sum(v for k, v in block.items() if k != "total")
        assert sp.Abs(sp.simplify(total - block["total"])) == 0


def test_explicit_aliev_uv_breakdown_sums_to_uv_block() -> None:
    inputs = make_linear_aliev_explicit_inputs(
        B=0.39,
        D_J=2.1e-7,
        omega_parallel=[1330.0, 2349.0],
        omega_perpendicular=[667.0, 720.0],
        zeta_nt=[[0.22, 0.05], [0.03, 0.17]],
        bxx_parallel=[-0.015, 0.021],
        k3_parallel=[
            [[0.10, 0.02], [0.02, 0.03]],
            [[0.04, 0.01], [0.01, 0.05]],
        ],
        k3_perp_pair=[
            [[0.03, 0.02], [0.01, 0.04]],
            [[0.02, 0.01], [0.05, 0.03]],
        ],
        pair_seed_perpendicular=[0.0, 0.0],
    )
    beta = build_explicit_aliev_beta_breakdown(inputs)
    uv = build_explicit_aliev_uv_breakdown(inputs)
    for idx in uv.parallel:
        block = uv.parallel[idx]
        assert (
            sp.Abs(
                sp.simplify(
                    block["u_diag_total"] + block["u_offdiag"] + block["v_offdiag"] - block["uv_total"]
                )
            )
            == 0
        )
        assert abs(float(block["uv_total"] - beta.parallel[idx]["uv_block"])) < 1.0e-15
    for idx in uv.perpendicular:
        block = uv.perpendicular[idx]
        assert (
            sp.Abs(
                sp.simplify(
                    block["u_diag_total"] + block["u_offdiag"] + block["v_offdiag"] - block["uv_total"]
                )
            )
            == 0
        )
        assert abs(float(block["uv_total"] - beta.perpendicular[idx]["uv_block"])) < 1.0e-15


def test_explicit_aliev_dv_model_uses_aliev_state_factors() -> None:
    wn0, wn1 = sp.symbols("wn0:2", positive=True)
    wt0, wt1 = sp.symbols("wt0:2", positive=True)
    v0, v1, v2, v3 = sp.symbols("v0:4", real=True)
    inputs = make_linear_aliev_explicit_inputs(
        B=sp.Symbol("B", positive=True),
        D_J=sp.Symbol("D_J", real=True),
        omega_parallel=(wn0, wn1),
        omega_perpendicular=(wt0, wt1),
        zeta_nt=[
            [sp.Symbol("z00", real=True), sp.Symbol("z01", real=True)],
            [sp.Symbol("z10", real=True), sp.Symbol("z11", real=True)],
        ],
        bxx_parallel=sp.symbols("Bxx0:2", real=True),
        k3_parallel=[
            [
                [sp.Symbol("k000", real=True), sp.Symbol("k001", real=True)],
                [sp.Symbol("k010", real=True), sp.Symbol("k011", real=True)],
            ],
            [
                [sp.Symbol("k100", real=True), sp.Symbol("k101", real=True)],
                [sp.Symbol("k110", real=True), sp.Symbol("k111", real=True)],
            ],
        ],
        k3_perp_pair=[
            [
                [sp.Symbol("ktt000", real=True), sp.Symbol("ktt001", real=True)],
                [sp.Symbol("ktt010", real=True), sp.Symbol("ktt011", real=True)],
            ],
            [
                [sp.Symbol("ktt100", real=True), sp.Symbol("ktt101", real=True)],
                [sp.Symbol("ktt110", real=True), sp.Symbol("ktt111", real=True)],
            ],
        ],
        pair_seed_perpendicular=[sp.Symbol("pt0", real=True), sp.Symbol("pt1", real=True)],
    )

    model = build_explicit_aliev_dv_model(inputs)
    expr = model.value_for_state((v0, v1, v2, v3))
    text = str(expr)
    assert model.mode_kinds == ("parallel", "parallel", "perpendicular", "perpendicular")
    assert "D_J" in text
    assert "v0 + 1/2" in text
    assert "v1 + 1/2" in text
    assert "v2 + 1" in text
    assert "v3 + 1" in text


def test_explicit_aliev_L_model_exposes_optical_constant() -> None:
    inputs = make_linear_aliev_explicit_inputs(
        B=sp.Symbol("B", positive=True),
        D_J=sp.Symbol("D_J", real=True),
        omega_parallel=sp.symbols("wn0:2", positive=True),
        omega_perpendicular=sp.symbols("wt0:2", positive=True),
        zeta_nt=[
            [sp.Symbol("z00", real=True), sp.Symbol("z01", real=True)],
            [sp.Symbol("z10", real=True), sp.Symbol("z11", real=True)],
        ],
        bxx_parallel=sp.symbols("Bxx0:2", real=True),
        k3_parallel=[
            [
                [sp.Symbol("k000", real=True), sp.Symbol("k001", real=True)],
                [sp.Symbol("k010", real=True), sp.Symbol("k011", real=True)],
            ],
            [
                [sp.Symbol("k100", real=True), sp.Symbol("k101", real=True)],
                [sp.Symbol("k110", real=True), sp.Symbol("k111", real=True)],
            ],
        ],
        k3_perp_pair=[
            [
                [sp.Symbol("ktt000", real=True), sp.Symbol("ktt001", real=True)],
                [sp.Symbol("ktt010", real=True), sp.Symbol("ktt011", real=True)],
            ],
            [
                [sp.Symbol("ktt100", real=True), sp.Symbol("ktt101", real=True)],
                [sp.Symbol("ktt110", real=True), sp.Symbol("ktt111", real=True)],
            ],
        ],
        pair_seed_perpendicular=[sp.Symbol("pt0", real=True), sp.Symbol("pt1", real=True)],
        k4_parallel=[
            [
                [
                    [sp.Symbol(f"k4_{a}{b}{c}{d}", real=True) for d in range(2)]
                    for c in range(2)
                ]
                for b in range(2)
            ]
            for a in range(2)
        ],
    )
    model = build_explicit_aliev_L_model(inputs)
    text = str(model.value)
    assert "D_J**3" in text
    assert "k4_0000" in text or "k4_1111" in text
    assert model.mode_kinds == ("parallel", "parallel", "perpendicular", "perpendicular")
    assert len(model.mode_contributions) == 4
    assert model.mode_contributions[2] == 0
    assert model.mode_contributions[3] == 0
    assert sp.simplify(sum(model.mode_contributions) + model.shared_offset - model.value) == 0
    assert sp.simplify(sum(model.mode_contributions_with_shared_offset) - model.value) == 0


def test_explicit_aliev_discards_only_four_distinct_quartic_indices() -> None:
    k4_0123 = sp.Symbol("k4_0123", real=True)
    k4_0011 = sp.Symbol("k4_0011", real=True)
    inputs = make_linear_aliev_explicit_inputs(
        B=sp.Symbol("B", positive=True),
        D_J=sp.Symbol("D_J", real=True),
        omega_parallel=sp.symbols("wn0:4", positive=True),
        omega_perpendicular=(sp.Symbol("wt0", positive=True),),
        zeta_nt=[
            [sp.Symbol("z00", real=True)],
            [sp.Symbol("z10", real=True)],
            [sp.Symbol("z20", real=True)],
            [sp.Symbol("z30", real=True)],
        ],
        bxx_parallel=sp.symbols("Bxx0:4", real=True),
        k3_parallel=[
            [[0, 0, 0, 0] for _ in range(4)]
            for _ in range(4)
        ],
        k3_perp_pair=[
            [[0, 0, 0, 0]],
        ],
        k4_parallel=[
            [
                [
                    [
                        k4_0123 if (a, b, c, d) == (0, 1, 2, 3) else
                        k4_0011 if (a, b, c, d) == (0, 0, 1, 1) else
                        0
                        for d in range(4)
                    ]
                    for c in range(4)
                ]
                for b in range(4)
            ]
            for a in range(4)
        ],
    )

    fam = build_explicit_aliev_families(inputs)
    text = str(fam.L)
    assert fam.L is not None
    assert "k4_0011" in text
    assert "k4_0123" not in text


def test_reduced_three_index_quartics_are_accepted_for_betas_and_L() -> None:
    inputs = make_linear_aliev_explicit_inputs(
        B=sp.Symbol("B", positive=True),
        D_J=sp.Symbol("D_J", real=True),
        omega_parallel=sp.symbols("wn0:3", positive=True),
        omega_perpendicular=(sp.Symbol("wt0", positive=True),),
        zeta_nt=[
            [sp.Symbol("z00", real=True)],
            [sp.Symbol("z10", real=True)],
            [sp.Symbol("z20", real=True)],
        ],
        bxx_parallel=sp.symbols("Bxx0:3", real=True),
        k3_parallel=[
            [[0, 0, 0] for _ in range(3)]
            for _ in range(3)
        ],
        k3_perp_pair=[
            [[0, 0, 0]],
        ],
        pair_seed_perpendicular=[0],
        k4_reduced=[
            [
                [sp.Symbol("kred_000", real=True), sp.Symbol("kred_001", real=True), sp.Symbol("kred_002", real=True)],
                [sp.Symbol("kred_011", real=True), sp.Symbol("kred_011", real=True), sp.Symbol("kred_012", real=True)],
                [sp.Symbol("kred_022", real=True), sp.Symbol("kred_012", real=True), sp.Symbol("kred_022", real=True)],
            ],
            [
                [sp.Symbol("kred_100", real=True), sp.Symbol("kred_101", real=True), sp.Symbol("kred_102", real=True)],
                [sp.Symbol("kred_111", real=True), sp.Symbol("kred_111", real=True), sp.Symbol("kred_112", real=True)],
                [sp.Symbol("kred_122", real=True), sp.Symbol("kred_112", real=True), sp.Symbol("kred_122", real=True)],
            ],
            [
                [sp.Symbol("kred_200", real=True), sp.Symbol("kred_201", real=True), sp.Symbol("kred_202", real=True)],
                [sp.Symbol("kred_211", real=True), sp.Symbol("kred_211", real=True), sp.Symbol("kred_212", real=True)],
                [sp.Symbol("kred_222", real=True), sp.Symbol("kred_212", real=True), sp.Symbol("kred_222", real=True)],
            ],
        ],
    )

    betas = build_explicit_aliev_betas(inputs)
    l_model = build_explicit_aliev_L_model(inputs)
    par_text = str(betas.beta_parallel[0])
    perp_text = str(betas.beta_perpendicular[0])
    l_text = str(l_model.value)
    assert "kred_0" in par_text
    assert "kred_" in perp_text
    assert "kred_" in l_text


def test_full_and_reduced_quartic_representations_agree_on_repeated_index_patterns() -> None:
    full = [[[[0 for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)]
    for perm in set(itertools.permutations((0, 0, 1, 2))):
        full[perm[0]][perm[1]][perm[2]][perm[3]] = sp.Symbol("k0012", real=True)
    for perm in set(itertools.permutations((1, 1, 0, 2))):
        full[perm[0]][perm[1]][perm[2]][perm[3]] = sp.Symbol("k1102", real=True)
    for perm in set(itertools.permutations((2, 2, 2, 1))):
        full[perm[0]][perm[1]][perm[2]][perm[3]] = sp.Symbol("k2221", real=True)
    reduced = [
        [
            [0, 0, 0],
            [0, 0, sp.Symbol("k0012", real=True)],
            [0, 0, 0],
        ],
        [
            [0, 0, sp.Symbol("k1102", real=True)],
            [0, 0, 0],
            [0, 0, 0],
        ],
        [
            [0, 0, 0],
            [0, 0, sp.Symbol("k2221", real=True)],
            [0, 0, 0],
        ],
    ]
    inputs_full = make_linear_aliev_explicit_inputs(
        B=1,
        D_J=1,
        omega_parallel=(1, 2, 3),
        omega_perpendicular=(4,),
        zeta_nt=[[0], [0], [0]],
        bxx_parallel=(1, 1, 1),
        k3_parallel=[[[0, 0, 0] for _ in range(3)] for _ in range(3)],
        k3_perp_pair=[[[0, 0, 0]]],
        k4_parallel=full,
    )
    inputs_reduced = make_linear_aliev_explicit_inputs(
        B=1,
        D_J=1,
        omega_parallel=(1, 2, 3),
        omega_perpendicular=(4,),
        zeta_nt=[[0], [0], [0]],
        bxx_parallel=(1, 1, 1),
        k3_parallel=[[[0, 0, 0] for _ in range(3)] for _ in range(3)],
        k3_perp_pair=[[[0, 0, 0]]],
        k4_reduced=reduced,
    )
    fam_full = build_explicit_aliev_families(inputs_full)
    fam_reduced = build_explicit_aliev_families(inputs_reduced)
    assert sp.simplify(fam_full.L - fam_reduced.L) == 0


def test_resolve_linear_aliev_quartic_mode() -> None:
    assert resolve_linear_aliev_quartic_mode(quartic_mode="none") == "none"
    assert resolve_linear_aliev_quartic_mode(k4_reduced=[[[1]]], quartic_mode="auto") == "reduced"
    assert resolve_linear_aliev_quartic_mode(k4_parallel=[[[[1]]]], quartic_mode="auto") == "full"
    assert resolve_linear_aliev_quartic_mode(k4_parallel=[[[[1]]]], k4_reduced=[[[2]]], quartic_mode="auto") == "full"


def test_make_linear_aliev_inputs_from_mapping_respects_quartic_mode() -> None:
    payload = {
        "B": 1,
        "D_J": 2,
        "omega_parallel": [3, 4],
        "omega_perpendicular": [5],
        "zeta_nt": [[0], [0]],
        "bxx_parallel": [1, 1],
        "k3_parallel": [
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
        ],
        "k3_perp_pair": [
            [[0, 0]],
        ],
        "k4_reduced": [
            [[11, 12], [13, 14]],
            [[21, 22], [23, 24]],
        ],
        "k4_parallel": [
            [
                [[101, 0], [0, 0]],
                [[0, 0], [0, 0]],
            ],
            [
                [[0, 0], [0, 0]],
                [[0, 0], [0, 202]],
            ],
        ],
    }
    reduced = make_linear_aliev_explicit_inputs_from_mapping(payload, quartic_mode="reduced")
    full = make_linear_aliev_explicit_inputs_from_mapping(payload, quartic_mode="full")
    none = make_linear_aliev_explicit_inputs_from_mapping(payload, quartic_mode="none")
    assert reduced.k4_reduced is not None and reduced.k4_parallel is None
    assert full.k4_parallel is not None and full.k4_reduced is None
    assert none.k4_parallel is None and none.k4_reduced is None
