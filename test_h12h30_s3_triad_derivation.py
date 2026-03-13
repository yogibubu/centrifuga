from __future__ import annotations

from collections import defaultdict

import sympy as sp

from quartic_channels import (
    _H12H30_SCAFFOLD_COEFFS,
    _component_mix_h12h30,
    _h12h30_threeindex_triad_component_numerators,
    _h12h30_threeindex_triad_effective_weights,
    _h12h30_threeindex_triad_leading_mode_sums,
    _h12h30_threeindex_triad_cm_terms,
    _h12h30_threeindex_xxxx_triad_numerators,
    _sum_tau_dicts,
    _zero_tau,
)


def _build_test_tensors():
    n_modes = 4
    mu1 = sp.MutableDenseNDimArray.zeros(3, 3, n_modes)
    mu2 = sp.MutableDenseNDimArray.zeros(3, 3, n_modes, n_modes)
    phi3 = sp.MutableDenseNDimArray.zeros(n_modes, n_modes, n_modes)

    for a in range(3):
        for b in range(3):
            for i in range(n_modes):
                mu1[a, b, i] = sp.Float((a + 1) * 10 + (b + 1) + 0.1 * (i + 1))
                for j in range(n_modes):
                    mu2[a, b, i, j] = sp.Float((a + 1) * 100 + (b + 1) * 10 + (i + 1) + 0.01 * (j + 1))

    triads = {
        (0, 1, 2): sp.Float("1.25"),
        (0, 1, 3): sp.Float("-0.75"),
        (0, 2, 3): sp.Float("0.50"),
        (1, 2, 3): sp.Float("1.10"),
    }
    for (i, j, k), value in triads.items():
        for a, b, c in (
            (i, j, k),
            (i, k, j),
            (j, i, k),
            (j, k, i),
            (k, i, j),
            (k, j, i),
        ):
            phi3[a, b, c] = value

    omega = tuple(sp.Float(x) for x in (0.015, 0.020, 0.027, 0.033))
    return mu1, mu2, phi3, omega


def _ordered_loop_reference(mu1, mu2, phi3, omega):
    pair = defaultdict(lambda: sp.Integer(0))
    addition = defaultdict(lambda: sp.Integer(0))
    n_modes = mu1.shape[2]
    for i in range(n_modes):
        for j in range(n_modes):
            for k in range(n_modes):
                if i == j or j == k or i == k:
                    continue
                phi_tri = phi3[i, j, k]
                if abs(float(phi_tri)) <= 1.0e-14:
                    continue
                pair_denom = (omega[i] + omega[j]) * (omega[j] + omega[k]) * (omega[i] + omega[k])
                for key in _zero_tau():
                    mix = _component_mix_h12h30(mu1, mu2, i, j, key)
                    pair[key] += (
                        _H12H30_SCAFFOLD_COEFFS["tri_ijk_over_pair_sums"] * phi_tri * mix / pair_denom
                    )
                    addition[key] += (
                        _H12H30_SCAFFOLD_COEFFS["tri_ijk_over_w_pair_sums"] * phi_tri * mix / (omega[i] * pair_denom)
                    )
    return dict(pair), dict(addition)


def test_unordered_triad_form_matches_ordered_solver_sum():
    mu1, mu2, phi3, omega = _build_test_tensors()
    ref_pair, ref_addition = _ordered_loop_reference(mu1, mu2, phi3, omega)
    triads = _h12h30_threeindex_triad_cm_terms(mu1, mu2, phi3, omega)

    triad_pair = _sum_tau_dicts(*(block["S3"] for block in triads.values()))
    triad_addition = _sum_tau_dicts(*(block["S3_addition"] for block in triads.values()))

    for key in _zero_tau():
        assert abs(float(sp.N(triad_pair[key] - ref_pair.get(key, sp.Integer(0))))) < 1.0e-4
        assert abs(float(sp.N(triad_addition[key] - ref_addition.get(key, sp.Integer(0))))) < 1.0e-4


def test_single_triad_total_is_s3_plus_addition():
    mu1, mu2, phi3, omega = _build_test_tensors()
    triads = _h12h30_threeindex_triad_cm_terms(mu1, mu2, phi3, omega)
    block = triads[(0, 1, 2)]
    recombined = _sum_tau_dicts(block["S3"], block["S3_addition"])
    for key in _zero_tau():
        assert sp.simplify(recombined[key] - block["three_index"][key]) == 0


def test_xxxx_single_triad_numerators_match_closed_form():
    mu1, mu2, _phi3, omega = _build_test_tensors()
    nums = _h12h30_threeindex_xxxx_triad_numerators(mu1, mu2, omega)
    i, j, k = 0, 1, 2
    A_i = mu1[0, 0, i]
    A_j = mu1[0, 0, j]
    A_k = mu1[0, 0, k]
    B_i = mu2[0, 0, i, i]
    B_j = mu2[0, 0, j, j]
    B_k = mu2[0, 0, k, k]

    expected_n0 = (
        A_i * (B_j + B_k)
        + A_j * (B_i + B_k)
        + A_k * (B_i + B_j)
    )
    expected_n1 = (
        A_i * (B_j + B_k) / omega[i]
        + A_j * (B_i + B_k) / omega[j]
        + A_k * (B_i + B_j) / omega[k]
    )

    assert abs(float(sp.N(nums[(i, j, k)]["N0_xxxx"] - expected_n0))) < 1.0e-10
    assert abs(float(sp.N(nums[(i, j, k)]["N1_xxxx"] - expected_n1))) < 1.0e-10


def test_generic_component_numerators_reconstruct_single_triad_block():
    mu1, mu2, phi3, omega = _build_test_tensors()
    triads = _h12h30_threeindex_triad_cm_terms(mu1, mu2, phi3, omega)
    nums = _h12h30_threeindex_triad_component_numerators(mu1, mu2, omega)
    triad = (0, 1, 2)
    phi_tri = phi3[triad]

    for key in _zero_tau():
        comp = nums[triad][key]
        reconstructed_s3 = sp.simplify(
            _H12H30_SCAFFOLD_COEFFS["tri_ijk_over_pair_sums"] * phi_tri * comp["N0"] / comp["D_pair"]
        )
        reconstructed_add = sp.simplify(
            _H12H30_SCAFFOLD_COEFFS["tri_ijk_over_w_pair_sums"] * phi_tri * comp["N1"] / comp["D_pair"]
        )
        assert abs(float(sp.N(reconstructed_s3 - triads[triad]["S3"][key]))) < 1.0e-4
        assert abs(float(sp.N(reconstructed_add - triads[triad]["S3_addition"][key]))) < 1.0e-4


def test_effective_weight_rewrites_three_index_block():
    mu1, mu2, phi3, omega = _build_test_tensors()
    triads = _h12h30_threeindex_triad_cm_terms(mu1, mu2, phi3, omega)
    nums = _h12h30_threeindex_triad_component_numerators(mu1, mu2, omega)
    rho = _h12h30_threeindex_triad_effective_weights(mu1, mu2, omega)
    triad = (0, 1, 2)
    phi_tri = phi3[triad]

    for key in _zero_tau():
        comp = nums[triad][key]
        rebuilt = sp.simplify(
            phi_tri
            * (_H12H30_SCAFFOLD_COEFFS["tri_ijk_over_pair_sums"] + _H12H30_SCAFFOLD_COEFFS["tri_ijk_over_w_pair_sums"] * rho[triad][key])
            * comp["N0"]
            / comp["D_pair"]
        )
        assert abs(float(sp.N(rebuilt - triads[triad]["three_index"][key]))) < 1.0e-4


def test_effective_weight_is_weighted_mean_of_inverse_frequencies():
    mu1, mu2, _phi3, omega = _build_test_tensors()
    triad = (0, 1, 2)
    rho = _h12h30_threeindex_triad_effective_weights(mu1, mu2, omega)
    lead = _h12h30_threeindex_triad_leading_mode_sums(mu1, mu2, omega)

    for key in _zero_tau():
        comp = lead[triad][key]
        rebuilt_rho = sp.simplify(
            (comp["Ci"] / omega[0] + comp["Cj"] / omega[1] + comp["Ck"] / omega[2])
            / (comp["Ci"] + comp["Cj"] + comp["Ck"])
        )
        assert abs(float(sp.N(rebuilt_rho - rho[triad][key]))) < 1.0e-10
