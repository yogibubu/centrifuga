#!/usr/bin/env python3
"""Minimal quartic channel builder for reconstruction.

``H12H30`` is implemented as an explicit scaffold reconstruction split into
the same one-mode / two-mode / three-mode sectors used in the appendix of
``paper2.tex``:

- ``S1``: diagonal cubic block ``Phi_iii``
- ``S2``: semi-diagonal cubic block ``Phi_iij`` / ``Phi_ijj``
- ``S3``: genuine three-index cubic block ``Phi_ijk``

``H30H30`` remains a placeholder and is still diagnostic only.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable
import warnings

import sympy as sp
import numpy as np
from h30h30_resonance import regularized_two_level_term

MON_KEYS = ("xxxx", "yyyy", "zzzz", "xxyy", "xxzz", "yyzz")
_PLACEHOLDER_WARNED: set[str] = set()
WILSON_TAU_AU_TO_CMINV = 4.0 * 219474.6313705
_H12H30_SCAFFOLD_COEFFS = {
    "diag_iii_over_w5": sp.Float("8.30447968e-06"),
    "sd_iij_over_w2_w_wp": sp.Float("-4.61259401e-02"),
    "sd_iij_over_w_wp_2wipj": sp.Float("6.49157973"),
    "sd_iij_over_w_wp_wi2wj": sp.Float("8.74757939"),
    "tri_ijk_over_pair_sums": sp.Float("-4.29204642e+01"),
    "tri_ijk_over_w_pair_sums": sp.Float("3.15816267e-01"),
}
_H12H30_DIAG_SCAFFOLDS = ("diag_iii_over_w5",)
_H12H30_SEMIDIAG_SCAFFOLDS = (
    "sd_iij_over_w2_w_wp",
    "sd_iij_over_w_wp_2wipj",
    "sd_iij_over_w_wp_wi2wj",
)
_H12H30_TRI_SCAFFOLDS = (
    "tri_ijk_over_pair_sums",
    "tri_ijk_over_w_pair_sums",
)
_H30H30_DIAG1_COEFFS = {
    "tau_xxxx": -sp.Rational(1, 1200),
    "tau_yyyy": -sp.Rational(1, 270000),
    "tau_zzzz": -sp.Rational(1, 675),
    "tau_xxyy": -sp.Rational(1, 9000),
    "tau_xxzz": -sp.Rational(1, 450),
    "tau_yyzz": -sp.Rational(1, 6750),
}
_H30H30_DIAG1_III_III_CORR_COEFFS = {
    "tau_xxxx": sp.Rational(95380905500501926630542832453, 78597476697155953731211100160000),
    "tau_yyyy": sp.Rational(4665533883647099384253680209777, 38407728618788958519503177318400000),
    "tau_zzzz": sp.Rational(2938992038611330760549838559979, 2389355529692276679534356547502080),
    "tau_xxyy": sp.Rational(1607740160042717136437130066887045259597407, 1246055980446161799287711984926400286661017600),
    "tau_xxzz": sp.Rational(2110666378664113721263895466152755623451, 387696403325576518640091138025254247464960),
    "tau_yyzz": -sp.Rational(1315576067630219637513598955718127721039429, 2576142871015368516864322807338654849564672000),
}
_H30H30_DIAG1_III_III_ROTMIX_COEFFS = {
    "tau_xxxx": sp.Integer(0),
    "tau_yyyy": sp.Integer(0),
    "tau_zzzz": sp.Integer(0),
    "tau_xxyy": -sp.Rational(529, 12441600),
    "tau_xxzz": -sp.Rational(3, 5120000),
    "tau_yyzz": -sp.Rational(121, 1382400),
}
_H30H30_DIAG1_III_IIJ0_COEFFS = {
    "tau_xxxx": -sp.Rational(227363455692979534858168034909, 23579243009146786119363330048000),
    "tau_yyyy": -sp.Rational(132870342778744056191079919357, 93525313195102983407881113600000),
    "tau_zzzz": -sp.Rational(7203214178712084856197679706731, 716806658907683003860306964250624),
    "tau_xxyy": -sp.Rational(256014843720190531580320168166841827914919, 26701199580989181413308113962708577571307520),
    "tau_xxzz": -sp.Rational(357058524421360253321553700426262738557, 9572750699396951077533114519142080184320),
    "tau_yyzz": sp.Rational(7588749350269126232385665625360171839875777, 1932107153261526387648242105503991137173504000),
}
_H30H30_DIAG0_III_IIJ1_COEFFS = {
    "tau_xxxx": sp.Rational(712454982922014923900500951, 535891886571517866349166592000),
    "tau_yyyy": -sp.Rational(45508853414388670487647554779, 561151879170617900447286681600000),
    "tau_zzzz": -sp.Rational(16613915503968890164344558437161, 63000585255558076511159791779840000),
    "tau_xxyy": sp.Rational(94984702792100712326341262521802248841, 333764994762364767666351424533857219641344),
    "tau_xxzz": sp.Rational(4756903441008869353372586963090568356939, 1570170433468584900492369109002279702233088),
    "tau_yyzz": -sp.Rational(147893492940862692381337950801113843314157, 217362054741921718610427236869199002932019200),
}
_H30H30_DIAG0_III_IIJ2_COEFFS = {
    "tau_xxxx": -sp.Rational(438532066996440782413516299313, 265266483852901343842837463040000),
    "tau_yyyy": -sp.Rational(9636185244522385908224693217616021, 8641738939227515666888214896640000000),
    "tau_zzzz": -sp.Rational(106392049509917391552004970086462217, 60480561845335753450713400108646400000),
    "tau_xxyy": -sp.Rational(127558197189119260754299070794511739907303717, 127039301131425089693005010963199404225986560000),
    "tau_xxzz": -sp.Rational(2028035868070401130846736371427011472281933, 588813912550719337684638415875854888337408000),
    "tau_yyzz": sp.Rational(235383200534512692877933907870493374206066871, 573835824518673337131527905334685367740530688000),
}
_H30H30_IIJ_IIJ1_COEFFS = {
    "tau_xxxx": -sp.Rational(1423325721574754350122972529, 1768443225686008958952249753600),
    "tau_yyyy": -sp.Rational(2342819248469263597431749047349, 1080217367403439458361026862080000),
    "tau_zzzz": -sp.Rational(3346599684038788746878574958109, 6300058525555807651115979177984000),
    "tau_xxyy": -sp.Rational(3811404382454666303729694206747223606395, 3504532445004830060496689957605500806234112),
    "tau_xxzz": -sp.Rational(970133581163586025009826597520948258205, 785085216734292450246184554501139851116544),
    "tau_yyzz": -sp.Rational(377655083952895771689907883880675869233811, 695558575174149499553367157981436809382461440),
}
_H30H30_IIJ_IIJ2_COEFFS = {
    "tau_xxxx": sp.Rational(105323423687644952657628091, 175440796199008825292881920000),
    "tau_yyyy": sp.Rational(873641643321250912126568126898883753, 129626084088412735003323223449600000000),
    "tau_zzzz": sp.Rational(1117694852646885108524117612727467, 114546518646469230020290530508800000),
    "tau_xxyy": sp.Rational(72941392926741425861045260010844192597882121, 23525796505819461054260187215407297078886400000),
    "tau_xxzz": sp.Rational(64540823071388631974630915045526119703559, 20444927519122199225161056106800516956160000),
    "tau_yyzz": -sp.Rational(22372449244727659421033485146368876653681, 664161833933649695754083223766996953403392000),
}


def _zero_tau() -> Dict[str, sp.Expr]:
    return {f"tau_{key}": sp.Integer(0) for key in MON_KEYS}


def _complete_tau(partial: Dict[str, sp.Expr]) -> Dict[str, sp.Expr]:
    base = _zero_tau()
    for key, val in partial.items():
        base[key] = sp.simplify(val)
    return base


def _warn_placeholder(channel: str) -> None:
    if channel in _PLACEHOLDER_WARNED:
        return
    warnings.warn(
        f"{channel} currently uses a minimal placeholder formula in quartic_channels.py; "
        "it is not yet the benchmark-faithful Paper 2 implementation.",
        RuntimeWarning,
        stacklevel=2,
    )
    _PLACEHOLDER_WARNED.add(channel)


def _component_mix_h12h30(mu1, mu2, i: int, j: int, key: str):
    if key == "tau_xxxx":
        return mu1[0, 0, i] * mu2[0, 0, j, j]
    if key == "tau_yyyy":
        return mu1[1, 1, i] * mu2[1, 1, j, j]
    if key == "tau_zzzz":
        return mu1[2, 2, i] * mu2[2, 2, j, j]
    if key == "tau_xxyy":
        return (
            mu1[0, 0, i] * mu2[1, 1, j, j]
            + mu1[0, 1, i] * mu2[0, 1, j, j]
            + mu1[0, 1, i] * mu2[1, 0, j, j]
            + mu1[1, 0, i] * mu2[0, 1, j, j]
            + mu1[1, 0, i] * mu2[1, 0, j, j]
            + mu1[1, 1, i] * mu2[0, 0, j, j]
        )
    if key == "tau_xxzz":
        return (
            mu1[0, 0, i] * mu2[2, 2, j, j]
            + mu1[0, 2, i] * mu2[0, 2, j, j]
            + mu1[0, 2, i] * mu2[2, 0, j, j]
            + mu1[2, 0, i] * mu2[0, 2, j, j]
            + mu1[2, 0, i] * mu2[2, 0, j, j]
            + mu1[2, 2, i] * mu2[0, 0, j, j]
        )
    if key == "tau_yyzz":
        return (
            mu1[1, 1, i] * mu2[2, 2, j, j]
            + mu1[1, 2, i] * mu2[1, 2, j, j]
            + mu1[1, 2, i] * mu2[2, 1, j, j]
            + mu1[2, 1, i] * mu2[1, 2, j, j]
            + mu1[2, 1, i] * mu2[2, 1, j, j]
            + mu1[2, 2, i] * mu2[1, 1, j, j]
        )
    raise KeyError(key)


def channel_h12h12(mu1: sp.MutableDenseNDimArray, omega: Iterable[float]) -> Dict[str, sp.Expr]:
    omega = list(omega)
    n_modes = len(omega)
    tau = defaultdict(lambda: sp.Integer(0))
    for k in range(n_modes):
        w = sp.Rational(1, 8) / omega[k] ** 2
        tau["tau_xxxx"] += mu1[0, 0, k] ** 2 * w
        tau["tau_yyyy"] += mu1[1, 1, k] ** 2 * w
        tau["tau_zzzz"] += mu1[2, 2, k] ** 2 * w
        tau["tau_xxyy"] += (mu1[0, 0, k] * mu1[1, 1, k]) * w
        tau["tau_xxzz"] += (mu1[0, 0, k] * mu1[2, 2, k]) * w
        tau["tau_yyzz"] += (mu1[1, 1, k] * mu1[2, 2, k]) * w
    return _complete_tau(tau)


def channel_h22(mu2: sp.MutableDenseNDimArray, omega: Iterable[float], hbar: sp.Symbol) -> Dict[str, sp.Expr]:
    omega = list(omega)
    n_modes = len(omega)
    tau = defaultdict(lambda: sp.Integer(0))
    coef = sp.Rational(1, 32) * hbar
    for k in range(n_modes):
        for l in range(n_modes):
            symmetry = sp.Integer(2) if k == l else sp.Integer(1)
            w = coef * symmetry / (omega[k] * omega[l])
            xx = mu2[0, 0, k, l]
            yy = mu2[1, 1, k, l]
            zz = mu2[2, 2, k, l]
            xy = mu2[0, 1, k, l]
            yx = mu2[1, 0, k, l]
            xz = mu2[0, 2, k, l]
            zx = mu2[2, 0, k, l]
            yz = mu2[1, 2, k, l]
            zy = mu2[2, 1, k, l]

            tau["tau_xxxx"] += xx**2 * w
            tau["tau_yyyy"] += yy**2 * w
            tau["tau_zzzz"] += zz**2 * w
            tau["tau_xxyy"] += (xx * yy + xy * xy + xy * yx + yx * yx) * w
            tau["tau_xxzz"] += (xx * zz + xz * xz + xz * zx + zx * zx) * w
            tau["tau_yyzz"] += (yy * zz + yz * yz + yz * zy + zy * zy) * w
    return _complete_tau(tau)


def bilinear_modepair_from_mu1(
    mu1: sp.MutableDenseNDimArray,
    inertia0: sp.MutableDenseNDimArray,
) -> sp.MutableDenseNDimArray:
    """Build the bilinear mode-pair closure B(mu1) = mu1 I mu1 + mu1 I mu1."""
    mu1_np = np.asarray(mu1.tolist(), dtype=float)
    inertia0_np = np.asarray(inertia0.tolist(), dtype=float)
    n_modes = mu1_np.shape[2]
    bilinear = np.zeros((3, 3, n_modes, n_modes), dtype=float)
    for k in range(n_modes):
        for l in range(n_modes):
            bilinear[:, :, k, l] = (
                mu1_np[:, :, k] @ inertia0_np @ mu1_np[:, :, l]
                + mu1_np[:, :, l] @ inertia0_np @ mu1_np[:, :, k]
            )
    flat = [sp.Float(float(x)) for x in bilinear.reshape(-1)]
    return sp.MutableDenseNDimArray(flat, bilinear.shape)


def channel_h22_decomposed(
    bilinear: sp.MutableDenseNDimArray,
    intrinsic: sp.MutableDenseNDimArray,
    omega: Iterable[float],
    hbar: sp.Expr,
) -> dict[str, Dict[str, sp.Expr]]:
    """Return H22[B,B], H22[B,N], H22[N,N], and the assembled total."""
    total = sp.MutableDenseNDimArray(
        [sp.simplify(bilinear[idx] - intrinsic[idx]) for idx in np.ndindex(bilinear.shape)],
        bilinear.shape,
    )

    bb = channel_h22(bilinear, omega, hbar)
    bn_tensor = sp.MutableDenseNDimArray(
        [sp.simplify((bilinear[idx] + intrinsic[idx]) / 2) for idx in np.ndindex(bilinear.shape)],
        bilinear.shape,
    )
    # Use polarization on the quadratic form to isolate the mixed interference:
    #   Q(B-N) = Q(B) - 2 cross(B,N) + Q(N)
    total_tau = channel_h22(total, omega, hbar)
    nn = channel_h22(intrinsic, omega, hbar)
    cross = {
        key: sp.simplify((bb[key] + nn[key] - total_tau[key]) / 2)
        for key in bb
    }
    return {
        "total": total_tau,
        "bilinear": bb,
        "intrinsic": nn,
        "cross": cross,
    }


def channel_h12h30(mu1, mu2, phi3: sp.MutableDenseNDimArray, omega: Iterable[float], hbar: sp.Symbol, seed=None, exact_calibration=False):
    """Return the total mixed ``H12H30`` scaffold reconstruction."""
    return channel_h12h30_decomposed(mu1, mu2, phi3, omega, hbar)["total"]


def _sum_tau_dicts(*taus: Dict[str, sp.Expr]) -> Dict[str, sp.Expr]:
    total = _zero_tau()
    for tau in taus:
        for key, val in tau.items():
            total[key] = sp.simplify(total[key] + val)
    return total


def _scale_tau(tau: Dict[str, sp.Expr], factor: sp.Expr) -> Dict[str, sp.Expr]:
    return _complete_tau({key: sp.simplify(factor * val) for key, val in tau.items()})


def _h12h30_threeindex_triad_cm_terms(
    mu1,
    mu2,
    phi3: sp.MutableDenseNDimArray,
    omega: Iterable[float],
) -> dict[tuple[int, int, int], dict[str, Dict[str, sp.Expr]]]:
    """Build the tri-index ``H12H30`` terms triad by triad in cm^-1 units.

    For each unordered triad ``{i,j,k}``, the current solver can be written as
    the sum over the six ordered permutations. This helper makes that
    decomposition explicit so the ``S3`` block can be discussed at the level of
    denominator families on a single triad.
    """
    n_modes = mu1.shape[2]
    triad_terms: dict[tuple[int, int, int], dict[str, Dict[str, sp.Expr]]] = {}
    for i, j, k in combinations(range(n_modes), 3):
        phi_tri = phi3[i, j, k]
        if abs(float(phi_tri)) <= 1.0e-14:
            continue
        pair = defaultdict(lambda: sp.Integer(0))
        addition = defaultdict(lambda: sp.Integer(0))
        common_pair = (omega[i] + omega[j]) * (omega[j] + omega[k]) * (omega[i] + omega[k])
        for a, b, _c in (
            (i, j, k),
            (i, k, j),
            (j, i, k),
            (j, k, i),
            (k, i, j),
            (k, j, i),
        ):
            for key in _zero_tau():
                mix = _component_mix_h12h30(mu1, mu2, a, b, key)
                pair[key] += (
                    _H12H30_SCAFFOLD_COEFFS["tri_ijk_over_pair_sums"]
                    * phi_tri
                    * mix
                    / common_pair
                )
                addition[key] += (
                    _H12H30_SCAFFOLD_COEFFS["tri_ijk_over_w_pair_sums"]
                    * phi_tri
                    * mix
                    / (omega[a] * common_pair)
                )
        triad_terms[(i, j, k)] = {
            "S3": _complete_tau(pair),
            "S3_addition": _complete_tau(addition),
            "three_index": _sum_tau_dicts(_complete_tau(pair), _complete_tau(addition)),
        }
    return triad_terms


def _h12h30_threeindex_xxxx_triad_numerators(
    mu1,
    mu2,
    omega: Iterable[float],
) -> dict[tuple[int, int, int], dict[str, sp.Expr]]:
    """Return the explicit ``tau_xxxx`` triad numerators used by the solver.

    For an unordered triad ``{i,j,k}``, the current ``S3`` block can be written
    as a common pair-sum denominator times an unweighted numerator

    ``N0_xxxx({i,j,k}) = sum_perm M_xxxx(a,b)``

    while ``S3_addition`` uses the same denominator and the weighted numerator

    ``N1_xxxx({i,j,k}) = sum_perm M_xxxx(a,b) / w_a``.
    """
    n_modes = mu1.shape[2]
    numerators: dict[tuple[int, int, int], dict[str, sp.Expr]] = {}
    for i, j, k in combinations(range(n_modes), 3):
        n0 = sp.Integer(0)
        n1 = sp.Integer(0)
        for a, b, _c in (
            (i, j, k),
            (i, k, j),
            (j, i, k),
            (j, k, i),
            (k, i, j),
            (k, j, i),
        ):
            mix = _component_mix_h12h30(mu1, mu2, a, b, "tau_xxxx")
            n0 += mix
            n1 += mix / omega[a]
        numerators[(i, j, k)] = {
            "N0_xxxx": sp.simplify(n0),
            "N1_xxxx": sp.simplify(n1),
            "D_pair": sp.simplify((omega[i] + omega[j]) * (omega[j] + omega[k]) * (omega[i] + omega[k])),
        }
    return numerators


def _h12h30_threeindex_triad_component_numerators(
    mu1,
    mu2,
    omega: Iterable[float],
) -> dict[tuple[int, int, int], dict[str, Dict[str, sp.Expr]]]:
    """Return generic triad numerators ``N0_mu`` and ``N1_mu`` for all components.

    For every unordered triad ``{i,j,k}`` and every tensor component ``mu``,
    the current solver uses the same pair-sum denominator
    ``D_pair(i,j,k)``. The distinction between the appendix ``S3`` block and
    its weighted solver deformation is entirely in the numerators:

    - ``N0_mu``: unweighted sum over ordered-pair mixings
    - ``N1_mu``: the same sum weighted by ``1 / w_a``
    """
    n_modes = mu1.shape[2]
    numerators: dict[tuple[int, int, int], dict[str, Dict[str, sp.Expr]]] = {}
    for i, j, k in combinations(range(n_modes), 3):
        by_component: dict[str, Dict[str, sp.Expr]] = {}
        d_pair = sp.simplify((omega[i] + omega[j]) * (omega[j] + omega[k]) * (omega[i] + omega[k]))
        for key in _zero_tau():
            n0 = sp.Integer(0)
            n1 = sp.Integer(0)
            for a, b, _c in (
                (i, j, k),
                (i, k, j),
                (j, i, k),
                (j, k, i),
                (k, i, j),
                (k, j, i),
            ):
                mix = _component_mix_h12h30(mu1, mu2, a, b, key)
                n0 += mix
                n1 += mix / omega[a]
            by_component[key] = {
                "N0": sp.simplify(n0),
                "N1": sp.simplify(n1),
                "D_pair": d_pair,
            }
        numerators[(i, j, k)] = by_component
    return numerators


def _h12h30_threeindex_triad_effective_weights(
    mu1,
    mu2,
    omega: Iterable[float],
) -> dict[tuple[int, int, int], dict[str, sp.Expr]]:
    """Return the effective inverse-frequency weights ``rho_mu = N1_mu / N0_mu``.

    This gives the cleanest solver-faithful rewriting of the three-index block:

    ``tau_mu^three_index = Phi_ijk / D_pair * (c_pair + c_add * rho_mu) * N0_mu``

    whenever ``N0_mu`` does not vanish.
    """
    numerators = _h12h30_threeindex_triad_component_numerators(mu1, mu2, omega)
    ratios: dict[tuple[int, int, int], dict[str, sp.Expr]] = {}
    for triad, by_component in numerators.items():
        comp_ratios: dict[str, sp.Expr] = {}
        for key, comp in by_component.items():
            if sp.simplify(comp["N0"]) == 0:
                comp_ratios[key] = sp.nan
            else:
                comp_ratios[key] = sp.simplify(comp["N1"] / comp["N0"])
        ratios[triad] = comp_ratios
    return ratios


def _h12h30_threeindex_triad_leading_mode_sums(
    mu1,
    mu2,
    omega: Iterable[float],
) -> dict[tuple[int, int, int], dict[str, Dict[str, sp.Expr]]]:
    """Split ``N0_mu`` and ``N1_mu`` into the three leading-mode partial sums.

    For a triad ``{i,j,k}``, define

    ``C_mu^(i) = sum_{b in {j,k}} M_mu(i,b)``

    and analogously for ``j`` and ``k``. Then

    ``N0_mu = C_mu^(i) + C_mu^(j) + C_mu^(k)``

    and

    ``N1_mu = C_mu^(i)/w_i + C_mu^(j)/w_j + C_mu^(k)/w_k``.
    """
    n_modes = mu1.shape[2]
    by_triad: dict[tuple[int, int, int], dict[str, Dict[str, sp.Expr]]] = {}
    for i, j, k in combinations(range(n_modes), 3):
        per_component: dict[str, Dict[str, sp.Expr]] = {}
        for key in _zero_tau():
            ci = _component_mix_h12h30(mu1, mu2, i, j, key) + _component_mix_h12h30(mu1, mu2, i, k, key)
            cj = _component_mix_h12h30(mu1, mu2, j, i, key) + _component_mix_h12h30(mu1, mu2, j, k, key)
            ck = _component_mix_h12h30(mu1, mu2, k, i, key) + _component_mix_h12h30(mu1, mu2, k, j, key)
            per_component[key] = {
                "Ci": sp.simplify(ci),
                "Cj": sp.simplify(cj),
                "Ck": sp.simplify(ck),
                "N0": sp.simplify(ci + cj + ck),
                "N1": sp.simplify(ci / omega[i] + cj / omega[j] + ck / omega[k]),
            }
        by_triad[(i, j, k)] = per_component
    return by_triad


def _h12h30_scaffold_terms(
    mu1,
    mu2,
    phi3: sp.MutableDenseNDimArray,
    omega: Iterable[float],
) -> dict[str, Dict[str, sp.Expr]]:
    """Build explicit ``H12H30`` scaffold terms in the internal tau scale."""
    tau_cm_terms = {name: defaultdict(lambda: sp.Integer(0)) for name in _H12H30_SCAFFOLD_COEFFS}
    n_modes = mu1.shape[2]
    for i in range(n_modes):
        phi_diag = phi3[i, i, i]
        if abs(float(phi_diag)) > 1.0e-14:
            denom = omega[i] ** 5
            coeff = _H12H30_SCAFFOLD_COEFFS["diag_iii_over_w5"]
            for key in _zero_tau():
                tau_cm_terms["diag_iii_over_w5"][key] += (
                    coeff * phi_diag * _component_mix_h12h30(mu1, mu2, i, i, key) / denom
                )
        for j in range(n_modes):
            if i == j:
                continue
            phi_sd = phi3[i, i, j]
            if abs(float(phi_sd)) > 1.0e-14:
                denoms = {
                    "sd_iij_over_w2_w_wp": omega[i] ** 2 * omega[j] * (omega[i] + omega[j]),
                    "sd_iij_over_w_wp_2wipj": omega[i] * (omega[i] + omega[j]) * (2 * omega[i] + omega[j]),
                    "sd_iij_over_w_wp_wi2wj": omega[i] * (omega[i] + omega[j]) * (omega[i] + 2 * omega[j]),
                }
                for name, denom in denoms.items():
                    coeff = _H12H30_SCAFFOLD_COEFFS[name]
                    for key in _zero_tau():
                        tau_cm_terms[name][key] += (
                            coeff * phi_sd * _component_mix_h12h30(mu1, mu2, i, j, key) / denom
                        )
    triad_terms = _h12h30_threeindex_triad_cm_terms(mu1, mu2, phi3, omega)
    for triad in triad_terms.values():
        for key, val in triad["S3"].items():
            tau_cm_terms["tri_ijk_over_pair_sums"][key] += val
        for key, val in triad["S3_addition"].items():
            tau_cm_terms["tri_ijk_over_w_pair_sums"][key] += val
    tau_au_terms = {}
    for name, tau_cm in tau_cm_terms.items():
        tau_au_terms[name] = _complete_tau(
            {key: sp.simplify(val / WILSON_TAU_AU_TO_CMINV) for key, val in tau_cm.items()}
        )
    return tau_au_terms


def _h30h30_diag1_iii_iii(
    mu1,
    phi3: sp.MutableDenseNDimArray,
    omega: Iterable[float],
    hbar: sp.Symbol,
) -> Dict[str, sp.Expr]:
    """Leading diagonal cubic-cubic scaffold ``D^(0)_{iii,iii}``.

    This is the first non-placeholder restoration step for ``H30H30``. It
    matches the exact one-mode symbolic scaling ``Phi_iii^2 / omega_i^7`` and
    uses the componentwise one-mode coefficients recovered from the BCH engine.
    """
    omega = list(omega)
    n_modes = mu1.shape[2]
    tau = defaultdict(lambda: sp.Integer(0))
    for i in range(n_modes):
        phi_sq = phi3[i, i, i] ** 2
        denom = omega[i] ** 7
        tau["tau_xxxx"] += _H30H30_DIAG1_COEFFS["tau_xxxx"] * hbar * phi_sq * mu1[0, 0, i] ** 2 / denom
        tau["tau_yyyy"] += _H30H30_DIAG1_COEFFS["tau_yyyy"] * hbar * phi_sq * mu1[1, 1, i] ** 2 / denom
        tau["tau_zzzz"] += _H30H30_DIAG1_COEFFS["tau_zzzz"] * hbar * phi_sq * mu1[2, 2, i] ** 2 / denom
        tau["tau_xxyy"] += _H30H30_DIAG1_COEFFS["tau_xxyy"] * hbar * phi_sq * mu1[0, 0, i] * mu1[1, 1, i] / denom
        tau["tau_xxzz"] += _H30H30_DIAG1_COEFFS["tau_xxzz"] * hbar * phi_sq * mu1[0, 0, i] * mu1[2, 2, i] / denom
        tau["tau_yyzz"] += _H30H30_DIAG1_COEFFS["tau_yyzz"] * hbar * phi_sq * mu1[1, 1, i] * mu1[2, 2, i] / denom
    return _complete_tau(tau)


def _h30h30_diag1_iii_iii_correction_candidate(
    mu1,
    phi3: sp.MutableDenseNDimArray,
    omega: Iterable[float],
    hbar: sp.Symbol,
) -> Dict[str, sp.Expr]:
    """Second diagonal-family candidate for ``D^(1)_{iii,iii}``.

    This uses the same one-mode structural class ``Phi_iii^2 / omega_i^7`` as
    the leading diagonal scaffold, but with the componentwise appendix
    coefficients associated with ``D^(1)_{iii,iii}``. It is kept separate so we
    can test whether the oversized diagonal core is really missing a second
    diagonal family rather than a resonance treatment.
    """
    omega = list(omega)
    n_modes = mu1.shape[2]
    tau = defaultdict(lambda: sp.Integer(0))
    for i in range(n_modes):
        phi_sq = phi3[i, i, i] ** 2
        denom = omega[i] ** 7
        tau["tau_xxxx"] += _H30H30_DIAG1_III_III_CORR_COEFFS["tau_xxxx"] * hbar * phi_sq * mu1[0, 0, i] ** 2 / denom
        tau["tau_yyyy"] += _H30H30_DIAG1_III_III_CORR_COEFFS["tau_yyyy"] * hbar * phi_sq * mu1[1, 1, i] ** 2 / denom
        tau["tau_zzzz"] += _H30H30_DIAG1_III_III_CORR_COEFFS["tau_zzzz"] * hbar * phi_sq * mu1[2, 2, i] ** 2 / denom
        tau["tau_xxyy"] += _H30H30_DIAG1_III_III_CORR_COEFFS["tau_xxyy"] * hbar * phi_sq * mu1[0, 0, i] * mu1[1, 1, i] / denom
        tau["tau_xxzz"] += _H30H30_DIAG1_III_III_CORR_COEFFS["tau_xxzz"] * hbar * phi_sq * mu1[0, 0, i] * mu1[2, 2, i] / denom
        tau["tau_yyzz"] += _H30H30_DIAG1_III_III_CORR_COEFFS["tau_yyzz"] * hbar * phi_sq * mu1[1, 1, i] * mu1[2, 2, i] / denom
    return _complete_tau(tau)


def _h30h30_diag1_iii_iii_rotmix_candidate(
    mu1,
    phi3: sp.MutableDenseNDimArray,
    omega: Iterable[float],
    hbar: sp.Symbol,
) -> Dict[str, sp.Expr]:
    """Mixed-only rotational-mixing candidate for ``D^(1)_{iii,iii}``.

    This candidate is extracted from the restricted symbolic probe
    ``iii-only/full-rot - iii-only/diag-rot``. It keeps the same scalar
    ``Phi_iii^2 / omega_i^7`` class as the leading diagonal core, but carries
    support only on the mixed quartic tensor components.
    """
    omega = list(omega)
    n_modes = mu1.shape[2]
    tau = defaultdict(lambda: sp.Integer(0))
    for i in range(n_modes):
        phi_sq = phi3[i, i, i] ** 2
        denom = omega[i] ** 7
        tau["tau_xxyy"] += _H30H30_DIAG1_III_III_ROTMIX_COEFFS["tau_xxyy"] * hbar * phi_sq * mu1[0, 0, i] * mu1[1, 1, i] / denom
        tau["tau_xxzz"] += _H30H30_DIAG1_III_III_ROTMIX_COEFFS["tau_xxzz"] * hbar * phi_sq * mu1[0, 0, i] * mu1[2, 2, i] / denom
        tau["tau_yyzz"] += _H30H30_DIAG1_III_III_ROTMIX_COEFFS["tau_yyzz"] * hbar * phi_sq * mu1[1, 1, i] * mu1[2, 2, i] / denom
    return _complete_tau(tau)


def _h30h30_placeholder_residual(
    mu1,
    phi3: sp.MutableDenseNDimArray,
    omega: Iterable[float],
) -> Dict[str, sp.Expr]:
    """Current placeholder residual kept explicit for diagnostics."""
    omega = list(omega)
    tau = defaultdict(lambda: sp.Integer(0))
    n_modes = mu1.shape[2]
    for i in range(n_modes):
        for j in range(n_modes):
            tau["tau_xxxx"] += phi3[i, j, j] * mu1[0, 0, i] * mu1[0, 0, j] / (omega[i] + omega[j] + 1)
            tau["tau_yyyy"] += phi3[i, j, j] * mu1[1, 1, i] * mu1[1, 1, j] / (omega[i] + omega[j] + 1)
            tau["tau_zzzz"] += phi3[i, j, j] * mu1[2, 2, i] * mu1[2, 2, j] / (omega[i] + omega[j] + 1)
    return _complete_tau(tau)


def _h30h30_diag1_iii_iij_0(
    mu1,
    phi3: sp.MutableDenseNDimArray,
    omega: Iterable[float],
    hbar: sp.Symbol,
) -> Dict[str, sp.Expr]:
    """First semi-diagonal scaffold guess for ``D^(0;1)_{iii,iij}``.

    This is a scaffold-first reconstruction step motivated by the H2O
    diagnostics and by the leading appendix coefficient for
    ``D^(0;1)_{iii,iij}``. It keeps the expected high-order frequency scaling
    explicit while remaining separate from the diagonal baseline.
    """
    omega = list(omega)
    n_modes = mu1.shape[2]
    tau = defaultdict(lambda: sp.Integer(0))
    for i in range(n_modes):
        for j in range(n_modes):
            if i == j:
                continue
            phi_mix = phi3[i, i, i] * phi3[i, i, j]
            denom = omega[i] ** 5 * omega[j] ** 2
            coeff_xx = mu1[0, 0, i] * mu1[0, 0, j]
            coeff_yy = mu1[1, 1, i] * mu1[1, 1, j]
            coeff_zz = mu1[2, 2, i] * mu1[2, 2, j]
            tau["tau_xxxx"] += _H30H30_DIAG1_III_IIJ0_COEFFS["tau_xxxx"] * hbar * phi_mix * coeff_xx / denom
            tau["tau_yyyy"] += _H30H30_DIAG1_III_IIJ0_COEFFS["tau_yyyy"] * hbar * phi_mix * coeff_yy / denom
            tau["tau_zzzz"] += _H30H30_DIAG1_III_IIJ0_COEFFS["tau_zzzz"] * hbar * phi_mix * coeff_zz / denom
            tau["tau_xxyy"] += _H30H30_DIAG1_III_IIJ0_COEFFS["tau_xxyy"] * hbar * phi_mix * (coeff_xx + coeff_yy) / (2 * denom)
            tau["tau_xxzz"] += _H30H30_DIAG1_III_IIJ0_COEFFS["tau_xxzz"] * hbar * phi_mix * (coeff_xx + coeff_zz) / (2 * denom)
            tau["tau_yyzz"] += _H30H30_DIAG1_III_IIJ0_COEFFS["tau_yyzz"] * hbar * phi_mix * (coeff_yy + coeff_zz) / (2 * denom)
    return _complete_tau(tau)


def _h30h30_diag0_iii_iij_1_candidate(
    mu1,
    phi3: sp.MutableDenseNDimArray,
    omega: Iterable[float],
    hbar: sp.Symbol,
) -> Dict[str, sp.Expr]:
    """Candidate pair-sum scaffold for ``D^(0;1)_{iii,iij}``.

    This is not yet promoted into the channel total. It is kept separate so the
    first ``iii,iij`` family beyond ``diag_1_iii_iij_0`` can be inspected
    numerically without contaminating the partially restored channel.
    """
    omega = list(omega)
    n_modes = mu1.shape[2]
    tau = defaultdict(lambda: sp.Integer(0))
    for i in range(n_modes):
        for j in range(n_modes):
            if i == j:
                continue
            phi_mix = phi3[i, i, i] * phi3[i, i, j]
            denom = omega[i] ** 4 * omega[j] * (omega[i] + omega[j]) ** 2
            coeff_xx = mu1[0, 0, i] * mu1[0, 0, j]
            coeff_yy = mu1[1, 1, i] * mu1[1, 1, j]
            coeff_zz = mu1[2, 2, i] * mu1[2, 2, j]
            tau["tau_xxxx"] += _H30H30_DIAG0_III_IIJ1_COEFFS["tau_xxxx"] * hbar * phi_mix * coeff_xx / denom
            tau["tau_yyyy"] += _H30H30_DIAG0_III_IIJ1_COEFFS["tau_yyyy"] * hbar * phi_mix * coeff_yy / denom
            tau["tau_zzzz"] += _H30H30_DIAG0_III_IIJ1_COEFFS["tau_zzzz"] * hbar * phi_mix * coeff_zz / denom
            tau["tau_xxyy"] += _H30H30_DIAG0_III_IIJ1_COEFFS["tau_xxyy"] * hbar * phi_mix * (coeff_xx + coeff_yy) / (2 * denom)
            tau["tau_xxzz"] += _H30H30_DIAG0_III_IIJ1_COEFFS["tau_xxzz"] * hbar * phi_mix * (coeff_xx + coeff_zz) / (2 * denom)
            tau["tau_yyzz"] += _H30H30_DIAG0_III_IIJ1_COEFFS["tau_yyzz"] * hbar * phi_mix * (coeff_yy + coeff_zz) / (2 * denom)
    return _complete_tau(tau)


def _h30h30_diag0_iii_iij_2_resonance_candidate(
    mu1,
    phi3: sp.MutableDenseNDimArray,
    omega: Iterable[float],
    hbar: sp.Symbol,
) -> Dict[str, sp.Expr]:
    """Signed-denominator candidate for ``D^(0;2)_{iii,iij}``.

    This candidate is kept *out* of the operative channel total. Its purpose is
    to expose the first plausible near-resonant family carrying a signed factor
    ``2*omega_i - omega_j`` so that the Martin/2x2 machinery has a meaningful
    scaffold-level target once numerator units are fully validated.
    """
    omega = list(omega)
    n_modes = mu1.shape[2]
    tau = defaultdict(lambda: sp.Integer(0))
    for i in range(n_modes):
        for j in range(n_modes):
            if i == j:
                continue
            phi_mix = phi3[i, i, i] * phi3[i, i, j]
            detuning = 2 * omega[i] - omega[j]
            denom = omega[i] ** 3 * omega[j] * (omega[i] + omega[j]) * detuning * (2 * omega[i] + omega[j])
            coeff_xx = mu1[0, 0, i] * mu1[0, 0, j]
            coeff_yy = mu1[1, 1, i] * mu1[1, 1, j]
            coeff_zz = mu1[2, 2, i] * mu1[2, 2, j]
            tau["tau_xxxx"] += _H30H30_DIAG0_III_IIJ2_COEFFS["tau_xxxx"] * hbar * phi_mix * coeff_xx / denom
            tau["tau_yyyy"] += _H30H30_DIAG0_III_IIJ2_COEFFS["tau_yyyy"] * hbar * phi_mix * coeff_yy / denom
            tau["tau_zzzz"] += _H30H30_DIAG0_III_IIJ2_COEFFS["tau_zzzz"] * hbar * phi_mix * coeff_zz / denom
            tau["tau_xxyy"] += _H30H30_DIAG0_III_IIJ2_COEFFS["tau_xxyy"] * hbar * phi_mix * (coeff_xx + coeff_yy) / (2 * denom)
            tau["tau_xxzz"] += _H30H30_DIAG0_III_IIJ2_COEFFS["tau_xxzz"] * hbar * phi_mix * (coeff_xx + coeff_zz) / (2 * denom)
            tau["tau_yyzz"] += _H30H30_DIAG0_III_IIJ2_COEFFS["tau_yyzz"] * hbar * phi_mix * (coeff_yy + coeff_zz) / (2 * denom)
    return _complete_tau(tau)


def _h30h30_diag0_iii_iij_2_regularized_preview(
    mu1,
    phi3: sp.MutableDenseNDimArray,
    omega: Iterable[float],
    hbar: sp.Symbol,
    *,
    metric_center: float = 1.0,
    metric_width: float = 0.05,
    level_shift: float = 1.0e-3,
    method: str = "tanh",
) -> Dict[str, sp.Expr]:
    """Regularized preview for the signed ``D^(0;2)_{iii,iij}`` candidate.

    This stays outside the operative channel total. It is only meant to show
    how the Martin/2x2 machinery would modify the first signed-denominator
    family once numerator units are considered stable enough.
    """
    omega = list(omega)
    n_modes = mu1.shape[2]
    tau = defaultdict(lambda: sp.Integer(0))
    for i in range(n_modes):
        for j in range(n_modes):
            if i == j:
                continue
            phi_mix = phi3[i, i, i] * phi3[i, i, j]
            detuning = 2 * omega[i] - omega[j]
            rest = omega[i] ** 3 * omega[j] * (omega[i] + omega[j]) * (2 * omega[i] + omega[j])
            coeffs = (
                ("tau_xxxx", _H30H30_DIAG0_III_IIJ2_COEFFS["tau_xxxx"], mu1[0, 0, i] * mu1[0, 0, j]),
                ("tau_yyyy", _H30H30_DIAG0_III_IIJ2_COEFFS["tau_yyyy"], mu1[1, 1, i] * mu1[1, 1, j]),
                ("tau_zzzz", _H30H30_DIAG0_III_IIJ2_COEFFS["tau_zzzz"], mu1[2, 2, i] * mu1[2, 2, j]),
                ("tau_xxyy", _H30H30_DIAG0_III_IIJ2_COEFFS["tau_xxyy"], (mu1[0, 0, i] * mu1[0, 0, j] + mu1[1, 1, i] * mu1[1, 1, j]) / 2),
                ("tau_xxzz", _H30H30_DIAG0_III_IIJ2_COEFFS["tau_xxzz"], (mu1[0, 0, i] * mu1[0, 0, j] + mu1[2, 2, i] * mu1[2, 2, j]) / 2),
                ("tau_yyzz", _H30H30_DIAG0_III_IIJ2_COEFFS["tau_yyzz"], (mu1[1, 1, i] * mu1[1, 1, j] + mu1[2, 2, i] * mu1[2, 2, j]) / 2),
            )
            for key, coeff, mix in coeffs:
                numerator_like = sp.simplify(coeff * hbar * phi_mix * mix / rest)
                try:
                    reg = regularized_two_level_term(
                        float(numerator_like),
                        float(detuning),
                        metric_center=metric_center,
                        metric_width=metric_width,
                        level_shift_cm=level_shift,
                        method=method,
                    )
                    tau[key] += sp.Float(reg)
                except (TypeError, ValueError):
                    tau[key] += sp.simplify(numerator_like / detuning)
    return _complete_tau(tau)


def _h30h30_iij_iij_1_candidate(
    mu1,
    phi3: sp.MutableDenseNDimArray,
    omega: Iterable[float],
    hbar: sp.Symbol,
) -> Dict[str, sp.Expr]:
    """First ``iij,iij`` scaffold candidate with pure-power denominator."""
    omega = list(omega)
    n_modes = mu1.shape[2]
    tau = defaultdict(lambda: sp.Integer(0))
    for i in range(n_modes):
        for j in range(n_modes):
            if i == j:
                continue
            phi_sq = phi3[i, i, j] ** 2
            denom = omega[i] ** 4 * omega[j] ** 3
            coeff_xx = mu1[0, 0, i] * mu1[0, 0, j]
            coeff_yy = mu1[1, 1, i] * mu1[1, 1, j]
            coeff_zz = mu1[2, 2, i] * mu1[2, 2, j]
            tau["tau_xxxx"] += _H30H30_IIJ_IIJ1_COEFFS["tau_xxxx"] * hbar * phi_sq * coeff_xx / denom
            tau["tau_yyyy"] += _H30H30_IIJ_IIJ1_COEFFS["tau_yyyy"] * hbar * phi_sq * coeff_yy / denom
            tau["tau_zzzz"] += _H30H30_IIJ_IIJ1_COEFFS["tau_zzzz"] * hbar * phi_sq * coeff_zz / denom
            tau["tau_xxyy"] += _H30H30_IIJ_IIJ1_COEFFS["tau_xxyy"] * hbar * phi_sq * (coeff_xx + coeff_yy) / (2 * denom)
            tau["tau_xxzz"] += _H30H30_IIJ_IIJ1_COEFFS["tau_xxzz"] * hbar * phi_sq * (coeff_xx + coeff_zz) / (2 * denom)
            tau["tau_yyzz"] += _H30H30_IIJ_IIJ1_COEFFS["tau_yyzz"] * hbar * phi_sq * (coeff_yy + coeff_zz) / (2 * denom)
    return _complete_tau(tau)


def _h30h30_iij_iij_2_candidate(
    mu1,
    phi3: sp.MutableDenseNDimArray,
    omega: Iterable[float],
    hbar: sp.Symbol,
) -> Dict[str, sp.Expr]:
    """Second ``iij,iij`` scaffold candidate with pair-sum denominator."""
    omega = list(omega)
    n_modes = mu1.shape[2]
    tau = defaultdict(lambda: sp.Integer(0))
    for i in range(n_modes):
        for j in range(n_modes):
            if i == j:
                continue
            phi_sq = phi3[i, i, j] ** 2
            denom = omega[i] ** 3 * omega[j] ** 2 * (omega[i] + omega[j]) ** 2
            coeff_xx = mu1[0, 0, i] * mu1[0, 0, j]
            coeff_yy = mu1[1, 1, i] * mu1[1, 1, j]
            coeff_zz = mu1[2, 2, i] * mu1[2, 2, j]
            tau["tau_xxxx"] += _H30H30_IIJ_IIJ2_COEFFS["tau_xxxx"] * hbar * phi_sq * coeff_xx / denom
            tau["tau_yyyy"] += _H30H30_IIJ_IIJ2_COEFFS["tau_yyyy"] * hbar * phi_sq * coeff_yy / denom
            tau["tau_zzzz"] += _H30H30_IIJ_IIJ2_COEFFS["tau_zzzz"] * hbar * phi_sq * coeff_zz / denom
            tau["tau_xxyy"] += _H30H30_IIJ_IIJ2_COEFFS["tau_xxyy"] * hbar * phi_sq * (coeff_xx + coeff_yy) / (2 * denom)
            tau["tau_xxzz"] += _H30H30_IIJ_IIJ2_COEFFS["tau_xxzz"] * hbar * phi_sq * (coeff_xx + coeff_zz) / (2 * denom)
            tau["tau_yyzz"] += _H30H30_IIJ_IIJ2_COEFFS["tau_yyzz"] * hbar * phi_sq * (coeff_yy + coeff_zz) / (2 * denom)
    return _complete_tau(tau)


def channel_h30h30_decomposed(
    mu1,
    phi3: sp.MutableDenseNDimArray,
    omega: Iterable[float],
    hbar: sp.Symbol,
) -> dict[str, Dict[str, sp.Expr]]:
    """Return the current ``H30H30`` split with trusted and extended totals.

    At the present stage the diagonal family is split explicitly into:

    - ``pure_diagonal_total`` = ``D^(0)_{iii,iii}``;
    - ``trusted_total`` = ``D^(0)_{iii,iii} + D^(1)_{iii,iii}``.

    The remaining scaffold families are kept explicit for diagnostics, but the
    external VPT-vs-RVCI sanity check shows that they are not yet in a
    physically acceptable normalization range. Therefore:

    - ``trusted_total`` is the solver-facing provisional channel;
    - ``extended_total`` is the diagnostic scaffold sum beyond the trusted core.
    """
    _warn_placeholder("H30H30")
    leading = _h30h30_diag1_iii_iii(mu1, phi3, omega, hbar)
    leading_corr = _h30h30_diag1_iii_iii_correction_candidate(mu1, phi3, omega, hbar)
    semidiag = _h30h30_diag1_iii_iij_0(mu1, phi3, omega, hbar)
    candidate = _h30h30_diag0_iii_iij_1_candidate(mu1, phi3, omega, hbar)
    candidate_res = _h30h30_diag0_iii_iij_2_resonance_candidate(mu1, phi3, omega, hbar)
    candidate_res_reg = _h30h30_diag0_iii_iij_2_regularized_preview(mu1, phi3, omega, hbar)
    iij1 = _h30h30_iij_iij_1_candidate(mu1, phi3, omega, hbar)
    iij2 = _h30h30_iij_iij_2_candidate(mu1, phi3, omega, hbar)
    rotmix_diag1 = _h30h30_diag1_iii_iii_rotmix_candidate(mu1, phi3, omega, hbar)
    residual = _h30h30_placeholder_residual(mu1, phi3, omega)
    pure_diagonal_total = leading
    trusted_total = _sum_tau_dicts(pure_diagonal_total, leading_corr)
    extended_total = _sum_tau_dicts(trusted_total, semidiag, residual)
    resonance_preview_total = _sum_tau_dicts(trusted_total, candidate_res_reg)
    diagonal_pair_preview_total = trusted_total
    return {
        "total": trusted_total,
        "pure_diagonal_total": pure_diagonal_total,
        "trusted_total": trusted_total,
        "extended_total": extended_total,
        "resonance_preview_total": resonance_preview_total,
        "diagonal_pair_preview_total": diagonal_pair_preview_total,
        "D0_iii_iii": leading,
        "diag_0_iii_iii": leading,
        "D1_iii_iii": leading_corr,
        "diag_1_iii_iii": leading_corr,
        "diag_1_iii_iii_correction_candidate": leading_corr,
        "diag_1_iii_iii_rotmix_candidate": rotmix_diag1,
        "D0_iii_iij_0": semidiag,
        "diag_1_iii_iij_0": semidiag,
        "D0_iii_iij_1_candidate": candidate,
        "diag_0_iii_iij_1_candidate": candidate,
        "D0_iii_iij_2_resonance_candidate": candidate_res,
        "diag_0_iii_iij_2_resonance_candidate": candidate_res,
        "D0_iii_iij_2_regularized_preview": candidate_res_reg,
        "diag_0_iii_iij_2_regularized_preview": candidate_res_reg,
        "D0_iij_iij_1_candidate": iij1,
        "diag_0_iij_iij_1_candidate": iij1,
        "D0_iij_iij_2_candidate": iij2,
        "diag_0_iij_iij_2_candidate": iij2,
        "placeholder_residual": residual,
        # Backward-compatible aliases retained while the H30H30 branch is being
        # rebuilt around the paper2 seven-scaffold notation.
        "legacy_diag_1_iii_iii": leading,
    }


def channel_h12h30_decomposed(
    mu1,
    mu2,
    phi3: sp.MutableDenseNDimArray,
    omega: Iterable[float],
    hbar: sp.Symbol,
    seed=None,
    exact_calibration=False,
) -> dict[str, Dict[str, sp.Expr]]:
    """Return ``H12H30`` split into appendix-style modal sectors.

    ``phi3`` is expected in the aligned reduced Gaussian convention (cm^-1).
    Each sector is accumulated in cm^-1 and converted back to the internal
    Wilson-tensor atomic-unit scale before return.

    Naming convention:
    - ``S1`` / ``one_mode`` / ``diagonal``: terms driven by ``Phi_iii``
    - ``S2`` / ``two_mode`` / ``semidiagonal``: terms driven by ``Phi_iij`` and ``Phi_ijj``
    - ``S3`` / ``three_mode``: appendix-style ``Phi_ijk`` block
    - ``S3_addition``: extra ``Phi_ijk`` family kept separately as an additive
      extension to the appendix block. In the current reconstruction this term
      is often numerically dominant and must therefore be reported explicitly.
    """
    scaffold_terms = _h12h30_scaffold_terms(mu1, mu2, phi3, omega)
    semia = scaffold_terms["sd_iij_over_w2_w_wp"]
    semib = scaffold_terms["sd_iij_over_w_wp_2wipj"]
    semic = scaffold_terms["sd_iij_over_w_wp_wi2wj"]
    semidiagonal_pair_mean = _scale_tau(_sum_tau_dicts(semib, semic), sp.Rational(1, 2))
    semidiagonal_pair_split = _scale_tau(_sum_tau_dicts(semic, _scale_tau(semib, -1)), sp.Rational(1, 2))
    # Effective semi-diagonal basis:
    # - ``semidiagonal_effective`` is the piece that actually enters the total
    # - ``semidiagonal_split`` measures the residual mismatch between the two
    #   nearly degenerate denominator families
    semidiagonal_effective = _sum_tau_dicts(semia, _scale_tau(semidiagonal_pair_mean, 2))
    diagonal = _sum_tau_dicts(*(scaffold_terms[name] for name in _H12H30_DIAG_SCAFFOLDS))
    semidiagonal = _sum_tau_dicts(*(scaffold_terms[name] for name in _H12H30_SEMIDIAG_SCAFFOLDS))
    threea = scaffold_terms["tri_ijk_over_pair_sums"]
    threeb = scaffold_terms["tri_ijk_over_w_pair_sums"]
    three_index = _sum_tau_dicts(threea, threeb)
    three_index_split = _scale_tau(_sum_tau_dicts(threeb, _scale_tau(threea, -1)), sp.Rational(1, 2))
    total = _sum_tau_dicts(diagonal, semidiagonal, three_index)
    return {
        "total": total,
        "S1": diagonal,
        "S2": semidiagonal,
        "S3": threea,
        "S3_addition": threeb,
        "S2_effective": semidiagonal_effective,
        "S2_split": semidiagonal_pair_split,
        "S3_effective": threea,
        "S3_split": three_index_split,
        "S3_pair": threea,
        "S3_wpair": threeb,
        "one_mode": diagonal,
        "two_mode": semidiagonal,
        "three_mode": threea,
        "diagonal": diagonal,
        "semidiagonal": semidiagonal,
        "semidiagonal_effective": semidiagonal_effective,
        "semidiagonal_pair_mean": semidiagonal_pair_mean,
        "semidiagonal_split": semidiagonal_pair_split,
        "three_index": three_index,
        "three_index_effective": threea,
        "three_index_addition": threeb,
        "three_index_split": three_index_split,
        "three_index_triads_cm": _h12h30_threeindex_triad_cm_terms(mu1, mu2, phi3, omega),
        **{name: scaffold_terms[name] for name in _H12H30_SCAFFOLD_COEFFS},
    }


def channel_h30h30(mu1, phi3: sp.MutableDenseNDimArray, omega: Iterable[float], hbar: sp.Symbol, seed: int, exact_calibration=False):
    return channel_h30h30_decomposed(mu1, phi3, omega, hbar)["total"]


def tau_to_watson_a(tau: Dict[str, sp.Expr]) -> Dict[str, sp.Expr]:
    c = {
        (4, 0, 0): tau["tau_xxxx"],
        (0, 4, 0): tau["tau_yyyy"],
        (0, 0, 4): tau["tau_zzzz"],
        (2, 2, 0): tau["tau_xxyy"],
        (2, 0, 2): tau["tau_xxzz"],
        (0, 2, 2): tau["tau_yyzz"],
    }
    return {
        "DJ": sp.simplify(sp.Rational(1, 8) * (c[(2, 2, 0)] + c[(2, 0, 2)] + c[(0, 2, 2)])),
        "DJK": sp.simplify(sp.Rational(1, 8) * (-2 * c[(2, 2, 0)] + c[(2, 0, 2)] + c[(0, 2, 2)])),
        "DK": sp.simplify(sp.Rational(1, 8) * (c[(4, 0, 0)] + c[(0, 4, 0)] + c[(0, 0, 4)] - 2 * c[(2, 0, 2)] - 2 * c[(0, 2, 2)])),
        "d1": sp.simplify(sp.Rational(1, 8) * (c[(2, 0, 2)] - c[(0, 2, 2)])),
        "d2": sp.simplify(sp.Rational(1, 16) * (c[(4, 0, 0)] - c[(0, 4, 0)])),
    }


def channel_h22_from_mu1_intrinsic(
    mu1: sp.MutableDenseNDimArray,
    intrinsic: sp.MutableDenseNDimArray,
    inertia0: sp.MutableDenseNDimArray,
    omega: Iterable[float],
    hbar: sp.Expr,
) -> dict[str, Dict[str, sp.Expr]]:
    """Return a simple decomposition of H22 into bilinear and intrinsic pieces."""
    bilinear = bilinear_modepair_from_mu1(mu1, inertia0)
    return channel_h22_decomposed(bilinear, intrinsic, omega, hbar)
