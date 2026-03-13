#!/usr/bin/env python3
"""Point-group detection from oriented molecular geometry.

This is the CeDiTT-side reuse of the geometry-based symmetry logic already
available in Merlino3.0.  The present module intentionally keeps only the
minimal API needed by the app:

- infer the spectroscopy-relevant rotor class from moments/ABC,
- detect a Schoenflies point group from oriented coordinates,
- return the associated rotational symmetry number.

It does not attempt irrep assignment of normal modes; that is a separate task.
"""

from __future__ import annotations

from functools import lru_cache
from math import gcd
from typing import Any

import re

import numpy as np


_Z_TO_SYMBOL = [
    "",
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc",
    "Lv", "Ts", "Og",
]


def symbols_from_atomic_numbers(atomic_numbers: np.ndarray) -> list[str]:
    out: list[str] = []
    for z in np.asarray(atomic_numbers, dtype=int).reshape(-1):
        if z <= 0 or z >= len(_Z_TO_SYMBOL):
            raise ValueError(f"Unsupported atomic number {z}")
        out.append(_Z_TO_SYMBOL[z])
    return out


def _close(x: float, y: float, *, rel_tol: float, abs_tol: float, scale: float) -> bool:
    return abs(x - y) <= max(abs_tol, rel_tol * max(abs(x), abs(y), scale, 1.0))


def rotor_type_for_symmetry(
    abc_mhz: np.ndarray,
    moments_amu_a2: np.ndarray,
    *,
    rel_tol: float = 1.0e-6,
    abs_mhz_tol: float = 1.0e-3,
    moment_tol: float = 1.0e-8,
) -> str:
    abc = np.asarray(abc_mhz, dtype=float).reshape(3)
    moments = np.asarray(moments_amu_a2, dtype=float).reshape(3)
    finite = abc[np.isfinite(abc)]
    scale = float(np.max(np.abs(finite))) if finite.size else 1.0

    if abs(moments[0]) <= max(moment_tol, rel_tol * max(abs(moments[2]), 1.0)):
        return "linear"

    if _close(moments[0], moments[2], rel_tol=rel_tol, abs_tol=moment_tol, scale=max(abs(moments[2]), 1.0)):
        return "spherical"

    if _close(abc[1], abc[2], rel_tol=rel_tol, abs_tol=abs_mhz_tol, scale=scale):
        return "symmetric_prolate"
    if _close(abc[0], abc[1], rel_tol=rel_tol, abs_tol=abs_mhz_tol, scale=scale):
        return "symmetric_oblate"
    return "asymmetric"


def _match_with_map(symbols: list[str], coords1: np.ndarray, coords2: np.ndarray, tol: float) -> list[int] | None:
    used = np.zeros(len(coords2), dtype=bool)
    mapping = [-1] * len(coords1)
    for i, v in enumerate(coords1):
        found = False
        for j, w in enumerate(coords2):
            if used[j]:
                continue
            if symbols[i] != symbols[j]:
                continue
            if np.linalg.norm(v - w) < tol:
                used[j] = True
                mapping[i] = j
                found = True
                break
        if not found:
            return None
    return mapping


def _rotation_matrix(axis: tuple[float, float, float], theta: float) -> np.ndarray:
    axis_vec = np.array(axis, dtype=float)
    axis_vec /= np.linalg.norm(axis_vec)
    x, y, z = axis_vec
    c = np.cos(theta)
    s = np.sin(theta)
    cc = 1.0 - c
    return np.array(
        [
            [c + x * x * cc, x * y * cc - z * s, x * z * cc + y * s],
            [y * x * cc + z * s, c + y * y * cc, y * z * cc - x * s],
            [z * x * cc - y * s, z * y * cc + x * s, c + z * z * cc],
        ]
    )


def _reflection_matrix_from_normal(normal: tuple[float, float, float]) -> np.ndarray:
    nvec = np.array(normal, dtype=float)
    nvec /= np.linalg.norm(nvec)
    return np.eye(3) - 2.0 * np.outer(nvec, nvec)


@lru_cache(maxsize=16)
def _candidate_ops(max_n: int = 6) -> list[tuple[str, np.ndarray]]:
    ops: list[tuple[str, np.ndarray]] = []
    ops.append(("E", np.eye(3)))
    ops.append(("i", -np.eye(3)))
    for axis, name in ((0, "sigma_yz"), (1, "sigma_xz"), (2, "sigma_xy")):
        r = np.eye(3)
        r[axis, axis] = -1.0
        ops.append((name, r))
    for n in range(2, max_n + 1):
        for k in range(1, n):
            if gcd(n, k) != 1:
                continue
            theta = 2.0 * np.pi * k / n
            ops.append((f"C{n}z^{k}", _rotation_matrix((0.0, 0.0, 1.0), theta)))
            ops.append((f"C{n}x^{k}", _rotation_matrix((1.0, 0.0, 0.0), theta)))
            ops.append((f"C{n}y^{k}", _rotation_matrix((0.0, 1.0, 0.0), theta)))
    for n in range(3, max_n + 1):
        for k in range(n):
            theta = np.pi * k / n
            ops.append((f"sigma_v_{n}_{k}", _reflection_matrix_from_normal((np.cos(theta), np.sin(theta), 0.0))))
    for n in range(2, max_n + 1):
        for k in range(n):
            theta = np.pi * k / n
            ops.append((f"C2_xy_{n}_{k}", _rotation_matrix((np.cos(theta), np.sin(theta), 0.0), np.pi)))
    for n in range(3, max_n + 1):
        theta = 2.0 * np.pi / n
        sigma_xy = np.eye(3)
        sigma_xy[2, 2] = -1.0
        ops.append((f"S{n}", sigma_xy @ _rotation_matrix((0.0, 0.0, 1.0), theta)))
    return ops


def _highest_cn_axis(labels: list[str]) -> tuple[int, str | None]:
    nmax = 1
    axis = None
    for lab in labels:
        if not lab.startswith("C"):
            continue
        m = re.match(r"C(\d+)", lab)
        if not m:
            continue
        n = int(m.group(1))
        if n > nmax:
            nmax = n
            axis = "z" if "z" in lab else "x" if "x" in lab else "y" if "y" in lab else None
    return nmax, axis


def _count_cn_axis(labels: list[str], axis: str = "z") -> tuple[int, set[int]]:
    ns: set[int] = set()
    for lab in labels:
        if not lab.startswith("C") or axis not in lab:
            continue
        m = re.match(r"C(\d+)", lab)
        if m:
            ns.add(int(m.group(1)))
    return len(ns), ns


def _highest_sn(labels: list[str]) -> int:
    nmax = 1
    for lab in labels:
        if not lab.startswith("S"):
            continue
        m = re.match(r"S(\d+)", lab)
        if m:
            nmax = max(nmax, int(m.group(1)))
    return nmax


def _spherical_top_guess(symbols: list[str], coords: np.ndarray) -> str | None:
    if len(symbols) < 5:
        return None
    if len(set(symbols)) == 1:
        return None
    center = 0
    ligands = [i for i in range(len(symbols)) if i != center]
    if len({symbols[i] for i in ligands}) > 1:
        return None
    radii = np.linalg.norm(coords[ligands] - coords[center], axis=1)
    if np.std(radii) < 1.0e-3:
        if len(ligands) == 4:
            return "Td"
        if len(ligands) in {6, 8}:
            return "Oh"
        if len(ligands) == 12:
            return "Ih"
    return None


def _group_label(elements: list[tuple[str, np.ndarray]], *, linear: bool = False) -> str:
    labels = [e[0] for e in elements]
    nmax, axis = _highest_cn_axis(labels)
    ncount_z, _ = _count_cn_axis(labels, axis="z")
    snmax = _highest_sn(labels)
    has_i = any(lab == "i" for lab in labels)
    has_sigma = any(lab.startswith("sigma") for lab in labels)
    has_c2 = any(lab.startswith("C2") for lab in labels)
    has_s = any(lab.startswith("S") for lab in labels)
    has_poly = any(lab.endswith(("_t", "_o", "_i")) for lab in labels)
    axis_use = axis if axis in {"x", "y", "z"} else "z"
    sigma_h_label = {"x": "sigma_yz", "y": "sigma_xz", "z": "sigma_xy"}[axis_use]
    sigma_v_labels = {"x": {"sigma_xy", "sigma_xz"}, "y": {"sigma_xy", "sigma_yz"}, "z": {"sigma_xz", "sigma_yz"}}[axis_use]
    has_sigma_h = sigma_h_label in labels
    has_sigma_v = any(lab.startswith("sigma_v") for lab in labels) or any(lab in sigma_v_labels for lab in labels)
    c2_axes: set[str] = set()
    for lab in labels:
        m = re.match(r"C2([xyz])\^", lab)
        if m:
            c2_axes.add(m.group(1))
        if lab.startswith("C2_xy"):
            c2_axes.add("xy")
    has_c2_perp = any(ax != axis for ax in c2_axes)
    t_ops = [lab for lab in labels if lab.endswith("_t")]
    o_ops = [lab for lab in labels if lab.endswith("_o")]
    i_ops = [lab for lab in labels if lab.endswith("_i")]

    if linear and not has_poly:
        return "Dinfh" if has_i else "Cinfv"
    if not has_poly:
        if (nmax >= 3 and ncount_z >= 2 and has_sigma_v) or (nmax >= 6 and has_sigma_v):
            return "Dinfh" if (has_i or has_c2_perp) else "Cinfv"

    if len(i_ops) >= 20:
        return "Ih" if has_i else "I"
    if len(o_ops) >= 12:
        return "Oh" if has_i else "O"
    if len(t_ops) >= 6:
        return "Th" if has_i else "Td" if has_sigma else "T"
    if has_s and (has_c2 or has_c2_perp) and not has_sigma:
        n_eff = nmax if nmax > 1 else snmax
        return f"D{n_eff}d"
    if has_s and not has_sigma and not has_i:
        return f"S{snmax}"
    if nmax >= 2:
        if has_sigma_h and has_c2_perp:
            return f"D{nmax}h"
        if has_sigma_h:
            return f"C{nmax}h"
        if has_sigma:
            return f"C{nmax}v"
        if has_c2 or has_c2_perp:
            return f"D{nmax}"
        return f"C{nmax}"
    if has_i:
        return "Ci"
    if has_sigma:
        return "Cs"
    return "C1"


def rotational_symmetry_number(point_group: str) -> int:
    if point_group == "Cinfv":
        return 1
    if point_group == "Dinfh":
        return 2
    if point_group in {"C1", "Cs", "Ci"}:
        return 1
    if point_group == "Td":
        return 12
    if point_group == "Oh":
        return 24
    if point_group == "Ih":
        return 60
    m = re.match(r"D(\d+)", point_group)
    if m:
        return 2 * int(m.group(1))
    m = re.match(r"C(\d+)", point_group)
    if m:
        return int(m.group(1))
    m = re.match(r"S(\d+)", point_group)
    if m:
        n = int(m.group(1))
        if n % 2 != 0:
            raise ValueError(f"Invalid improper rotation group '{point_group}'")
        return n // 2
    if point_group in {"T", "Th"}:
        return 12
    if point_group in {"O"}:
        return 24
    if point_group in {"I"}:
        return 60
    raise ValueError(f"Unsupported or unknown point group '{point_group}'")


def symmetry_elements_from_geometry(
    symbols: list[str],
    coords_oriented: np.ndarray,
    *,
    tol: float = 1.0e-3,
    max_n: int = 6,
) -> tuple[list[tuple[str, np.ndarray]], list[list[int]], list[list[int]]]:
    coords = np.asarray(coords_oriented, dtype=float)
    elements: list[tuple[str, np.ndarray]] = []
    permutations: list[list[int]] = []
    for label, r in _candidate_ops(max_n=max_n):
        mapping = _match_with_map(symbols, coords, coords @ r.T, tol)
        if mapping is not None:
            elements.append((label, r))
            permutations.append(mapping)

    n = len(symbols)
    parent = list(range(n))

    def _find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def _union(a: int, b: int) -> None:
        ra = _find(a)
        rb = _find(b)
        if ra != rb:
            parent[rb] = ra

    for mapping in permutations:
        for i, j in enumerate(mapping):
            _union(i, j)

    classes: dict[int, list[int]] = {}
    for i in range(n):
        classes.setdefault(_find(i), []).append(i)
    return elements, list(classes.values()), permutations


def point_group_from_geometry(
    symbols: list[str],
    coords_oriented: np.ndarray,
    rotor_type: str,
    *,
    tol: float = 1.0e-3,
) -> tuple[int, str]:
    coords = np.asarray(coords_oriented, dtype=float)
    rt = rotor_type.lower()
    if rt == "spherical":
        guess = _spherical_top_guess(symbols, coords)
        if guess is not None:
            return rotational_symmetry_number(guess), guess
    elements, _classes, _perms = symmetry_elements_from_geometry(symbols, coords, tol=tol)
    point_group = _group_label(elements, linear=(rt == "linear"))
    return rotational_symmetry_number(point_group), point_group


def point_group_metadata_from_model(model: Any, *, tol: float = 1.0e-3) -> dict[str, Any] | None:
    symbols = getattr(model, "symbols", None)
    if not symbols:
        return None
    rotor_type = rotor_type_for_symmetry(
        np.asarray(model.abc_mhz, dtype=float),
        np.asarray(model.moments_amu_a2, dtype=float),
    )
    sigma, point_group = point_group_from_geometry(symbols, np.asarray(model.coords_pa_ang, dtype=float), rotor_type, tol=tol)
    elements, classes, permutations = symmetry_elements_from_geometry(symbols, np.asarray(model.coords_pa_ang, dtype=float), tol=tol)
    return {
        "point_group": point_group,
        "rotational_symmetry_number": int(sigma),
        "rotor_type_for_symmetry": rotor_type,
        "n_symmetry_operations": len(elements),
        "n_atom_classes": len(classes),
        "equivalent_atom_classes": classes,
        "operation_labels": [label for label, _r in elements],
        "has_permutations": bool(permutations),
    }
