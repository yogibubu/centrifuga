#!/usr/bin/env python3
"""Point-group detection from oriented molecular geometry.

This is the CeDiTT-side reuse of the geometry-based symmetry logic already
available in Merlino3.0.  The present module intentionally keeps only the
minimal API needed by the app:

- infer the spectroscopy-relevant rotor class from moments/ABC,
- detect a Schoenflies point group from oriented coordinates,
- return the associated rotational symmetry number.

It can also assign one-dimensional normal-mode irreps for the abelian groups
used in the current CeDiTT workflow.
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


def _geometry_vertical_reflections(coords: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Return candidate vertical-reflection planes inferred from the xy geometry.

    This supplements the canonical ``sigma_v_{n,k}`` family for cases where the
    oriented structure is rotated by an arbitrary in-plane offset relative to the
    hard-coded angular grid. Candidate planes are generated both through atomic
    directions and through bisectors between projected directions.
    """
    xy = np.asarray(coords, dtype=float)[:, :2]
    radii = np.linalg.norm(xy, axis=1)
    angles = [float(np.mod(np.arctan2(v[1], v[0]), np.pi)) for v, r in zip(xy, radii) if r > 1.0e-6]
    if not angles:
        return []
    plane_angles = set()
    for a in angles:
        plane_angles.add(a)
    n_ang = len(angles)
    for i in range(n_ang):
        for j in range(i + 1, n_ang):
            a = angles[i]
            b = angles[j]
            da = ((b - a + np.pi / 2.0) % np.pi) - np.pi / 2.0
            plane_angles.add(float(np.mod(a + 0.5 * da, np.pi)))
    out: list[tuple[str, np.ndarray]] = []
    for idx, beta in enumerate(sorted(plane_angles)):
        normal_angle = beta + np.pi / 2.0
        out.append(
            (
                f"sigma_v_geom_{idx}",
                _reflection_matrix_from_normal((float(np.cos(normal_angle)), float(np.sin(normal_angle)), 0.0)),
            )
        )
    return out


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
            # Some oriented geometries place the vertical planes halfway between
            # the canonical angles above; include the shifted family as well.
            theta_shift = np.pi * (k + 0.5) / n
            ops.append((f"sigma_vh_{n}_{k}", _reflection_matrix_from_normal((np.cos(theta_shift), np.sin(theta_shift), 0.0))))
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
    candidate_ops = list(_candidate_ops(max_n=max_n))
    candidate_ops.extend(_geometry_vertical_reflections(coords))
    for label, r in candidate_ops:
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
    explicit_pg = getattr(model, "point_group", None)
    sigma_geom, point_group_geom = point_group_from_geometry(symbols, np.asarray(model.coords_pa_ang, dtype=float), rotor_type, tol=tol)
    elements, classes, permutations = symmetry_elements_from_geometry(symbols, np.asarray(model.coords_pa_ang, dtype=float), tol=tol)
    if explicit_pg and rotational_symmetry_number(explicit_pg) > 1:
        point_group = explicit_pg
    else:
        point_group = point_group_geom
    sigma = rotational_symmetry_number(point_group)
    return {
        "point_group": point_group,
        "point_group_geometry": point_group_geom,
        "rotational_symmetry_number": int(sigma),
        "rotor_type_for_symmetry": rotor_type,
        "n_symmetry_operations": len(elements),
        "n_atom_classes": len(classes),
        "equivalent_atom_classes": classes,
        "operation_labels": [label for label, _r in elements],
        "has_permutations": bool(permutations),
    }


def _operation_matrix_from_mapping(rotation: np.ndarray, mapping: list[int], n_atoms: int) -> np.ndarray:
    op = np.zeros((3 * n_atoms, 3 * n_atoms), dtype=float)
    for i, j in enumerate(mapping):
        op[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = rotation
    return op


def _axis_plane_labels(axis: str) -> tuple[str, tuple[str, str], tuple[str, str]]:
    if axis == "x":
        return "sigma_yz", ("sigma_xy", "sigma_xz"), ("y", "z")
    if axis == "y":
        return "sigma_xz", ("sigma_xy", "sigma_yz"), ("x", "z")
    return "sigma_xy", ("sigma_xz", "sigma_yz"), ("x", "y")


def _abelian_operation_keys(point_group: str, labels: list[str]) -> tuple[str, ...] | None:
    axis, _axis_name = _highest_cn_axis(labels)
    axis_use = _axis_name if _axis_name in {"x", "y", "z"} else "z"
    sigma_h_label, sigma_v_labels, _perp = _axis_plane_labels(axis_use)
    c2_label = f"C2{axis_use}^1"
    if point_group == "C1":
        return ("E",)
    if point_group == "Ci":
        return ("E", "i")
    if point_group == "Cs":
        for lab in labels:
            if lab.startswith("sigma"):
                return ("E", lab)
        return None
    if point_group == "C2":
        return ("E", c2_label) if c2_label in labels else None
    if point_group == "C2v":
        need = ("E", c2_label, sigma_v_labels[0], sigma_v_labels[1])
        return need if all(lab in labels for lab in need) else None
    if point_group == "C2h":
        need = ("E", c2_label, "i", sigma_h_label)
        return need if all(lab in labels for lab in need) else None
    if point_group == "D2":
        need = ("E", "C2z^1", "C2y^1", "C2x^1")
        return need if all(lab in labels for lab in need) else None
    if point_group == "D2h":
        need = ("E", "C2z^1", "C2y^1", "C2x^1", "i", "sigma_xy", "sigma_xz", "sigma_yz")
        return need if all(lab in labels for lab in need) else None
    return None


def _abelian_character_table(point_group: str, axis_label: str | None, labels: tuple[str, ...]) -> dict[str, tuple[int, ...]] | None:
    if point_group == "C1":
        return {"A": (1,)}
    if point_group == "Ci":
        return {"Ag": (1, 1), "Au": (1, -1)}
    if point_group == "Cs":
        return {"A'": (1, 1), "A''": (1, -1)}
    if point_group == "C2":
        return {"A": (1, 1), "B": (1, -1)}
    if point_group == "C2v":
        if axis_label not in {"x", "y", "z"}:
            axis_label = "z"
        sigma_h_label, sigma_v_labels, perp = _axis_plane_labels(axis_label)
        # Standard naming with the principal C2 axis along ``axis_label``.
        return {
            "A1": (1, 1, 1, 1),
            "A2": (1, 1, -1, -1),
            f"B1[{perp[0]}]": (1, -1, 1, -1),
            f"B2[{perp[1]}]": (1, -1, -1, 1),
        }
    if point_group == "C2h":
        return {
            "Ag": (1, 1, 1, 1),
            "Bg": (1, -1, 1, -1),
            "Au": (1, 1, -1, -1),
            "Bu": (1, -1, -1, 1),
        }
    if point_group == "D2":
        return {
            "A": (1, 1, 1, 1),
            "B1": (1, 1, -1, -1),
            "B2": (1, -1, 1, -1),
            "B3": (1, -1, -1, 1),
        }
    if point_group == "D2h":
        return {
            "Ag": (1, 1, 1, 1, 1, 1, 1, 1),
            "B1g": (1, 1, -1, -1, 1, 1, -1, -1),
            "B2g": (1, -1, 1, -1, 1, -1, 1, -1),
            "B3g": (1, -1, -1, 1, 1, -1, -1, 1),
            "Au": (1, 1, 1, 1, -1, -1, -1, -1),
            "B1u": (1, 1, -1, -1, -1, -1, 1, 1),
            "B2u": (1, -1, 1, -1, -1, 1, -1, 1),
            "B3u": (1, -1, -1, 1, -1, 1, 1, -1),
        }
    return None


def _nonabelian_operation_keys(point_group: str, labels: list[str]) -> tuple[str, ...] | None:
    if point_group == "C3v":
        sigma_v = next((lab for lab in labels if lab.startswith("sigma_v")), None)
        need = ("E", "C3z^1", sigma_v) if sigma_v is not None else None
        return need if need is not None and all(lab in labels for lab in need) else None
    if point_group == "C4v":
        sigma_v = next((lab for lab in labels if lab in {"sigma_xz", "sigma_yz"}), None)
        need = ("E", "C4z^1", sigma_v) if sigma_v is not None else None
        return need if need is not None and all(lab in labels for lab in need) else None
    if point_group == "C5v":
        sigma_v = next((lab for lab in labels if lab in {"sigma_xz", "sigma_yz"} or lab.startswith("sigma_v")), None)
        need = ("E", "C5z^1", sigma_v) if sigma_v is not None else None
        return need if need is not None and all(lab in labels for lab in need) else None
    if point_group == "C6v":
        sigma_v = next((lab for lab in labels if lab in {"sigma_xz", "sigma_yz"} or lab.startswith("sigma_v")), None)
        need = ("E", "C6z^1", sigma_v) if sigma_v is not None else None
        return need if need is not None and all(lab in labels for lab in need) else None
    if point_group == "D3":
        c2_perp = next((lab for lab in labels if lab.startswith("C2_xy")), None)
        need = ("E", "C3z^1", c2_perp) if c2_perp is not None else None
        return need if need is not None and all(lab in labels for lab in need) else None
    if point_group == "D4":
        c2_perp = next((lab for lab in labels if lab.startswith("C2_xy") or lab in {"C2x^1", "C2y^1"}), None)
        need = ("E", "C4z^1", c2_perp) if c2_perp is not None else None
        return need if need is not None and all(lab in labels for lab in need) else None
    if point_group == "D5":
        c2_perp = next((lab for lab in labels if lab.startswith("C2_xy") or lab in {"C2x^1", "C2y^1"}), None)
        need = ("E", "C5z^1", c2_perp) if c2_perp is not None else None
        return need if need is not None and all(lab in labels for lab in need) else None
    if point_group == "D6":
        c2_perp = next((lab for lab in labels if lab.startswith("C2_xy") or lab in {"C2x^1", "C2y^1"}), None)
        need = ("E", "C6z^1", c2_perp) if c2_perp is not None else None
        return need if need is not None and all(lab in labels for lab in need) else None
    if point_group == "D3h":
        c2_perp = next((lab for lab in labels if lab.startswith("C2_xy")), None)
        sigma_v = next((lab for lab in labels if lab.startswith("sigma_v")), None)
        need = ("E", "C3z^1", c2_perp, "sigma_xy", "S3", sigma_v) if c2_perp is not None and sigma_v is not None else None
        return need if need is not None and all(lab in labels for lab in need) else None
    if point_group == "D4h":
        c2_perp = next((lab for lab in labels if lab.startswith("C2_xy") or lab in {"C2x^1", "C2y^1"}), None)
        sigma_v = next((lab for lab in labels if lab in {"sigma_xz", "sigma_yz"} or lab.startswith("sigma_v")), None)
        need = ("E", "C4z^1", c2_perp, "i", "S4", sigma_v) if c2_perp is not None and sigma_v is not None and "i" in labels and "S4" in labels else None
        return need if need is not None and all(lab in labels for lab in need) else None
    if point_group == "D5h":
        c2_perp = next((lab for lab in labels if lab.startswith("C2_xy") or lab in {"C2x^1", "C2y^1"}), None)
        sigma_v = next((lab for lab in labels if lab in {"sigma_xz", "sigma_yz"} or lab.startswith("sigma_v")), None)
        need = ("E", "C5z^1", c2_perp, "sigma_xy", sigma_v) if c2_perp is not None and sigma_v is not None and "sigma_xy" in labels else None
        return need if need is not None and all(lab in labels for lab in need) else None
    if point_group == "D6h":
        c2_perp = next((lab for lab in labels if lab.startswith("C2_xy") or lab in {"C2x^1", "C2y^1"}), None)
        sigma_v = next((lab for lab in labels if lab in {"sigma_xz", "sigma_yz"} or lab.startswith("sigma_v")), None)
        need = ("E", "C6z^1", c2_perp, "i", "S6", sigma_v) if c2_perp is not None and sigma_v is not None and "i" in labels and "S6" in labels else None
        return need if need is not None and all(lab in labels for lab in need) else None
    return None


def _nonabelian_character_table(point_group: str, labels: tuple[str, ...]) -> dict[str, tuple[int, ...]] | None:
    if point_group == "C3v":
        return {
            "A1": (1, 1, 1),
            "A2": (1, 1, -1),
            "E": (2, -1, 0),
        }
    if point_group == "C4v":
        return {
            "A1": (1, 1, 1),
            "A2": (1, 1, -1),
            "B1": (1, -1, 1),
            "B2": (1, -1, -1),
            "E": (2, 0, 0),
        }
    if point_group == "C5v":
        return {
            "A1": (1, 1, 1),
            "A2": (1, 1, -1),
            "E1": (2, 2 * np.cos(2.0 * np.pi / 5.0), 0),
            "E2": (2, 2 * np.cos(4.0 * np.pi / 5.0), 0),
        }
    if point_group == "C6v":
        return {
            "A1": (1, 1, 1),
            "A2": (1, 1, -1),
            "B1": (1, -1, 1),
            "B2": (1, -1, -1),
            "E1": (2, 1, 0),
            "E2": (2, -1, 0),
        }
    if point_group == "D3":
        return {
            "A1": (1, 1, 1),
            "A2": (1, 1, -1),
            "E": (2, -1, 0),
        }
    if point_group == "D4":
        return {
            "A1": (1, 1, 1),
            "A2": (1, 1, -1),
            "B1": (1, -1, 1),
            "B2": (1, -1, -1),
            "E": (2, 0, 0),
        }
    if point_group == "D5":
        return {
            "A1": (1, 1, 1),
            "A2": (1, 1, -1),
            "E1": (2, 2 * np.cos(2.0 * np.pi / 5.0), 0),
            "E2": (2, 2 * np.cos(4.0 * np.pi / 5.0), 0),
        }
    if point_group == "D6":
        return {
            "A1": (1, 1, 1),
            "A2": (1, 1, -1),
            "B1": (1, -1, 1),
            "B2": (1, -1, -1),
            "E1": (2, 1, 0),
            "E2": (2, -1, 0),
        }
    if point_group == "D3h":
        return {
            "A1'": (1, 1, 1, 1, 1, 1),
            "A2'": (1, 1, -1, 1, 1, -1),
            "E'": (2, -1, 0, 2, -1, 0),
            'A1"': (1, 1, 1, -1, -1, -1),
            'A2"': (1, 1, -1, -1, -1, 1),
            'E"': (2, -1, 0, -2, 1, 0),
        }
    if point_group == "D4h":
        return {
            "A1g": (1, 1, 1, 1, 1, 1),
            "A2g": (1, 1, -1, 1, 1, -1),
            "B1g": (1, -1, 1, 1, -1, 1),
            "B2g": (1, -1, -1, 1, -1, -1),
            "Eg": (2, 0, 0, 2, 0, 0),
            "A1u": (1, 1, 1, -1, -1, -1),
            "A2u": (1, 1, -1, -1, -1, 1),
            "B1u": (1, -1, 1, -1, 1, -1),
            "B2u": (1, -1, -1, -1, 1, 1),
            "Eu": (2, 0, 0, -2, 0, 0),
        }
    if point_group == "D5h":
        c1 = 2 * np.cos(2.0 * np.pi / 5.0)
        c2 = 2 * np.cos(4.0 * np.pi / 5.0)
        return {
            "A1'": (1, 1, 1, 1, 1),
            "A2'": (1, 1, -1, 1, -1),
            "E1'": (2, c1, 0, 2, 0),
            "E2'": (2, c2, 0, 2, 0),
            'A1"': (1, 1, 1, -1, -1),
            'A2"': (1, 1, -1, -1, 1),
            'E1"': (2, c1, 0, -2, 0),
            'E2"': (2, c2, 0, -2, 0),
        }
    if point_group == "D6h":
        return {
            "A1g": (1, 1, 1, 1, 1, 1),
            "A2g": (1, 1, -1, 1, 1, -1),
            "B1g": (1, -1, 1, 1, -1, 1),
            "B2g": (1, -1, -1, 1, -1, -1),
            "E1g": (2, 1, 0, 2, 1, 0),
            "E2g": (2, -1, 0, 2, -1, 0),
            "A1u": (1, 1, 1, -1, -1, -1),
            "A2u": (1, 1, -1, -1, -1, 1),
            "B1u": (1, -1, 1, -1, 1, -1),
            "B2u": (1, -1, -1, -1, 1, 1),
            "E1u": (2, 1, 0, -2, -1, 0),
            "E2u": (2, -1, 0, -2, 1, 0),
        }
    return None


def _frequency_blocks(freq_cm: np.ndarray, *, tol: float = 2.0e-3) -> list[list[int]]:
    vals = np.asarray(freq_cm, dtype=float).reshape(-1)
    if vals.size == 0:
        return []
    blocks: list[list[int]] = [[0]]
    for idx in range(1, vals.size):
        if abs(vals[idx] - vals[blocks[-1][-1]]) <= tol:
            blocks[-1].append(idx)
        else:
            blocks.append([idx])
    return blocks


def _nonabelian_mode_irreps(
    point_group: str,
    chart: dict[str, tuple[int, ...]],
    keys: tuple[str, ...],
    op_map: dict[str, tuple[np.ndarray, list[int]]],
    vib_arr: np.ndarray,
    freq_cm: np.ndarray,
    n_atoms: int,
    *,
    one_dim_char_tol: float = 2.0e-1,
    two_dim_sum_tol: float = 2.0e-1,
    pair_freq_tol: float = 2.5e-1,
) -> list[str]:
    n_modes = vib_arr.shape[1]
    per_mode_chars: list[tuple[float, ...]] = []
    for mode_idx in range(n_modes):
        vec = vib_arr[:, mode_idx]
        chars: list[float] = []
        for key in keys:
            rot, perm = op_map[key]
            op = _operation_matrix_from_mapping(np.asarray(rot, dtype=float), list(perm), n_atoms)
            chars.append(float(vec @ (op @ vec)))
        per_mode_chars.append(tuple(chars))

    out = ["?"] * n_modes
    one_dim_rows = {name: row for name, row in chart.items() if int(round(float(row[0]))) == 1}
    two_dim_rows = {name: row for name, row in chart.items() if int(round(float(row[0]))) == 2}

    for mode_idx, sig in enumerate(per_mode_chars):
        match = next(
            (
                name
                for name, row in one_dim_rows.items()
                if len(row) == len(sig) and all(abs(float(a) - float(b)) <= one_dim_char_tol for a, b in zip(sig, row))
            ),
            None,
        )
        if match is not None:
            out[mode_idx] = match

    for irrep_name, row in two_dim_rows.items():
        target = tuple(float(x) for x in row)
        pending = [idx for idx, label in enumerate(out) if label == "?"]
        consumed: set[int] = set()
        for i in pending:
            if i in consumed:
                continue
            if out[i] != "?":
                continue
            best_j: int | None = None
            best_gap = None
            for j in pending:
                if j == i or j in consumed:
                    continue
                if out[j] != "?":
                    continue
                if abs(float(freq_cm[j]) - float(freq_cm[i])) > pair_freq_tol:
                    continue
                summed = tuple(float(a) + float(b) for a, b in zip(per_mode_chars[i], per_mode_chars[j]))
                if len(summed) != len(target):
                    continue
                if not all(abs(a - b) <= two_dim_sum_tol for a, b in zip(summed, target)):
                    continue
                gap = abs(float(freq_cm[j]) - float(freq_cm[i]))
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_j = j
            if best_j is not None:
                out[i] = irrep_name
                out[best_j] = irrep_name
                consumed.add(i)
                consumed.add(best_j)

    return out


def _linear_sigma_v_label(axis_label: str, labels: list[str]) -> str | None:
    if axis_label == "x":
        candidates = ("sigma_xy", "sigma_xz")
    elif axis_label == "y":
        candidates = ("sigma_xy", "sigma_yz")
    else:
        candidates = ("sigma_xz", "sigma_yz")
    return next((lab for lab in candidates if lab in labels), None)


def _linear_mode_irreps(
    point_group: str,
    axis_label: str,
    labels: list[str],
    op_map: dict[str, tuple[np.ndarray, list[int]]],
    vib_arr: np.ndarray,
    freq_cm: np.ndarray,
    n_atoms: int,
) -> list[str]:
    out: list[str] = []
    inv_key = "i" if point_group == "Dinfh" and "i" in op_map else None
    sigma_v_key = _linear_sigma_v_label(axis_label, labels)

    for block in _frequency_blocks(freq_cm):
        V = vib_arr[:, block]
        parity = ""
        if inv_key is not None:
            rot, perm = op_map[inv_key]
            op = _operation_matrix_from_mapping(np.asarray(rot, dtype=float), list(perm), n_atoms)
            tr = float(np.trace(V.T @ (op @ V)))
            parity = "_g" if tr > 0.0 else "_u"

        if len(block) == 1:
            sign = ""
            if sigma_v_key is not None:
                rot, perm = op_map[sigma_v_key]
                op = _operation_matrix_from_mapping(np.asarray(rot, dtype=float), list(perm), n_atoms)
                overlap = float(V[:, 0].T @ (op @ V[:, 0]))
                sign = "+" if overlap >= 0.0 else "-"
            out.append(f"Sigma{parity}{sign}")
            continue

        if len(block) == 2:
            out.extend([f"Pi{parity}", f"Pi{parity}"])
            continue

        out.extend(["?"] * len(block))
    return out


def assign_normal_mode_irreps(model: Any, *, tol: float = 1.0e-5) -> list[str] | None:
    symbols = getattr(model, "symbols", None)
    coords = getattr(model, "coords_pa_ang", None)
    vib = getattr(model, "vib_vecs_mw_pa", None)
    abc = getattr(model, "abc_mhz", None)
    moments = getattr(model, "moments_amu_a2", None)
    if symbols is None or coords is None or vib is None or abc is None or moments is None:
        return None

    rotor_type = rotor_type_for_symmetry(np.asarray(abc, dtype=float), np.asarray(moments, dtype=float))
    explicit_pg = getattr(model, "point_group", None)
    _sigma, point_group_geom = point_group_from_geometry(symbols, np.asarray(coords, dtype=float), rotor_type, tol=1.0e-3)
    if explicit_pg and rotational_symmetry_number(explicit_pg) > 1:
        point_group = explicit_pg
    else:
        point_group = point_group_geom
    elements, _classes, permutations = symmetry_elements_from_geometry(symbols, np.asarray(coords, dtype=float), tol=1.0e-3)
    op_map = {label: (rot, perm) for (label, rot), perm in zip(elements, permutations)}
    labels = [label for label, _rot in elements]
    n_atoms = len(symbols)
    vib_arr = np.asarray(vib, dtype=float)
    freq_cm = np.asarray(getattr(model, "vib_freq_cm", []), dtype=float)
    if freq_cm.size != vib_arr.shape[1]:
        return None
    _axis_n, axis_label = _highest_cn_axis(labels)
    axis_label = axis_label if axis_label in {"x", "y", "z"} else "z"

    if point_group in {"Dinfh", "Cinfv"}:
        return _linear_mode_irreps(point_group, axis_label, labels, op_map, vib_arr, freq_cm, n_atoms)

    keys = _abelian_operation_keys(point_group, labels)
    if keys is not None:
        chart = _abelian_character_table(point_group, axis_label, keys)
        if chart is None:
            return None
        out: list[str] = []
        for mode_idx in range(vib_arr.shape[1]):
            vec = vib_arr[:, mode_idx]
            chars: list[int] = []
            supported = True
            for key in keys:
                rot, perm = op_map[key]
                op = _operation_matrix_from_mapping(np.asarray(rot, dtype=float), list(perm), n_atoms)
                overlap = float(vec @ (op @ vec))
                if abs(abs(overlap) - 1.0) > tol:
                    supported = False
                    break
                chars.append(1 if overlap >= 0.0 else -1)
            if not supported:
                out.append("?")
                continue
            sig = tuple(chars)
            match = next((name for name, row in chart.items() if tuple(row) == sig), "?")
            out.append(match)
        return out

    keys = _nonabelian_operation_keys(point_group, labels)
    if keys is None:
        return None
    chart = _nonabelian_character_table(point_group, keys)
    if chart is None:
        return None

    return _nonabelian_mode_irreps(point_group, chart, keys, op_map, vib_arr, freq_cm, n_atoms)
