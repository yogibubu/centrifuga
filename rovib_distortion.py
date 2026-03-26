#!/usr/bin/env python3
"""Rovibrational quartic centrifugal distortion from geometry + Cartesian Hessian.

Workflow (G-tensor / Coriolis formalism):
1) Read XYZ geometry and Cartesian Hessian.
2) Build mass-weighted Hessian Fmw = M^{-1/2} F M^{-1/2}.
3) Diagonalize Fmw to obtain harmonic normal modes.
4) Compute principal moments and rotational constants A,B,C.
5) Compute Coriolis constants zeta_{ab,k}, zeta_{bc,k}, zeta_{ca,k} from
   angular-momentum couplings in mass-weighted normal coordinates.
6) Build second-order rovibrational quartic effective energy from Coriolis
   constants and harmonic frequencies, then fit Watson quartic constants
   (AJ, AJK, AK, dJ, dK) in A reduction.
7) Transform quartic constants between I/II/III and r/l with Yamada matrices.

The main CLI workflow below still uses the Coriolis route. This file also
contains helper routines for the analytic first and second derivatives of the
inverse inertia tensor with respect to normal coordinates, for use in the
channel-resolved VPT4 implementation.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- Constants ---------------------------------------------------------------------

ANGSTROM_TO_BOHR = 1.8897261254578281
AMU_TO_AU_MASS = 1822.888486209
CMINV_TO_MHZ = 29979.2458
ROT_CONST_MHZ_AMU_A2 = 505379.009405
AU_FREQ_TO_CMINV = 219474.6313705

REPRESENTATIONS = ("I", "II", "III")
HANDEDNESS = ("r", "l")

# Standard atomic weights (u), full periodic table (Z=1..118).
ATOMIC_MASSES = {
    "H": 1.008,
    "HE": 4.002602,
    "LI": 6.94,
    "BE": 9.0121831,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998403163,
    "NE": 20.1797,
    "NA": 22.98976928,
    "MG": 24.305,
    "AL": 26.9815385,
    "SI": 28.085,
    "P": 30.973761998,
    "S": 32.06,
    "CL": 35.45,
    "AR": 39.948,
    "K": 39.0983,
    "CA": 40.078,
    "SC": 44.955908,
    "TI": 47.867,
    "V": 50.9415,
    "CR": 51.9961,
    "MN": 54.938044,
    "FE": 55.845,
    "CO": 58.933194,
    "NI": 58.6934,
    "CU": 63.546,
    "ZN": 65.38,
    "GA": 69.723,
    "GE": 72.63,
    "AS": 74.921595,
    "SE": 78.971,
    "BR": 79.904,
    "KR": 83.798,
    "RB": 85.4678,
    "SR": 87.62,
    "Y": 88.90584,
    "ZR": 91.224,
    "NB": 92.90637,
    "MO": 95.95,
    "TC": 98.0,
    "RU": 101.07,
    "RH": 102.9055,
    "PD": 106.42,
    "AG": 107.8682,
    "CD": 112.414,
    "IN": 114.818,
    "SN": 118.71,
    "SB": 121.76,
    "TE": 127.6,
    "I": 126.90447,
    "XE": 131.293,
    "CS": 132.905452,
    "BA": 137.327,
    "LA": 138.90547,
    "CE": 140.116,
    "PR": 140.90766,
    "ND": 144.242,
    "PM": 145.0,
    "SM": 150.36,
    "EU": 151.964,
    "GD": 157.25,
    "TB": 158.92535,
    "DY": 162.5,
    "HO": 164.93033,
    "ER": 167.259,
    "TM": 168.93422,
    "YB": 173.045,
    "LU": 174.9668,
    "HF": 178.49,
    "TA": 180.94788,
    "W": 183.84,
    "RE": 186.207,
    "OS": 190.23,
    "IR": 192.217,
    "PT": 195.084,
    "AU": 196.96657,
    "HG": 200.592,
    "TL": 204.38,
    "PB": 207.2,
    "BI": 208.9804,
    "PO": 209.0,
    "AT": 210.0,
    "RN": 222.0,
    "FR": 223.0,
    "RA": 226.0,
    "AC": 227.0,
    "TH": 232.0377,
    "PA": 231.03588,
    "U": 238.02891,
    "NP": 237.0,
    "PU": 244.0,
    "AM": 243.0,
    "CM": 247.0,
    "BK": 247.0,
    "CF": 251.0,
    "ES": 252.0,
    "FM": 257.0,
    "MD": 258.0,
    "NO": 259.0,
    "LR": 266.0,
    "RF": 267.0,
    "DB": 268.0,
    "SG": 269.0,
    "BH": 270.0,
    "HS": 269.0,
    "MT": 278.0,
    "DS": 281.0,
    "RG": 282.0,
    "CN": 285.0,
    "NH": 286.0,
    "FL": 289.0,
    "MC": 290.0,
    "LV": 293.0,
    "TS": 294.0,
    "OG": 294.0,
}
ATOMIC_NUMBER_TO_SYMBOL = {i: sym for i, sym in enumerate(ATOMIC_MASSES.keys(), start=1)}

# Selected exact isotopic masses (u). If an isotope is not listed, the mass
# number is used as fallback (e.g., Xe-136 -> 136.0 u).
ISOTOPE_MASSES = {
    ("H", 1): 1.00782503223,
    ("H", 2): 2.01410177812,
    ("H", 3): 3.0160492779,
    ("HE", 3): 3.01602932265,
    ("HE", 4): 4.00260325413,
    ("LI", 6): 6.0151228874,
    ("LI", 7): 7.0160034366,
    ("B", 10): 10.01293695,
    ("B", 11): 11.00930536,
    ("C", 12): 12.0,
    ("C", 13): 13.00335483507,
    ("C", 14): 14.0032419884,
    ("N", 14): 14.00307400443,
    ("N", 15): 15.00010889888,
    ("O", 16): 15.99491461957,
    ("O", 17): 16.9991317565,
    ("O", 18): 17.99915961286,
    ("F", 19): 18.99840316273,
    ("NE", 20): 19.9924401762,
    ("NE", 21): 20.993846685,
    ("NE", 22): 21.991385114,
    ("CL", 35): 34.968852682,
    ("CL", 37): 36.965902602,
    ("BR", 79): 78.9183376,
    ("BR", 81): 80.9162897,
    ("I", 127): 126.9044719,
    ("SI", 28): 27.97692653465,
    ("SI", 29): 28.9764946649,
    ("SI", 30): 29.973770136,
    ("S", 32): 31.9720711744,
    ("S", 33): 32.9714589098,
    ("S", 34): 33.967867004,
    ("S", 36): 35.96708071,
}

SYMBOLS_CANONICAL = {
    "H": "H", "HE": "He", "LI": "Li", "BE": "Be", "B": "B", "C": "C", "N": "N", "O": "O", "F": "F", "NE": "Ne",
    "NA": "Na", "MG": "Mg", "AL": "Al", "SI": "Si", "P": "P", "S": "S", "CL": "Cl", "AR": "Ar", "K": "K", "CA": "Ca",
    "SC": "Sc", "TI": "Ti", "V": "V", "CR": "Cr", "MN": "Mn", "FE": "Fe", "CO": "Co", "NI": "Ni", "CU": "Cu", "ZN": "Zn",
    "GA": "Ga", "GE": "Ge", "AS": "As", "SE": "Se", "BR": "Br", "KR": "Kr", "RB": "Rb", "SR": "Sr", "Y": "Y", "ZR": "Zr",
    "NB": "Nb", "MO": "Mo", "TC": "Tc", "RU": "Ru", "RH": "Rh", "PD": "Pd", "AG": "Ag", "CD": "Cd", "IN": "In", "SN": "Sn",
    "SB": "Sb", "TE": "Te", "I": "I", "XE": "Xe", "CS": "Cs", "BA": "Ba", "LA": "La", "CE": "Ce", "PR": "Pr", "ND": "Nd",
    "PM": "Pm", "SM": "Sm", "EU": "Eu", "GD": "Gd", "TB": "Tb", "DY": "Dy", "HO": "Ho", "ER": "Er", "TM": "Tm", "YB": "Yb",
    "LU": "Lu", "HF": "Hf", "TA": "Ta", "W": "W", "RE": "Re", "OS": "Os", "IR": "Ir", "PT": "Pt", "AU": "Au", "HG": "Hg",
    "TL": "Tl", "PB": "Pb", "BI": "Bi", "PO": "Po", "AT": "At", "RN": "Rn", "FR": "Fr", "RA": "Ra", "AC": "Ac", "TH": "Th",
    "PA": "Pa", "U": "U", "NP": "Np", "PU": "Pu", "AM": "Am", "CM": "Cm", "BK": "Bk", "CF": "Cf", "ES": "Es", "FM": "Fm",
    "MD": "Md", "NO": "No", "LR": "Lr", "RF": "Rf", "DB": "Db", "SG": "Sg", "BH": "Bh", "HS": "Hs", "MT": "Mt", "DS": "Ds",
    "RG": "Rg", "CN": "Cn", "NH": "Nh", "FL": "Fl", "MC": "Mc", "LV": "Lv", "TS": "Ts", "OG": "Og",
}


@dataclass
class Molecule:
    symbols: list[str]
    masses_amu: np.ndarray
    coords_ang: np.ndarray

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)


@dataclass
class HarmonicInertiaModel:
    """Canonical geometry+hessian harmonic model.

    This object is the single source of truth for the harmonic normal-coordinate
    convention used to build inertia derivatives. Keeping the full chain here
    avoids subtle inconsistencies between the mode normalization and the
    resulting first/second derivatives of I and I^{-1}.
    """

    masses_amu: np.ndarray
    representation: str
    xyz_to_abc: tuple[str, str, str]
    coords_com_ang: np.ndarray
    coords_pa_ang: np.ndarray
    principal_axes: np.ndarray
    moments_amu_a2: np.ndarray
    abc_mhz: np.ndarray
    vib_freq_cm: np.ndarray
    vib_vecs_mw_pa: np.ndarray
    coriolis_zeta_pairs_xyz: np.ndarray
    coriolis_g_au: np.ndarray
    coriolis_zeta_xyz: dict[str, np.ndarray]
    coriolis_m_tensor_au: np.ndarray
    quartic_pairpair_metric_au: np.ndarray
    i_tensor_au: np.ndarray
    dI_au: np.ndarray
    d2I_au: np.ndarray
    invI_au: np.ndarray
    dInv_au: np.ndarray
    d2Inv_bilinear_au: np.ndarray
    d2Inv_intrinsic_au: np.ndarray
    d2Inv_au: np.ndarray
    symbols: list[str] | None = None
    point_group: str | None = None


def _norm_rep(rep: str) -> str:
    rep = rep.strip().upper()
    if rep not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation '{rep}'. Use I, II, or III.")
    return rep


def representation_axis_labels(rep: str) -> tuple[str, str, str]:
    """Return the `(x,y,z)` to `(a,b,c)` correspondence for a representation."""
    rep = _norm_rep(rep)
    if rep == "I":
        return ("a", "b", "c")
    if rep == "II":
        return ("b", "c", "a")
    return ("c", "a", "b")


def representation_permutation(rep: str) -> np.ndarray:
    """Return column permutation mapping principal axes `(a,b,c)` to `(x,y,z)`."""
    labels = representation_axis_labels(rep)
    abc_to_idx = {"a": 0, "b": 1, "c": 2}
    return np.array([abc_to_idx[label] for label in labels], dtype=int)


def representation_component_permutation(rep_from: str, rep_to: str) -> np.ndarray:
    """Return axis-index permutation mapping tensor components from ``rep_from`` to ``rep_to``.

    If ``T_from`` is a rank-2 spatial tensor written in the `(x,y,z)` axes of
    ``rep_from``, then the same tensor in ``rep_to`` is obtained as

        T_to[a,b] = T_from[q[a], q[b]]

    where ``q = representation_component_permutation(rep_from, rep_to)``.
    """
    p_from = representation_permutation(rep_from)
    p_to = representation_permutation(rep_to)
    return np.argsort(p_from)[p_to]


def apply_representation_to_coords(coords_pa_ang: np.ndarray, rep: str) -> np.ndarray:
    """Permute principal-axis coordinates into the requested `(x,y,z)` representation."""
    perm = representation_permutation(rep)
    return coords_pa_ang[:, perm]


def apply_representation_to_hessian(hessian_pa: np.ndarray, n_atoms: int, rep: str) -> np.ndarray:
    """Permute a principal-axis Cartesian Hessian into the requested representation."""
    perm = representation_permutation(rep)
    block = np.eye(3, dtype=float)[:, perm]
    rot = np.zeros((3 * n_atoms, 3 * n_atoms), dtype=float)
    for i in range(n_atoms):
        rot[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = block
    return rot.T @ hessian_pa @ rot


def _norm_hand(h: str) -> str:
    h = h.strip().lower()
    if h not in HANDEDNESS:
        raise ValueError(f"Unknown handedness '{h}'. Use r or l.")
    return h


def _canonical_symbol(sym: str) -> str:
    s = sym.strip().upper()
    if s in SYMBOLS_CANONICAL:
        return SYMBOLS_CANONICAL[s]
    raise ValueError(f"Unknown chemical symbol '{sym}'.")


def _parse_atom_token(token: str) -> tuple[str, float]:
    """Parse atom token and return (canonical_symbol, mass_u).

    Supported isotope formats:
    - 13C, 18O, 2H
    - C-13, O18, H_2
    - D, T (aliases for 2H, 3H)

    Fallback for unspecified isotopes: average atomic mass.
    Fallback for unknown isotope exact mass: mass number as u.
    """
    tok = token.strip()
    up = tok.upper()

    if up == "D":
        return "H", ISOTOPE_MASSES[("H", 2)]
    if up == "T":
        return "H", ISOTOPE_MASSES[("H", 3)]

    if tok.isdigit():
        z = int(tok)
        if z in ATOMIC_NUMBER_TO_SYMBOL:
            sym = ATOMIC_NUMBER_TO_SYMBOL[z]
            return sym, ATOMIC_MASSES[sym.upper()]
        raise ValueError(f"Unknown atomic number '{tok}'.")

    m = re.fullmatch(r"(\d+)([A-Za-z]{1,2})", tok)
    if m:
        a = int(m.group(1))
        sym = _canonical_symbol(m.group(2))
        key = (sym.upper(), a)
        return sym, ISOTOPE_MASSES.get(key, float(a))

    m = re.fullmatch(r"([A-Za-z]{1,2})[-_]?([0-9]+)", tok)
    if m:
        sym = _canonical_symbol(m.group(1))
        a = int(m.group(2))
        key = (sym.upper(), a)
        return sym, ISOTOPE_MASSES.get(key, float(a))

    sym = _canonical_symbol(tok)
    return sym, ATOMIC_MASSES[sym.upper()]


# --- IO -----------------------------------------------------------------------------

def molecule_from_xyz_text(text: str) -> Molecule:
    """Build a molecule from XYZ text content.

    Format:
      N
      comment
      atom x y z [optional_mass_u]

    atom can be element symbol or isotope tag (13C, C-13, D, T...).
    If optional mass is provided, it overrides symbol/isotope-derived mass.
    """
    lines = text.strip().splitlines()
    if len(lines) < 3:
        raise ValueError("XYZ file too short.")

    n_atoms = int(lines[0].strip())
    rows = lines[2 : 2 + n_atoms]
    if len(rows) != n_atoms:
        raise ValueError("XYZ atom count does not match number of lines.")

    symbols: list[str] = []
    masses = np.zeros(n_atoms, dtype=float)
    coords = np.zeros((n_atoms, 3), dtype=float)

    for i, ln in enumerate(rows):
        parts = ln.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed XYZ atom line {i + 1}: '{ln}'")

        sym, mass = _parse_atom_token(parts[0])
        xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
        if len(parts) >= 5:
            mass = float(parts[4])

        symbols.append(sym)
        masses[i] = mass
        coords[i] = xyz

    return Molecule(symbols=symbols, masses_amu=masses, coords_ang=coords)


def read_xyz(path: str | Path) -> Molecule:
    """Read XYZ geometry from a file path."""
    return molecule_from_xyz_text(Path(path).read_text(encoding="utf-8"))


def read_hessian(path: str | Path, n_atoms: int) -> np.ndarray:
    """Read symmetric 3N x 3N Hessian from text (Hartree / Bohr^2)."""
    dim = 3 * n_atoms
    raw = Path(path).read_text(encoding="utf-8")
    vals = np.fromstring(raw.replace(",", " "), sep=" ", dtype=float)

    if vals.size == dim * dim:
        h = vals.reshape((dim, dim))
        return 0.5 * (h + h.T)

    ntri = dim * (dim + 1) // 2
    if vals.size == ntri:
        h = np.zeros((dim, dim), dtype=float)
        p = 0
        for i in range(dim):
            for j in range(i + 1):
                h[i, j] = vals[p]
                h[j, i] = vals[p]
                p += 1
        return h

    raise ValueError(f"Unexpected Hessian size: {vals.size}. Expected {dim*dim} or {ntri}.")


# --- Harmonic analysis --------------------------------------------------------------

def mass_weight_hessian(hessian: np.ndarray, masses_amu: np.ndarray) -> np.ndarray:
    """Construct Fmw = M^{-1/2} F M^{-1/2}.

    M is diagonal in Cartesian coordinates with entries m_i (in electron masses).
    This is the standard kinetic-energy metric for harmonic normal coordinates.
    """
    masses_au = masses_amu * AMU_TO_AU_MASS
    m_cart = np.repeat(masses_au, 3)
    scale = 1.0 / np.sqrt(m_cart)
    return hessian * np.outer(scale, scale)


def _orthonormal_tr_basis(
    masses_amu: np.ndarray,
    coords_ang: np.ndarray,
    *,
    linear: bool,
) -> np.ndarray:
    """Return an orthonormal MW translation/rotation basis in Cartesian space."""
    n_atoms = masses_amu.size
    sqrt_m = np.sqrt(masses_amu)
    basis: list[np.ndarray] = []

    for axis in range(3):
        vec = np.zeros(3 * n_atoms, dtype=float)
        for a in range(n_atoms):
            vec[3 * a + axis] = sqrt_m[a]
        basis.append(vec)

    for axis in range(3):
        e = np.zeros(3, dtype=float)
        e[axis] = 1.0
        vec = np.zeros(3 * n_atoms, dtype=float)
        for a in range(n_atoms):
            rot = np.cross(e, coords_ang[a]) * sqrt_m[a]
            vec[3 * a : 3 * a + 3] = rot
        basis.append(vec)

    cols: list[np.ndarray] = []
    for vec in basis:
        work = vec.copy()
        for col in cols:
            work -= np.dot(col, work) * col
        norm = np.linalg.norm(work)
        if norm > 1.0e-10:
            cols.append(work / norm)

    expected = 5 if linear else 6
    if len(cols) < expected:
        raise ValueError(
            f"TR basis rank {len(cols)} is smaller than expected {expected}. "
            "Check geometry/representation passed to normal_modes()."
        )
    return np.column_stack(cols[:expected])


def normal_modes(
    f_mw: np.ndarray,
    n_atoms: int,
    masses_amu: np.ndarray | None = None,
    coords_ang: np.ndarray | None = None,
    linear: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Diagonalize mass-weighted Hessian and return vibrational modes."""
    f_eff = np.array(f_mw, dtype=float, copy=True)
    if masses_amu is not None and coords_ang is not None:
        tr_basis = _orthonormal_tr_basis(masses_amu, coords_ang, linear=linear)
        projector = np.eye(f_eff.shape[0], dtype=float) - tr_basis @ tr_basis.T
        f_eff = projector @ f_eff @ projector
        f_eff = 0.5 * (f_eff + f_eff.T)

    evals, evecs = np.linalg.eigh(f_eff)

    n_remove = 5 if linear else 6
    idx_abs = np.argsort(np.abs(evals))
    keep = np.ones(evals.size, dtype=bool)
    keep[idx_abs[:n_remove]] = False

    vib_evals = evals[keep]
    vib_vecs = evecs[:, keep]

    order = np.argsort(np.abs(vib_evals))
    vib_evals = vib_evals[order]
    vib_vecs = vib_vecs[:, order]

    omega_au = np.sign(vib_evals) * np.sqrt(np.abs(vib_evals))
    vib_freq_cm = omega_au * AU_FREQ_TO_CMINV
    return vib_freq_cm, vib_vecs, evals, evecs


# --- Rotational constants -----------------------------------------------------------

def inertia_tensor(masses_amu: np.ndarray, coords_ang: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inertia tensor (amu*Angstrom^2) around COM."""
    m_tot = float(np.sum(masses_amu))
    com = (masses_amu[:, None] * coords_ang).sum(axis=0) / m_tot
    x = coords_ang - com

    I = np.zeros((3, 3), dtype=float)
    for m, r in zip(masses_amu, x):
        r2 = float(np.dot(r, r))
        I += m * (r2 * np.eye(3) - np.outer(r, r))
    return I, com


def mass_weighted_modes_to_cartesian(
    vib_vecs_mw: np.ndarray,
    masses_amu: np.ndarray,
) -> np.ndarray:
    """Convert mass-weighted normal modes to Cartesian displacements.

    The input eigenvectors must be the direct output of ``normal_modes`` after
    diagonalizing the mass-weighted Hessian. The returned array has the same
    shape ``(3N, n_modes)`` and gives Cartesian displacements in bohr per unit
    mass-weighted normal coordinate (atomic units).
    """
    masses_au = masses_amu * AMU_TO_AU_MASS
    scale = np.repeat(np.sqrt(masses_au), 3)
    return vib_vecs_mw / scale[:, None]


def inertia_derivative_tensors(
    masses_au: np.ndarray,
    coords_bohr: np.ndarray,
    mode_cart_bohr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return I, dI/dQ_k, and d^2I/(dQ_k dQ_l) in a fixed body frame.

    Parameters
    ----------
    masses_au
        Atomic masses in electron-mass atomic units, shape ``(N,)``.
    coords_bohr
        Equilibrium Cartesian coordinates in the body-fixed frame, shape
        ``(N,3)``.
    mode_cart_bohr
        Cartesian derivatives ``dr/dQ_k`` in the same frame, shape
        ``(3N, n_modes)`` or ``(N,3,n_modes)``.

    Notes
    -----
    The formulas assume a linear normal-coordinate embedding
    ``r(Q) = r_0 + sum_k (dr/dQ_k) Q_k``. Under that assumption the second
    coordinate derivatives vanish and the second inertia derivative is built
    entirely from quadratic products of ``dr/dQ``.
    """
    n_atoms = masses_au.size
    if coords_bohr.shape != (n_atoms, 3):
        raise ValueError(f"coords_bohr must have shape {(n_atoms, 3)}, got {coords_bohr.shape}")

    if mode_cart_bohr.ndim == 2:
        if mode_cart_bohr.shape[0] != 3 * n_atoms:
            raise ValueError(
                f"mode_cart_bohr must have shape {(3 * n_atoms, 'n_modes')} or {(n_atoms, 3, 'n_modes')}, "
                f"got {mode_cart_bohr.shape}"
            )
        modes = mode_cart_bohr.reshape(n_atoms, 3, mode_cart_bohr.shape[1])
    elif mode_cart_bohr.ndim == 3 and mode_cart_bohr.shape[:2] == (n_atoms, 3):
        modes = mode_cart_bohr
    else:
        raise ValueError(
            f"mode_cart_bohr must have shape {(3 * n_atoms, 'n_modes')} or {(n_atoms, 3, 'n_modes')}, "
            f"got {mode_cart_bohr.shape}"
        )

    n_modes = modes.shape[2]
    I0 = np.zeros((3, 3), dtype=float)
    dI = np.zeros((3, 3, n_modes), dtype=float)
    d2I = np.zeros((3, 3, n_modes, n_modes), dtype=float)

    eye3 = np.eye(3)
    for a in range(n_atoms):
        m = float(masses_au[a])
        r = coords_bohr[a]
        r2 = float(np.dot(r, r))
        I0 += m * (r2 * eye3 - np.outer(r, r))

        for k in range(n_modes):
            u = modes[a, :, k]
            dI[:, :, k] += m * (2.0 * float(np.dot(r, u)) * eye3 - np.outer(r, u) - np.outer(u, r))

        for k in range(n_modes):
            uk = modes[a, :, k]
            for l in range(n_modes):
                ul = modes[a, :, l]
                d2I[:, :, k, l] += m * (2.0 * float(np.dot(uk, ul)) * eye3 - np.outer(uk, ul) - np.outer(ul, uk))

    return I0, dI, d2I


def inverse_inertia_derivatives(
    i_tensor: np.ndarray,
    dI: np.ndarray,
    d2I: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return I^{-1}, d(I^{-1})/dQ_k, and d^2(I^{-1})/(dQ_k dQ_l)."""
    if i_tensor.shape != (3, 3):
        raise ValueError(f"i_tensor must have shape (3,3), got {i_tensor.shape}")
    if dI.shape[:2] != (3, 3):
        raise ValueError(f"dI must start with shape (3,3,*), got {dI.shape}")
    if d2I.shape[:2] != (3, 3):
        raise ValueError(f"d2I must start with shape (3,3,*,*), got {d2I.shape}")

    n_modes = dI.shape[2]
    svals = np.linalg.svd(i_tensor, compute_uv=False)
    if svals[-1] <= max(1.0e-12, 1.0e-12 * svals[0]):
        invI = np.linalg.pinv(i_tensor, rcond=1.0e-12)
    else:
        invI = np.linalg.inv(i_tensor)
    d_inv = np.zeros((3, 3, n_modes), dtype=float)
    d2_inv = np.zeros((3, 3, n_modes, n_modes), dtype=float)

    for k in range(n_modes):
        d_inv[:, :, k] = -invI @ dI[:, :, k] @ invI

    for k in range(n_modes):
        for l in range(n_modes):
            d2_inv[:, :, k, l] = (
                invI @ dI[:, :, k] @ invI @ dI[:, :, l] @ invI
                + invI @ dI[:, :, l] @ invI @ dI[:, :, k] @ invI
                - invI @ d2I[:, :, k, l] @ invI
            )

    return invI, d_inv, d2_inv


def inverse_inertia_second_derivative_components(
    i_tensor: np.ndarray,
    dI: np.ndarray,
    d2I: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the canonical split of d^2(I^-1)/(dQ_k dQ_l).

    For each mode pair ``(k,l)``, the second inverse-inertia derivative is

        mu2_{kl} = B_{kl} - N_{kl}

    with

        B_{kl} = I^-1 (dI_k) I^-1 (dI_l) I^-1
               + I^-1 (dI_l) I^-1 (dI_k) I^-1

        N_{kl} = I^-1 (d2I_{kl}) I^-1.

    The returned tuple is ``(invI, bilinear, intrinsic, total)`` where
    ``total = bilinear - intrinsic``.
    """
    if i_tensor.shape != (3, 3):
        raise ValueError(f"i_tensor must have shape (3,3), got {i_tensor.shape}")
    if dI.shape[:2] != (3, 3):
        raise ValueError(f"dI must start with shape (3,3,*), got {dI.shape}")
    if d2I.shape[:2] != (3, 3):
        raise ValueError(f"d2I must start with shape (3,3,*,*), got {d2I.shape}")

    n_modes = dI.shape[2]
    svals = np.linalg.svd(i_tensor, compute_uv=False)
    if svals[-1] <= max(1.0e-12, 1.0e-12 * svals[0]):
        invI = np.linalg.pinv(i_tensor, rcond=1.0e-12)
    else:
        invI = np.linalg.inv(i_tensor)
    bilinear = np.zeros((3, 3, n_modes, n_modes), dtype=float)
    intrinsic = np.zeros((3, 3, n_modes, n_modes), dtype=float)
    total = np.zeros((3, 3, n_modes, n_modes), dtype=float)

    for k in range(n_modes):
        for l in range(n_modes):
            bilinear[:, :, k, l] = (
                invI @ dI[:, :, k] @ invI @ dI[:, :, l] @ invI
                + invI @ dI[:, :, l] @ invI @ dI[:, :, k] @ invI
            )
            intrinsic[:, :, k, l] = invI @ d2I[:, :, k, l] @ invI
            total[:, :, k, l] = bilinear[:, :, k, l] - intrinsic[:, :, k, l]

    return invI, bilinear, intrinsic, total


def inverse_inertia_bilinear_from_mu1(
    i_tensor: np.ndarray,
    d_inv: np.ndarray,
) -> np.ndarray:
    """Return the bilinear part of d^2(I^-1) from ``mu1 = d(I^-1)/dQ`` alone.

    Writing ``mu1_k = d(I^-1)/dQ_k``, the bilinear contribution to the second
    inverse-inertia derivative can be rewritten as

        B_{kl} = mu1_k I mu1_l + mu1_l I mu1_k,

    where ``I`` is the equilibrium inertia tensor.
    """
    if i_tensor.shape != (3, 3):
        raise ValueError(f"i_tensor must have shape (3,3), got {i_tensor.shape}")
    if d_inv.shape[:2] != (3, 3):
        raise ValueError(f"d_inv must start with shape (3,3,*), got {d_inv.shape}")

    n_modes = d_inv.shape[2]
    bilinear = np.zeros((3, 3, n_modes, n_modes), dtype=float)
    for k in range(n_modes):
        for l in range(n_modes):
            bilinear[:, :, k, l] = d_inv[:, :, k] @ i_tensor @ d_inv[:, :, l] + d_inv[:, :, l] @ i_tensor @ d_inv[:, :, k]
    return bilinear


def inverse_inertia_derivatives_from_normal_modes(
    masses_amu: np.ndarray,
    coords_pa_ang: np.ndarray,
    vib_vecs_mw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper returning μ, μ_{,k}, and μ_{,kl}.

    Parameters
    ----------
    masses_amu
        Atomic masses in amu.
    coords_pa_ang
        Equilibrium coordinates in the principal-axis body frame, in angstrom.
    vib_vecs_mw
        Mass-weighted normal-mode eigenvectors returned by ``normal_modes`` in
        the same principal-axis Cartesian frame.
    """
    masses_au = masses_amu * AMU_TO_AU_MASS
    coords_bohr = coords_pa_ang * ANGSTROM_TO_BOHR
    mode_cart = mass_weighted_modes_to_cartesian(vib_vecs_mw, masses_amu)
    I0, dI, d2I = inertia_derivative_tensors(masses_au, coords_bohr, mode_cart)
    return inverse_inertia_derivatives(I0, dI, d2I)


def harmonic_inertia_model_from_geometry_hessian(
    masses_amu: np.ndarray,
    coords_ang: np.ndarray,
    hessian: np.ndarray,
    *,
    linear: bool | None = None,
    representation: str = "I",
    symbols: list[str] | None = None,
    point_group: str | None = None,
) -> HarmonicInertiaModel:
    """Build the canonical harmonic model from geometry + Cartesian Hessian.

    The returned object preserves a single, self-consistent normal-coordinate
    convention across:
    - harmonic frequencies and eigenvectors,
    - first/second derivatives of the inertia tensor,
    - first/second derivatives of the inverse inertia tensor.
    """
    rep = _norm_rep(representation)
    i_tensor, com = inertia_tensor(masses_amu, coords_ang)
    moments, abc_mhz, principal_axes = rotational_constants(i_tensor)

    coords_com = coords_ang - com
    coords_pa = coords_com @ principal_axes

    n_atoms = masses_amu.size
    perm = representation_permutation(rep)
    rot = np.zeros((3 * n_atoms, 3 * n_atoms), dtype=float)
    for i in range(n_atoms):
        rot[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = principal_axes.T
    hessian_pa = rot @ hessian @ rot.T
    coords_rep = apply_representation_to_coords(coords_pa, rep)
    hessian_rep = apply_representation_to_hessian(hessian_pa, n_atoms, rep)

    if linear is None:
        linear = float(np.min(moments)) < 1.0e-8

    f_mw = mass_weight_hessian(hessian_rep, masses_amu)
    vib_freq_cm, vib_vecs_mw_pa, _, _ = normal_modes(
        f_mw,
        n_atoms,
        masses_amu=masses_amu,
        coords_ang=coords_rep,
        linear=linear,
    )

    masses_au = masses_amu * AMU_TO_AU_MASS
    coords_bohr = coords_rep * ANGSTROM_TO_BOHR
    mode_cart_bohr = mass_weighted_modes_to_cartesian(vib_vecs_mw_pa, masses_amu)
    i0_au, dI_au, d2I_au = inertia_derivative_tensors(masses_au, coords_bohr, mode_cart_bohr)
    invI_au, d2Inv_bilinear_au, d2Inv_intrinsic_au, d2Inv_au = inverse_inertia_second_derivative_components(
        i0_au, dI_au, d2I_au
    )
    dInv_au = np.zeros((3, 3, dI_au.shape[2]), dtype=float)
    for k in range(dI_au.shape[2]):
        dInv_au[:, :, k] = -invI_au @ dI_au[:, :, k] @ invI_au
    coriolis_zeta_pairs = coriolis_zeta_tensor(vib_vecs_mw_pa)
    coriolis_g_au = coriolis_vectors(masses_amu, coords_rep, vib_vecs_mw_pa)
    moments_xyz = moments[perm]
    coriolis_zeta = coriolis_constants_from_vectors(coriolis_g_au, moments_xyz)
    coriolis_m_au = coriolis_m_tensor(coriolis_g_au, vib_freq_cm)
    quartic_pairpair_metric_au = quartic_pairpair_metric(dInv_au, vib_freq_cm)

    return HarmonicInertiaModel(
        masses_amu=masses_amu.copy(),
        representation=rep,
        xyz_to_abc=representation_axis_labels(rep),
        coords_com_ang=coords_com,
        coords_pa_ang=coords_rep,
        principal_axes=principal_axes,
        moments_amu_a2=moments,
        abc_mhz=abc_mhz,
        vib_freq_cm=vib_freq_cm,
        vib_vecs_mw_pa=vib_vecs_mw_pa,
        coriolis_zeta_pairs_xyz=coriolis_zeta_pairs,
        coriolis_g_au=coriolis_g_au,
        coriolis_zeta_xyz=coriolis_zeta,
        coriolis_m_tensor_au=coriolis_m_au,
        quartic_pairpair_metric_au=quartic_pairpair_metric_au,
        i_tensor_au=i0_au,
        dI_au=dI_au,
        d2I_au=d2I_au,
        invI_au=invI_au,
        dInv_au=dInv_au,
        d2Inv_bilinear_au=d2Inv_bilinear_au,
        d2Inv_intrinsic_au=d2Inv_intrinsic_au,
        d2Inv_au=d2Inv_au,
        symbols=None if symbols is None else list(symbols),
        point_group=point_group,
    )


def rotational_constants(i_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Principal moments, rotational constants (MHz), principal-axis matrix."""
    moments, vecs = np.linalg.eigh(i_tensor)
    idx = np.argsort(moments)
    moments = moments[idx]
    vecs = vecs[:, idx]

    if np.linalg.det(vecs) < 0.0:
        vecs[:, 2] *= -1.0

    abc_mhz = np.empty_like(moments)
    np.divide(
        ROT_CONST_MHZ_AMU_A2,
        moments,
        out=abc_mhz,
        where=np.abs(moments) > 1.0e-30,
    )
    abc_mhz[np.abs(moments) <= 1.0e-30] = np.inf
    return moments, abc_mhz, vecs


# --- Coriolis / quartic constants ---------------------------------------------------

def coriolis_vectors(
    masses_amu: np.ndarray,
    coords_pa_ang: np.ndarray,
    vib_vecs_mw: np.ndarray,
) -> np.ndarray:
    """Return harmonic Coriolis vectors G_k in the body-fixed frame.

    Parameters
    ----------
    masses_amu
        Atomic masses in amu.
    coords_pa_ang
        Equilibrium coordinates in the body-fixed Cartesian frame, in angstrom.
    vib_vecs_mw
        Mass-weighted harmonic normal modes, shape ``(3N, n_modes)``.

    Returns
    -------
    np.ndarray
        Array of shape ``(3, n_modes)`` containing the Coriolis vectors
        ``G_k`` in atomic units.

    Notes
    -----
    For mass-weighted eigenvectors ``e_{ik}``, the Coriolis vector is

    ``G_k = sum_i sqrt(m_i) (r_i x e_{ik})``

    in atomic units. This is the coordinate-independent rotational object used
    in the tensorial formulation of the quartic and sextic distortion terms.
    """
    n_atoms = masses_amu.size
    masses_au = masses_amu * AMU_TO_AU_MASS
    coords_bohr = coords_pa_ang * ANGSTROM_TO_BOHR
    g = np.zeros((3, vib_vecs_mw.shape[1]), dtype=float)
    for k in range(vib_vecs_mw.shape[1]):
        for i in range(n_atoms):
            e_i = vib_vecs_mw[3 * i : 3 * i + 3, k]
            g[:, k] += np.sqrt(masses_au[i]) * np.cross(coords_bohr[i], e_i)
    return g


def coriolis_constants_from_vectors(
    coriolis_g_au: np.ndarray,
    moments_xyz_amu_a2: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return Coriolis zeta constants in the current Cartesian x,y,z representation."""
    moments_au = moments_xyz_amu_a2 * AMU_TO_AU_MASS * (ANGSTROM_TO_BOHR**2)
    def _safe_div(num: np.ndarray, den: float) -> np.ndarray:
        if den <= 1.0e-20:
            return np.zeros_like(num, dtype=float)
        return num / den

    z_xy = _safe_div(coriolis_g_au[2], float(np.sqrt(max(moments_au[0] * moments_au[1], 0.0))))
    z_yz = _safe_div(coriolis_g_au[0], float(np.sqrt(max(moments_au[1] * moments_au[2], 0.0))))
    z_zx = _safe_div(coriolis_g_au[1], float(np.sqrt(max(moments_au[2] * moments_au[0], 0.0))))
    return {"xy": z_xy, "yz": z_yz, "zx": z_zx}


def coriolis_m_tensor(
    coriolis_g_au: np.ndarray,
    vib_freq_cm: np.ndarray,
) -> np.ndarray:
    """Return the symmetric Coriolis tensor M_{mu nu} = sum_k G_{k,mu} G_{k,nu} / omega_k^2."""
    omega_au = np.abs(vib_freq_cm) / AU_FREQ_TO_CMINV
    m_tensor = np.zeros((3, 3), dtype=float)
    for k, omega_k in enumerate(omega_au):
        if omega_k <= 1.0e-14:
            continue
        gk = coriolis_g_au[:, k]
        m_tensor += np.outer(gk, gk) / (omega_k * omega_k)
    return m_tensor


def inverse_inertia_pair_basis_vectors(dInv_au: np.ndarray) -> np.ndarray:
    """Return mode vectors in the 6D symmetric pair basis of d(I^{-1})/dQ.

    The basis is ordered as ``(xx, yy, zz, xy, xz, yz)``. For each mode k we
    define

        u_k = (mu_xx, mu_yy, mu_zz, sqrt(3) mu_xy, sqrt(3) mu_xz, sqrt(3) mu_yz)

    where ``mu_ab = d(I^{-1})_{ab} / dQ_k`` in the current Cartesian frame.

    The sqrt(3) factors are fixed so that the harmonic H12H12 commuting quartic
    tensor follows from the Gram matrix

        G^(4) = sum_k u_k u_k^T / omega_k^2

    through

        tau_xxxx = -1/8 G^(4)_{11}
        tau_yyyy = -1/8 G^(4)_{22}
        tau_zzzz = -1/8 G^(4)_{33}
        tau_xxyy = -1/8 (G^(4)_{12} + G^(4)_{44})
        tau_xxzz = -1/8 (G^(4)_{13} + G^(4)_{55})
        tau_yyzz = -1/8 (G^(4)_{23} + G^(4)_{66})

    This makes explicit that the natural harmonic quartic spatial object lives
    in the 6D symmetric-pair space, not in the 3D space of Coriolis vectors.
    """
    n_modes = dInv_au.shape[2]
    out = np.zeros((6, n_modes), dtype=float)
    rt3 = np.sqrt(3.0)
    for k in range(n_modes):
        out[:, k] = [
            float(dInv_au[0, 0, k]),
            float(dInv_au[1, 1, k]),
            float(dInv_au[2, 2, k]),
            float(rt3 * dInv_au[0, 1, k]),
            float(rt3 * dInv_au[0, 2, k]),
            float(rt3 * dInv_au[1, 2, k]),
        ]
    return out


def quartic_pairpair_metric(
    dInv_au: np.ndarray,
    vib_freq_cm: np.ndarray,
) -> np.ndarray:
    """Return the 6x6 harmonic quartic spatial metric in the symmetric pair basis.

    This metric is the natural H12H12 analogue of the Coriolis 3x3 tensor, but
    it lives in ``Sym^2(R^3)`` rather than ``R^3``. It therefore contains the
    pair-pair information needed by the compressed quartic tensor, whereas the
    3x3 tensor ``sum_k G_k G_k^T / omega_k^2`` contains only a lower-dimensional
    invariant contraction of the same harmonic mode data.
    """
    omega_au = np.abs(vib_freq_cm) / AU_FREQ_TO_CMINV
    basis_vecs = inverse_inertia_pair_basis_vectors(dInv_au)
    gram = np.zeros((6, 6), dtype=float)
    for k, omega_k in enumerate(omega_au):
        if omega_k <= 1.0e-14:
            continue
        uk = basis_vecs[:, k]
        gram += np.outer(uk, uk) / (omega_k * omega_k)
    return gram


def coriolis_zeta_tensor(vib_vecs_mw: np.ndarray) -> np.ndarray:
    """Return pairwise Coriolis couplings zeta[axis, i, j] in the current x,y,z frame."""
    n_modes = vib_vecs_mw.shape[1]
    n_atoms = vib_vecs_mw.shape[0] // 3
    vib = vib_vecs_mw.reshape(n_atoms, 3, n_modes)
    zeta = np.zeros((3, n_modes, n_modes), dtype=float)
    for i in range(n_modes):
        for j in range(i + 1):
            zx = 0.0
            zy = 0.0
            zz = 0.0
            for atom in range(n_atoms):
                zx += vib[atom, 1, i] * vib[atom, 2, j] - vib[atom, 2, i] * vib[atom, 1, j]
                zy += vib[atom, 2, i] * vib[atom, 0, j] - vib[atom, 0, i] * vib[atom, 2, j]
                zz += vib[atom, 0, i] * vib[atom, 1, j] - vib[atom, 1, i] * vib[atom, 0, j]
            zeta[0, i, j] = zx
            zeta[1, i, j] = zy
            zeta[2, i, j] = zz
            zeta[0, j, i] = -zx
            zeta[1, j, i] = -zy
            zeta[2, j, i] = -zz
    return zeta

def coriolis_constants(
    masses_amu: np.ndarray,
    coords_pa_ang: np.ndarray,
    vib_vecs_cart: np.ndarray,
    moments_amu_a2: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute zeta_{ab,k}, zeta_{bc,k}, zeta_{ca,k}.

    Coriolis constants are built from the rovibrational kinetic coupling
    between rotational angular momentum and mass-weighted normal coordinates.
    They are evaluated in the principal-axis frame.
    """
    g = coriolis_vectors(masses_amu, coords_pa_ang, vib_vecs_cart)
    return coriolis_constants_from_vectors(g, moments_amu_a2)


def _quartic_basis_terms(ja: float, jb: float, jc: float) -> np.ndarray:
    j2 = ja * ja + jb * jb + jc * jc
    delta = jb * jb - jc * jc
    return np.array(
        [
            -(j2 * j2),
            -(j2 * ja * ja),
            -(ja * ja * ja * ja),
            -(j2 * delta),
            -(ja * ja * delta),
        ],
        dtype=float,
    )


def quartic_distortion_constants(
    vib_freq_cm: np.ndarray,
    coriolis: dict[str, np.ndarray],
    abc_mhz: np.ndarray,
    n_samples: int = 900,
    seed: int = 7,
) -> np.ndarray:
    """Compute Watson quartic constants from Coriolis + frequencies.

    The second-order rovibrational perturbative quartic energy is modeled as:

        E^(4)(J) = - sum_k [ Q_k(J)^2 / nu_k ]

    where Q_k is bilinear in rotational components and weighted by
    Coriolis constants and rotational constants. The resulting quartic
    polynomial is then projected onto the Watson A-reduction basis.
    """
    nu_mhz = np.abs(vib_freq_cm) * CMINV_TO_MHZ
    ok = nu_mhz > 1e-8
    if not np.any(ok):
        raise ValueError("No valid vibrational frequencies available.")

    A, B, C = [float(x) for x in abc_mhz]
    z_ab = coriolis["ab"][ok]
    z_bc = coriolis["bc"][ok]
    z_ca = coriolis["ca"][ok]
    nus = nu_mhz[ok]

    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, 5), dtype=float)
    y = np.zeros(n_samples, dtype=float)

    for i in range(n_samples):
        ja, jb, jc = rng.uniform(-8.0, 8.0, size=3)
        q = A * B * z_ab * (ja * jb) + B * C * z_bc * (jb * jc) + C * A * z_ca * (jc * ja)
        e4 = -np.sum((q * q) / nus)

        X[i] = _quartic_basis_terms(ja, jb, jc)
        y[i] = e4

    d, *_ = np.linalg.lstsq(X, y, rcond=None)
    return d


# --- Yamada transforms ---------------------------------------------------------------

def get_Q(rep: str, A: float, B: float, C: float) -> np.ndarray:
    rep = _norm_rep(rep)

    r1 = np.array([-1.0, -1.0, -1.0, 0.0, 0.0])
    r2 = np.array([-1.0, 0.0, 0.0, -2.0, 0.0])
    r3 = np.array([-1.0, 0.0, 0.0, 2.0, 0.0])
    r4 = np.array([-3.0, -1.0, 0.0, 0.0, 0.0])

    if rep == "I":
        first3 = (r1, r2, r3)
        r5 = np.array([A + B + C, -(B + C) / 2.0, 0.0, B - C, B - C])
    elif rep == "II":
        first3 = (r3, r1, r2)
        r5 = np.array([A + B + C, -(A + B) / 2.0, 0.0, A - B, A - B])
    else:
        first3 = (r2, r3, r1)
        r5 = np.array([A + B + C, -(C + A) / 2.0, 0.0, C - A, C - A])

    return np.vstack([*first3, r4, r5])


def transform_constants(
    d: np.ndarray,
    A: float,
    B: float,
    C: float,
    rep_in: str,
    rep_out: str,
) -> np.ndarray:
    rep_in = _norm_rep(rep_in)
    rep_out = _norm_rep(rep_out)
    d = np.asarray(d, dtype=float).reshape(5)

    q_in = get_Q(rep_in, A, B, C)
    q_out = get_Q(rep_out, A, B, C)
    return np.linalg.solve(q_out, q_in @ d)


def transform_rl(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float).reshape(5)
    AJ, AJK, AK, dJ, dK = d
    return np.array([AJ, -AJK, AK, -dJ, dK], dtype=float)


# --- Driver -------------------------------------------------------------------------

def _is_linear(moments: np.ndarray, tol: float = 1e-8) -> bool:
    return float(np.min(moments)) < tol


def _fmt(v: np.ndarray, fmt: str = "{: .6e}") -> str:
    return "[" + ", ".join(fmt.format(float(x)) for x in v) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(description="Rovibrational quartic distortion from XYZ + Hessian.")
    ap.add_argument("xyz", help="XYZ file (Angstrom). Atom labels can include isotopes, e.g. 13C or C-13.")
    ap.add_argument("hessian", help="Cartesian Hessian file (Hartree/Bohr^2).")
    ap.add_argument("--rep", choices=REPRESENTATIONS, default="I", help="Output representation.")
    ap.add_argument("--hand", choices=HANDEDNESS, default="r", help="Output handedness.")
    ap.add_argument(
        "--print-inverse-inertia-derivatives",
        action="store_true",
        help="Print I^{-1}, d(I^{-1})/dQ_k, and d^2(I^{-1})/(dQ_k dQ_l) in atomic units.",
    )
    args = ap.parse_args()

    mol = read_xyz(args.xyz)
    h = read_hessian(args.hessian, mol.n_atoms)

    I, com = inertia_tensor(mol.masses_amu, mol.coords_ang)
    moments, abc, pax = rotational_constants(I)

    coords_com = mol.coords_ang - com
    coords_pa = coords_com @ pax

    # Rotate Hessian to principal-axis Cartesian frame atom-by-atom.
    n = mol.n_atoms
    T = np.zeros((3 * n, 3 * n), dtype=float)
    for i in range(n):
        T[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = pax.T
    h_pa = T @ h @ T.T

    f_mw = mass_weight_hessian(h_pa, mol.masses_amu)
    vib_freq_cm, vib_vecs, _, _ = normal_modes(
        f_mw,
        mol.n_atoms,
        masses_amu=mol.masses_amu,
        coords_ang=coords_pa,
        linear=_is_linear(moments),
    )

    if args.print_inverse_inertia_derivatives:
        invI, dInv, d2Inv = inverse_inertia_derivatives_from_normal_modes(mol.masses_amu, coords_pa, vib_vecs)
        print("Inverse inertia tensor I^{-1} [a.u.]")
        print(invI)
        print()
        print("First derivatives d(I^{-1})/dQ_k [a.u.]")
        for k in range(dInv.shape[2]):
            print(f"mode {k+1:3d}")
            print(dInv[:, :, k])
        print()
        print("Second derivatives d^2(I^{-1})/(dQ_k dQ_l) [a.u.]")
        for k in range(d2Inv.shape[2]):
            for l in range(d2Inv.shape[3]):
                print(f"modes ({k+1:3d},{l+1:3d})")
                print(d2Inv[:, :, k, l])
        print()

    zeta = coriolis_constants(mol.masses_amu, coords_pa, vib_vecs, moments)
    d_i_r = quartic_distortion_constants(vib_freq_cm, zeta, abc)

    d_i = d_i_r if args.hand == "r" else transform_rl(d_i_r)
    d_target = transform_constants(d_i, abc[0], abc[1], abc[2], "I", args.rep)

    print("Rotational constants (MHz)")
    print(f"A = {abc[0]:.10f}")
    print(f"B = {abc[1]:.10f}")
    print(f"C = {abc[2]:.10f}\n")

    print("Harmonic frequencies (cm^-1)")
    for i, nu in enumerate(vib_freq_cm, start=1):
        print(f"mode {i:3d}: {nu: .6f}")
    print()

    print("Coriolis constants (zeta_ab, zeta_bc, zeta_ca)")
    for i in range(vib_freq_cm.size):
        print(
            f"mode {i+1:3d}: "
            f"{zeta['ab'][i]: .6f}  {zeta['bc'][i]: .6f}  {zeta['ca'][i]: .6f}"
        )
    print()

    names = ("AJ", "AJK", "AK", "dJ", "dK")
    print("Watson quartic constants (I,r) [MHz]")
    for nm, vv in zip(names, d_i_r):
        print(f"{nm:3s} = {vv: .8e}")
    print()

    print(f"Watson quartic constants ({args.rep},{args.hand}) [MHz]")
    for nm, vv in zip(names, d_target):
        print(f"{nm:3s} = {vv: .8e}")
    print()

    print("Transforms from I,r")
    for rep in REPRESENTATIONS:
        vals = transform_constants(d_i_r, abc[0], abc[1], abc[2], "I", rep)
        print(f"{rep},r { _fmt(vals) }")


if __name__ == "__main__":
    main()
