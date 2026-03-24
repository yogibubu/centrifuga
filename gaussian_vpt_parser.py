#!/usr/bin/env python3
"""Parse the Gaussian/GDV harmonic+anharmonic data needed by the VPT4 workflow."""

from __future__ import annotations

import argparse
import re
from itertools import permutations
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_DNUM_RE = re.compile(r"[Dd]")
CMINV_TO_MHZ = 29979.2458


def _to_float(token: str) -> float:
    return float(_DNUM_RE.sub("E", token))


@dataclass
class GaussianHarmonicData:
    atomic_numbers: np.ndarray
    masses_amu: np.ndarray
    coords_std_ang: np.ndarray
    principal_axes: np.ndarray
    moments_au: np.ndarray
    rot_ghz: np.ndarray
    frequencies_cm: np.ndarray
    reduced_masses_amu: np.ndarray
    force_constants_mdyne_a: np.ndarray
    normal_modes: np.ndarray
    mode_symmetry_labels: tuple[str, ...] | None = None
    harmonic_to_anharmonic: np.ndarray | None = None

    def principal_axis_coordinates(self) -> np.ndarray:
        """Equilibrium coordinates rotated into the principal-axis frame."""
        return self.coords_std_ang @ self.principal_axes

    def principal_axis_modes(self) -> np.ndarray:
        """Normal-mode Cartesian displacements rotated into the principal-axis frame."""
        n_atoms = self.atomic_numbers.size
        out = np.zeros_like(self.normal_modes)
        for a in range(n_atoms):
            out[3 * a : 3 * a + 3, :] = self.principal_axes.T @ self.normal_modes[3 * a : 3 * a + 3, :]
        return out

    def mass_weighted_modes(self) -> np.ndarray:
        """Reconstruct mass-weighted harmonic eigenvectors from Gaussian mode printout.

        Gaussian prints Cartesian normal-coordinate displacements ``L`` with the
        conventional normalization
            sum_a m_a |L_{ak}|^2 = mu_k,
        where ``mu_k`` is the reduced mass of mode ``k``. The corresponding
        mass-weighted eigenvectors are therefore
            e_{ak} = sqrt(m_a / mu_k) L_{ak}.
        """
        modes_pa = self.principal_axis_modes()
        n_atoms = self.atomic_numbers.size
        out = np.zeros_like(modes_pa)
        for k in range(self.reduced_masses_amu.size):
            scale = np.sqrt(self.masses_amu / self.reduced_masses_amu[k])
            for a in range(n_atoms):
                out[3 * a : 3 * a + 3, k] = scale[a] * modes_pa[3 * a : 3 * a + 3, k]
        return out

    def reordered_to_anharmonic(self) -> "GaussianHarmonicData":
        """Return a copy reordered to Gaussian's anharmonic/vibro-rot print order."""
        if self.harmonic_to_anharmonic is None:
            return self

        order = np.asarray(self.harmonic_to_anharmonic, dtype=int)
        return GaussianHarmonicData(
            atomic_numbers=self.atomic_numbers.copy(),
            masses_amu=self.masses_amu.copy(),
            coords_std_ang=self.coords_std_ang.copy(),
            principal_axes=self.principal_axes.copy(),
            moments_au=self.moments_au.copy(),
            rot_ghz=self.rot_ghz.copy(),
            frequencies_cm=self.frequencies_cm[order].copy(),
            reduced_masses_amu=self.reduced_masses_amu[order].copy(),
            force_constants_mdyne_a=self.force_constants_mdyne_a[order].copy(),
            normal_modes=self.normal_modes[:, order].copy(),
            mode_symmetry_labels=None if self.mode_symmetry_labels is None else tuple(self.mode_symmetry_labels[idx] for idx in order),
            harmonic_to_anharmonic=np.arange(order.size, dtype=int),
        )


@dataclass
class GaussianQuarticBenchmark:
    alpha_mode_indices: np.ndarray
    alpha_cm: np.ndarray
    alpha_mhz: np.ndarray
    alpha_axis_labels: tuple[str, str, str]
    tau_prime_cm: dict[str, float]
    tau_prime_mhz: dict[str, float]
    a_reduction_mhz: dict[str, float]
    s_reduction_mhz: dict[str, float]
    sigma: float
    kappa: float
    delta: float
    didq_amu_sqrt_ang: np.ndarray | None = None
    spectroscopic_axes: dict[str, int] | None = None


@dataclass
class GaussianAlphaData:
    mode_indices: np.ndarray
    alpha_cm: np.ndarray
    alpha_mhz: np.ndarray
    axis_labels: tuple[str, str, str]


@dataclass
class GaussianSexticBenchmark:
    phi_cart_cm: dict[str, float]
    phi_cart_hz: dict[str, float]
    a_reduction_cm: dict[str, float]
    a_reduction_hz: dict[str, float]
    s_reduction_cm: dict[str, float]
    s_reduction_hz: dict[str, float]
    tau_cm: np.ndarray | None = None
    c1: np.ndarray | None = None
    c2: np.ndarray | None = None
    c2_reordered_to_print: np.ndarray | None = None
    rho: float | None = None
    mu: float | None = None
    nu: float | None = None
    lam: float | None = None


@dataclass
class GaussianLinearLTypeConstants:
    q_e_cm: dict[int, float]
    q_e_mhz: dict[int, float]
    q_j_cm: dict[int, float]
    q_j_mhz: dict[int, float]
    q_k_cm: dict[int, float]
    q_k_mhz: dict[int, float]
    active_dd_22_count: int | None = None


@dataclass
class GaussianLinearRotDistConstants:
    d_mhz: float | None
    h_mhz: float | None


@dataclass
class GaussianFchkHarmonicData:
    atomic_numbers: np.ndarray
    masses_amu: np.ndarray
    coords_bohr: np.ndarray
    n_modes: int
    vib_atmass_amu: np.ndarray
    vib_e2: np.ndarray
    vib_modes: np.ndarray
    cartesian_force_constants: np.ndarray
    point_group: str | None = None


@dataclass
class GaussianAnharmonicForceData:
    frequencies_cm: np.ndarray
    phi3_reduced_cm: np.ndarray
    phi3_raw_au: np.ndarray
    phi4_reduced_cm: np.ndarray
    phi4_raw_au: np.ndarray


@dataclass
class GaussianResonanceEntry:
    kind: str
    lhs_modes: tuple[int, ...]
    rhs_modes: tuple[int, ...]
    freq_diff_cm: float
    metric_1: float | None
    metric_2: float | None
    status: str


@dataclass
class GaussianVariationalOverlap:
    dvpt2_state: str
    overlap: float
    variational_state_index: int


@dataclass
class GaussianVariationalEnergy:
    dvpt2_state: str
    deperturbed_energy_cm: float
    after_diag_energy_cm: float


@dataclass
class GaussianVariationalStateDefinition:
    variational_state_index: int
    coefficient: float
    dvpt2_state: str


@dataclass
class GaussianFundamentalBand:
    mode_index: int
    status: str
    harmonic_cm: float
    anharmonic_cm: float
    overlap_flag: str | None = None


@dataclass
class GaussianAnharmonicAnalysis:
    pt2_model: str
    x_coriolis_cm: np.ndarray
    x_third_derivative_cm: np.ndarray
    x_fourth_derivative_cm: np.ndarray
    x_total_cm: np.ndarray
    resonances: tuple[GaussianResonanceEntry, ...]
    active_resonance_counts: dict[str, int]
    variational_overlaps: tuple[GaussianVariationalOverlap, ...]
    variational_energies: tuple[GaussianVariationalEnergy, ...]
    variational_state_definitions: tuple[GaussianVariationalStateDefinition, ...]
    fundamental_bands: tuple[GaussianFundamentalBand, ...]


def _last_block(text: str, header: str, n_lines: int | None = None) -> list[str]:
    idx = text.rfind(header)
    if idx < 0:
        raise ValueError(f"Could not find block header: {header}")
    block = text[idx:].splitlines()
    return block if n_lines is None else block[:n_lines]


def _find_last_standard_orientation(lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    indices = [i for i, line in enumerate(lines) if "Standard orientation:" in line]
    if not indices:
        raise ValueError("Could not find any 'Standard orientation' block.")
    start = indices[-1]
    rows: list[list[float]] = []
    atomic_numbers: list[int] = []
    i = start + 5
    while i < len(lines):
        line = lines[i].rstrip()
        if line.strip().startswith("-----"):
            break
        parts = line.split()
        if len(parts) >= 6:
            atomic_numbers.append(int(parts[1]))
            rows.append([float(parts[3]), float(parts[4]), float(parts[5])])
        i += 1
    return np.array(atomic_numbers, dtype=int), np.array(rows, dtype=float)


def _find_last_harmonic_modes(lines: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = [i for i, line in enumerate(lines) if line.strip().startswith("Harmonic frequencies (cm**-1)")]
    if not indices:
        raise ValueError("Could not find harmonic-frequency section.")
    start = indices[-1]
    frequencies_blocks: list[float] = []
    reduced_mass_blocks: list[float] = []
    force_constant_blocks: list[float] = []
    mode_blocks: list[np.ndarray] = []

    i = start
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Frequencies --"):
            if i + 3 >= len(lines):
                break
            freq_line = lines[i]
            red_mass_line = lines[i + 1]
            frc_line = lines[i + 2]
            atom_header = lines[i + 4].strip() if i + 4 < len(lines) else ""
            if not atom_header.startswith("Atom  AN"):
                i += 1
                continue

            frequencies_blocks.extend(float(tok) for tok in freq_line.split("--", 1)[1].split())
            reduced_mass_blocks.extend(float(tok) for tok in red_mass_line.split("--", 1)[1].split())
            force_constant_blocks.extend(float(tok) for tok in frc_line.split("--", 1)[1].split())

            mode_rows: list[list[float]] = []
            j = i + 5
            while j < len(lines):
                row = lines[j].rstrip()
                if not row.strip():
                    break
                parts = row.split()
                if len(parts) < 11:
                    break
                mode_rows.append([float(tok) for tok in parts[2:11]])
                j += 1

            arr = np.array(mode_rows, dtype=float)
            if arr.size == 0:
                break
            n_atoms = arr.shape[0]
            block_modes = np.zeros((3 * n_atoms, 3), dtype=float)
            for a in range(n_atoms):
                for k in range(3):
                    block_modes[3 * a : 3 * a + 3, k] = arr[a, 3 * k : 3 * k + 3]
            mode_blocks.append(block_modes)
            i = j
            continue

        if line.startswith("Fundamental Bands") or line.startswith("And Diagonal Anharmonicity"):
            break
        i += 1

    if not mode_blocks:
        raise ValueError("Could not find compact normal-mode table.")

    frequencies = np.array(frequencies_blocks, dtype=float)
    reduced_masses = np.array(reduced_mass_blocks, dtype=float)
    force_constants = np.array(force_constant_blocks, dtype=float)
    modes = np.concatenate(mode_blocks, axis=1)
    return frequencies, reduced_masses, force_constants, modes


def _find_thermochemistry_masses(lines: list[str], n_atoms: int) -> np.ndarray:
    masses: list[float] = []
    for i, line in enumerate(lines):
        if line.strip().startswith("- Thermochemistry -"):
            j = i
            while j < len(lines) and len(masses) < n_atoms:
                m = re.match(r"\s*Atom\s+\d+\s+has atomic number\s+\d+\s+and mass\s+([0-9.]+)", lines[j])
                if m:
                    masses.append(float(m.group(1)))
                j += 1
    if len(masses) < n_atoms:
        raise ValueError("Could not parse thermochemistry masses.")
    return np.array(masses[:n_atoms], dtype=float)


def _find_last_harmonic_mode_symmetry_labels(lines: list[str]) -> tuple[str, ...] | None:
    indices = [i for i, line in enumerate(lines) if line.strip().startswith("Harmonic frequencies (cm**-1)")]
    if not indices:
        return None
    start = indices[-1]
    labels: list[str] = []
    i = start
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Frequencies --") or line.startswith("Frequencies ---"):
            freq_tokens = re.findall(r"[-+]?\d+\.\d+(?:[DdEe][-+]?\d+)?", line)
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            prev_tokens = lines[j].split() if j >= 0 else []
            if prev_tokens and len(prev_tokens) >= len(freq_tokens):
                tail = prev_tokens[-len(freq_tokens):]
                if all(not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", tok) for tok in tail):
                    labels.extend(tail)
                else:
                    labels.extend(["?"] * len(freq_tokens))
            else:
                labels.extend(["?"] * len(freq_tokens))
        if line.startswith("Fundamental Bands") or line.startswith("And Diagonal Anharmonicity"):
            break
        i += 1
    return tuple(labels) if labels else None


def _find_principal_axes(lines: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = [i for i, line in enumerate(lines) if "Principal axes and moments of inertia in atomic units:" in line]
    if not indices:
        raise ValueError("Could not find principal-axes section.")
    start = indices[-1]
    eigvals_line = lines[start + 2]
    moments = np.array([float(tok) for tok in eigvals_line.split("--", 1)[1].split()], dtype=float)
    axis_rows = []
    labels = []
    for off in range(3, 6):
        parts = lines[start + off].split()
        labels.append(parts[0])
        axis_rows.append([float(tok) for tok in parts[1:4]])
    # Columns are principal axes in the current Cartesian frame.
    principal_axes = np.array(axis_rows, dtype=float)
    rot_line = next(line for line in lines[start:] if "Rotational constants (GHZ):" in line)
    rot_ghz = np.array([float(tok) for tok in rot_line.split(":")[1].split()], dtype=float)
    return moments, principal_axes, rot_ghz


def _find_mode_equivalency(lines: list[str], n_modes: int) -> np.ndarray | None:
    """Parse Gaussian's anharmonic mode equivalency table (H -> A), if present."""
    for i, line in enumerate(lines):
        if "connection between this new numbering (A) and the one used before" not in line:
            continue
        block = lines[i : i + 8]
        h_line = next((row for row in block if row.lstrip().startswith("(H) |")), None)
        a_line = next((row for row in block if row.lstrip().startswith("(A) |")), None)
        if h_line is None or a_line is None:
            continue
        h_vals = [int(tok) for tok in re.findall(r"\d+", h_line)]
        a_vals = [int(tok) for tok in re.findall(r"\d+", a_line)]
        if len(h_vals) != len(a_vals) or not h_vals:
            continue

        mapping = np.arange(n_modes, dtype=int)
        for h_mode, a_mode in zip(h_vals, a_vals):
            if 1 <= h_mode <= n_modes and 1 <= a_mode <= n_modes:
                mapping[a_mode - 1] = h_mode - 1
        return mapping
    return None


def _find_alpha_matrix(lines: list[str], unit_label: str) -> tuple[np.ndarray, np.ndarray, tuple[str, str, str]]:
    indices = [i for i, line in enumerate(lines) if f"Vibro-Rot alpha Matrix (in {unit_label})" in line]
    if not indices:
        raise ValueError(f"Could not find alpha matrix in {unit_label}.")
    start = indices[-1]
    header = lines[start + 2].split()
    if len(header) < 3:
        raise ValueError(f"Malformed alpha-matrix header in {unit_label}.")
    axis_labels = tuple(header[-3:])
    mode_indices: list[int] = []
    rows: list[list[float]] = []
    i = start + 3
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) >= 5 and parts[0] == "Q(" and parts[1].endswith(")"):
            mode_indices.append(int(parts[1][:-1]))
            rows.append([_to_float(tok) for tok in parts[-3:]])
            i += 1
            continue
        break
    if not rows:
        raise ValueError(f"Could not parse any alpha-matrix rows in {unit_label}.")
    return np.array(mode_indices, dtype=int), np.array(rows, dtype=float), axis_labels


def _find_spectroscopic_axes(lines: list[str]) -> dict[str, int] | None:
    axis_map = {"x": 0, "y": 1, "z": 2}
    pattern = re.compile(r"A\(([xyz])\)\s+B\(([xyz])\)\s+C\(([xyz])\)", re.IGNORECASE)
    for line in lines:
        m = pattern.search(line)
        if m:
            return {
                "a": axis_map[m.group(1).lower()],
                "b": axis_map[m.group(2).lower()],
                "c": axis_map[m.group(3).lower()],
            }
    return None


def _project_quartic_tauprime_to_s_constants(
    tau_prime_cm: dict[str, float],
    axes: dict[str, int],
    sigma: float,
) -> dict[str, float]:
    """Project a Gaussian Tau Prime block to S-reduced quartics for a trial axis map."""
    if "aaaa" in tau_prime_cm:
        vals = [[0.0] * 3 for _ in range(3)]
        vals[0][0] = tau_prime_cm["aaaa"]
        vals[0][1] = tau_prime_cm["aabb"]
        vals[0][2] = tau_prime_cm["aacc"]
        vals[1][1] = tau_prime_cm["bbbb"]
        vals[1][2] = tau_prime_cm["bbcc"]
        vals[2][2] = tau_prime_cm["cccc"]
    else:
        vals = [[0.0] * 3 for _ in range(3)]
        vals[0][0] = tau_prime_cm["xxxx"]
        vals[0][1] = tau_prime_cm["xxyy"]
        vals[0][2] = tau_prime_cm["xxzz"]
        vals[1][1] = tau_prime_cm["yyyy"]
        vals[1][2] = tau_prime_cm["yyzz"]
        vals[2][2] = tau_prime_cm["zzzz"]
    for i in range(3):
        for j in range(i):
            vals[i][j] = vals[j][i]

    order = [axes["a"], axes["b"], axes["c"]]
    taup = [[vals[order[i]][order[j]] for j in range(3)] for i in range(3)]
    t11, t22, t33 = taup[0][0] / 4.0, taup[1][1] / 4.0, taup[2][2] / 4.0
    t12, t13, t23 = taup[0][1] / 4.0, taup[0][2] / 4.0, taup[1][2] / 4.0
    t400 = (3.0 * t11 + 3.0 * t22 + 2.0 * t12) / 8.0
    t220 = t13 + t23 - 2.0 * t400
    t040 = t33 - t220 - t400
    t202 = (t11 - t22) / 4.0
    t022 = (t13 - t23) / 2.0 - t202
    t004 = (t11 + t22 - 2.0 * t12) / 16.0
    sigma1 = 1.0 / sigma
    return {
        "D J": -t400 + 0.5 * t022 * sigma1,
        "D JK": -t220 - 3.0 * t022 * sigma1,
        "D K": -t040 + 2.5 * t022 * sigma1,
        "d 1": t202,
        "d 2": t004 + 0.25 * t022 * sigma1,
    }


def _infer_spectroscopic_axes_from_quartic_data(
    tau_prime_cm: dict[str, float],
    s_reduction_mhz: dict[str, float],
    sigma: float,
    fallback: dict[str, int] | None = None,
) -> dict[str, int] | None:
    """Infer the axis permutation that makes Tau Prime reproduce Gaussian quartics."""
    if not tau_prime_cm or not s_reduction_mhz:
        return fallback
    target_cm = {k: float(v) / CMINV_TO_MHZ for k, v in s_reduction_mhz.items()}
    best_axes = fallback
    best_err = None
    for perm in permutations((0, 1, 2)):
        axes = {"a": perm[0], "b": perm[1], "c": perm[2]}
        proj = _project_quartic_tauprime_to_s_constants(tau_prime_cm, axes, sigma)
        err = sum(abs(proj[k] - target_cm[k]) for k in ("D J", "D JK", "D K", "d 1", "d 2"))
        if best_err is None or err < best_err:
            best_err = err
            best_axes = axes
    return best_axes


def _find_tau_prime(lines: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    indices = [i for i, line in enumerate(lines) if "Quartic Centrifugal Distortion Constants Tau Prime" in line]
    if not indices:
        raise ValueError("Could not find Tau Prime section.")
    start = indices[-1]
    tau_cm: dict[str, float] = {}
    tau_mhz: dict[str, float] = {}
    pattern = re.compile(
        r"^\s*TauP\s+([abc]{4})\s+([\-0-9Dd.+]+)\s+([\-0-9Dd.+]+)\s*$",
        re.IGNORECASE,
    )
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if not line.strip():
            if tau_cm:
                break
            continue
        match = pattern.match(line)
        if match:
            key = match.group(1).lower()
            tau_cm[key] = _to_float(match.group(2))
            tau_mhz[key] = _to_float(match.group(3))
            continue
        if tau_cm:
            break
    if not tau_cm:
        raise ValueError("Could not parse Tau Prime section.")
    return tau_cm, tau_mhz


def _find_asymmetry(lines: list[str]) -> tuple[float, float, float]:
    indices = [i for i, line in enumerate(lines) if line.strip() == "Asymmetric Top Reduction"]
    if not indices:
        raise ValueError("Could not find asymmetry section.")
    start = indices[-1]
    vals = {}
    for i in range(start + 3, start + 7):
        line = lines[i]
        m = re.match(r"\s*(Kappa|Delta|Sigma)\s*:\s*([\-0-9Dd.+]+)", line)
        if m:
            vals[m.group(1)] = _to_float(m.group(2))
    return vals["Kappa"], vals["Delta"], vals["Sigma"]


def _find_didq_block(lines: list[str]) -> np.ndarray | None:
    indices = [i for i, line in enumerate(lines) if "Inertia Moments Derivatives w.r.t. Normal Modes" in line]
    if not indices:
        return None
    start = indices[-1]

    rows: list[list[float]] = []
    for i in range(start + 4, len(lines)):
        line = lines[i].strip()
        if not line:
            if rows:
                break
            continue
        if not line.startswith("Q("):
            if rows:
                break
            continue
        parts = line.replace(")", " ").split()
        values = [_to_float(tok) for tok in parts[-6:]]
        rows.append(values)
    if not rows:
        return None
    return np.array(rows, dtype=float).T


def _find_sextic_cartesian(lines: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    indices = [i for i, line in enumerate(lines) if "Sextic Distortion Constants" in line]
    if not indices:
        raise ValueError("Could not find sextic Cartesian section.")
    start = indices[-1]
    phi_cm: dict[str, float] = {}
    phi_hz: dict[str, float] = {}
    pattern = re.compile(r"^\s*Phi\s+([abc]{3})\s+([\-0-9Dd.+]+)\s+([\-0-9Dd.+]+)\s*$", re.IGNORECASE)
    for i in range(start + 3, len(lines)):
        line = lines[i]
        if not line.strip():
            if phi_cm:
                break
            continue
        m = pattern.match(line)
        if m:
            key = m.group(1).lower()
            phi_cm[key] = _to_float(m.group(2))
            phi_hz[key] = _to_float(m.group(3))
            continue
        if phi_cm and line.strip().startswith("Asymmetric Top Reduction"):
            break
    if not phi_cm:
        raise ValueError("Could not parse sextic Cartesian constants.")
    return phi_cm, phi_hz


def _find_sextic_reduction(
    lines: list[str],
    header: str,
    pattern: str,
) -> tuple[dict[str, float], dict[str, float]]:
    indices = [i for i, line in enumerate(lines) if header in line]
    if not indices:
        raise ValueError(f"Could not find sextic section: {header}")
    start = indices[-1]
    rx = re.compile(pattern, re.IGNORECASE)
    vals_cm: dict[str, float] = {}
    vals_hz: dict[str, float] = {}
    for i in range(start + 3, len(lines)):
        line = lines[i]
        if not line.strip():
            if vals_cm:
                break
            continue
        m = rx.match(line)
        if m:
            key = re.sub(r"\s+", " ", m.group(1).strip())
            vals_cm[key] = _to_float(m.group(2))
            vals_hz[key] = _to_float(m.group(3))
            continue
        if vals_cm and (line.strip().startswith("rho") or line.strip().startswith("Constants in the")):
            break
    if not vals_cm:
        raise ValueError(f"Could not parse sextic section: {header}")
    return vals_cm, vals_hz


def _find_sextic_auxiliary(lines: list[str]) -> tuple[float | None, float | None, float | None, float | None]:
    indices = [i for i, line in enumerate(lines) if "Sextic Centrifugal Distortion Constants" in line]
    if not indices:
        return None, None, None, None
    start = indices[-1]
    vals: dict[str, float] = {}
    for i in range(start, len(lines)):
        line = lines[i]
        stripped = line.strip()
        for key in ("rho", "mu", "nu", "lambda"):
            if stripped.startswith(f"{key}"):
                parts = stripped.replace(":", " ").split()
                if len(parts) >= 2:
                    vals[key] = _to_float(parts[1])
    return vals.get("rho"), vals.get("mu"), vals.get("nu"), vals.get("lambda")


def _find_last_section_index(lines: list[str], header: str) -> int:
    indices = [i for i, line in enumerate(lines) if header in line]
    if not indices:
        raise ValueError(f"Could not find section: {header}")
    return indices[-1]


def _find_linear_ltype_constants(lines: list[str]) -> GaussianLinearLTypeConstants:
    q_e_cm: dict[int, float] = {}
    q_j_cm: dict[int, float] = {}
    q_k_cm: dict[int, float] = {}
    current: str | None = None
    pattern = re.compile(r"^\s*Q\(\s*(\d+)\)\s+([\-0-9Dd.+]+)")
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("q^e constants"):
            current = "q_e"
            continue
        if stripped.startswith("q^J constants"):
            current = "q_j"
            continue
        if stripped.startswith("q^K constants"):
            current = "q_k"
            continue
        match = pattern.match(line)
        if current and match:
            idx = int(match.group(1))
            value = _to_float(match.group(2))
            if current == "q_e":
                q_e_cm[idx] = value
            elif current == "q_j":
                q_j_cm[idx] = value
            else:
                q_k_cm[idx] = value
            continue
        if current and stripped.startswith("===="):
            current = None
    if not (q_e_cm or q_j_cm or q_k_cm):
        raise ValueError("Could not parse linear l-type constants.")
    active_dd_22 = None
    for line in lines:
        m = re.search(r"(\d+)\s+active resonances out of\s+(\d+)", line)
        if m:
            active_dd_22 = int(m.group(1))
    return GaussianLinearLTypeConstants(
        q_e_cm=q_e_cm,
        q_e_mhz={key: val * CMINV_TO_MHZ for key, val in q_e_cm.items()},
        q_j_cm=q_j_cm,
        q_j_mhz={key: val * CMINV_TO_MHZ for key, val in q_j_cm.items()},
        q_k_cm=q_k_cm,
        q_k_mhz={key: val * CMINV_TO_MHZ for key, val in q_k_cm.items()},
        active_dd_22_count=active_dd_22,
    )


def _find_linear_rotdist_constants(lines: list[str]) -> GaussianLinearRotDistConstants:
    d_mhz = None
    h_mhz = None
    indices = [i for i, line in enumerate(lines) if "Pickett input using symmetry:" in line]
    if not indices:
        return GaussianLinearRotDistConstants(d_mhz=None, h_mhz=None)
    start = indices[-1]
    for line in lines[start:]:
        stripped = line.strip()
        m_d = re.match(r"^D\s*=\s*([\-0-9Dd.+]+)\s*$", stripped)
        if m_d:
            d_mhz = _to_float(m_d.group(1))
            continue
        m_h = re.match(r"^H\s*=\s*([\-0-9Dd.+]+)\s*$", stripped)
        if m_h:
            h_mhz = _to_float(m_h.group(1))
            continue
        if d_mhz is not None and h_mhz is not None:
            break
    return GaussianLinearRotDistConstants(d_mhz=d_mhz, h_mhz=h_mhz)


def _find_tau_tensor(lines: list[str], start: int) -> np.ndarray:
    tau = np.zeros((3, 3, 3, 3), dtype=float)
    i = start
    block_header = re.compile(r"\s*IXYZ\s*=\s*(\d+)\s+JXYZ\s*=\s*(\d+)")
    row_pattern = re.compile(r"^\s*(\d+)\s+([\-0-9Dd.+]+)\s+([\-0-9Dd.+]+)\s+([\-0-9Dd.+]+)\s*$")
    while i < len(lines):
        m = block_header.match(lines[i])
        if m:
            jx = int(m.group(1)) - 1
            ix = int(m.group(2)) - 1
            i += 2
            for _ in range(3):
                row = row_pattern.match(lines[i])
                if row is None:
                    raise ValueError("Malformed Tau block in Gaussian log.")
                kx = int(row.group(1)) - 1
                values = [_to_float(row.group(col)) for col in (2, 3, 4)]
                for lx, value in enumerate(values):
                    tau[kx, lx, jx, ix] = value
                i += 1
            continue
        if "Tau Prime" in lines[i] and np.any(tau):
            break
        i += 1
    if not np.any(tau):
        raise ValueError("Could not parse Tau tensor block.")
    return tau


def _find_lower_triangular_matrix_after_header(lines: list[str], header: str) -> np.ndarray:
    start = _find_last_section_index(lines, header)
    row_map: dict[int, list[float]] = {}
    for i in range(start + 3, len(lines)):
        line = lines[i]
        if not line.strip():
            continue
        parts = line.split()
        if parts and all(tok.isdigit() for tok in parts):
            continue
        if len(parts) >= 2 and parts[0].isdigit():
            row_idx = int(parts[0])
            row_map.setdefault(row_idx, []).extend(_to_float(tok) for tok in parts[1:])
            continue
        if row_map:
            break
    if not row_map:
        raise ValueError(f"Could not parse lower-triangular matrix after header: {header}")
    n = max(row_map)
    out = np.zeros((n, n), dtype=float)
    for row_idx in range(1, n + 1):
        vals = row_map.get(row_idx, [])
        if len(vals) != row_idx:
            raise ValueError(f"Malformed lower-triangular matrix row {row_idx} after header: {header}")
        out[row_idx - 1, :row_idx] = vals
        out[:row_idx, row_idx - 1] = vals
    return out


def _find_resonance_entries(lines: list[str]) -> tuple[tuple[GaussianResonanceEntry, ...], dict[str, int]]:
    entries: list[GaussianResonanceEntry] = []
    active_counts = {"fermi_12": 0, "darling_22": 0, "darling_11": 0}
    sections = (
        ("fermi_12", "1-2 Fermi resonances"),
        ("darling_22", "2-2 Darling-Dennison resonances"),
        ("darling_11", "1-1 Darling-Dennison resonances"),
    )
    for kind, header in sections:
        start = _find_last_section_index(lines, header)
        found_table = False
        for i in range(start + 1, len(lines)):
            stripped = lines[i].strip()
            if not stripped:
                continue
            if stripped.startswith("No resonances found."):
                break
            if "active resonances out of" in stripped:
                m = re.search(r"(\d+)\s+active resonances out of\s+(\d+)", stripped)
                if m:
                    active_counts[kind] = int(m.group(1))
                break
            if stripped.startswith("I"):
                found_table = True
                continue
            if not found_table:
                continue
            if kind == "fermi_12":
                m_entry = re.match(
                    r"^\s*(\d+)\s+\|\s+(\d+)\s+(\d+)\s+\|\s+([\-0-9Dd.+]+)\s+\|\s+([\-0-9Dd.+]+)\s+\|\s+([\-0-9Dd.+]+)\s+\|\s+(\w+)\s*$",
                    stripped,
                )
                if not m_entry:
                    if entries:
                        continue
                    continue
                entries.append(
                    GaussianResonanceEntry(
                        kind=kind,
                        lhs_modes=(int(m_entry.group(1)),),
                        rhs_modes=(int(m_entry.group(2)), int(m_entry.group(3))),
                        freq_diff_cm=_to_float(m_entry.group(4)),
                        metric_1=_to_float(m_entry.group(5)),
                        metric_2=_to_float(m_entry.group(6)),
                        status=m_entry.group(7),
                    )
                )
                continue
            if kind == "darling_22":
                m_entry = re.match(
                    r"^\s*(\d+)\s+(\d+)\s+\|\s+(\d+)\s+(\d+)\s+\|\s+([\-0-9Dd.+]+)\s+\|\s+([\-0-9Dd.+]+)\s+\|\s+(\w+)\s*$",
                    stripped,
                )
                if not m_entry:
                    continue
                entries.append(
                    GaussianResonanceEntry(
                        kind=kind,
                        lhs_modes=(int(m_entry.group(1)), int(m_entry.group(2))),
                        rhs_modes=(int(m_entry.group(3)), int(m_entry.group(4))),
                        freq_diff_cm=_to_float(m_entry.group(5)),
                        metric_1=_to_float(m_entry.group(6)),
                        metric_2=None,
                        status=m_entry.group(7),
                    )
                )
                continue
            if kind == "darling_11":
                m_entry = re.match(
                    r"^\s*(\d+)\s+(\d+)\s+\|\s+(\d+)\s+(\d+)\s+\|\s+([\-0-9Dd.+]+)\s+\|\s+([\-0-9Dd.+]+)\s+\|\s+([\-0-9Dd.+]+)\s+\|\s+(\w+)\s*$",
                    stripped,
                )
                if not m_entry:
                    continue
                entries.append(
                    GaussianResonanceEntry(
                        kind=kind,
                        lhs_modes=(int(m_entry.group(1)), int(m_entry.group(2))),
                        rhs_modes=(int(m_entry.group(3)), int(m_entry.group(4))),
                        freq_diff_cm=_to_float(m_entry.group(5)),
                        metric_1=_to_float(m_entry.group(6)),
                        metric_2=_to_float(m_entry.group(7)),
                        status=m_entry.group(8),
                    )
                )
                continue
    return tuple(entries), active_counts


def _find_variational_overlaps(lines: list[str]) -> tuple[GaussianVariationalOverlap, ...]:
    try:
        start = _find_last_section_index(lines, "Projection of DVPT2 states on New Variational States")
    except ValueError:
        return ()
    out: list[GaussianVariationalOverlap] = []
    pattern = re.compile(r"State\s+(.+?)\s+has overlap of\s+([0-9.]+)%\s+with state\s+(\d+)")
    for i in range(start + 1, len(lines)):
        stripped = lines[i].strip()
        m = pattern.search(stripped)
        if m:
            out.append(
                GaussianVariationalOverlap(
                    dvpt2_state=m.group(1).strip(),
                    overlap=float(m.group(2)) / 100.0,
                    variational_state_index=int(m.group(3)),
                )
            )
            continue
        if out and stripped.startswith("Vibrational Energies"):
            break
    return tuple(out)


def _find_variational_energies(lines: list[str]) -> tuple[GaussianVariationalEnergy, ...]:
    try:
        start = _find_last_section_index(lines, "Vibrational Energies (cm^-1)")
    except ValueError:
        return ()
    out: list[GaussianVariationalEnergy] = []
    pattern = re.compile(r"^\s*(\d+\(\d+\))?(?:\s+(\d+\(\d+\)))?\s+([\-0-9Dd.+]+)\s+([\-0-9Dd.+]+)\s*$")
    for i in range(start + 3, len(lines)):
        stripped = lines[i].rstrip()
        if not stripped:
            if out:
                break
            continue
        m = pattern.match(stripped)
        if not m:
            if out:
                break
            continue
        labels = [grp for grp in (m.group(1), m.group(2)) if grp]
        out.append(
            GaussianVariationalEnergy(
                dvpt2_state=";".join(labels),
                deperturbed_energy_cm=_to_float(m.group(3)),
                after_diag_energy_cm=_to_float(m.group(4)),
            )
        )
    return tuple(out)


def _find_variational_state_definitions(lines: list[str]) -> tuple[GaussianVariationalStateDefinition, ...]:
    try:
        start = _find_last_section_index(lines, "Definition of New States w.r.t. Deperturbed States")
    except ValueError:
        return ()
    out: list[GaussianVariationalStateDefinition] = []
    current_idx: int | None = None
    pattern = re.compile(r"^\s*(\d+)\s*:\s*([+\-]?[0-9.]+)\s+x\s+(.+?)\s*$")
    cont_pattern = re.compile(r"^\s*([+\-]?[0-9.]+)\s+x\s+(.+?)\s*$")
    for i in range(start + 1, len(lines)):
        stripped = lines[i].rstrip()
        if not stripped:
            if out:
                break
            continue
        m = pattern.match(stripped)
        if m:
            current_idx = int(m.group(1))
            out.append(
                GaussianVariationalStateDefinition(
                    variational_state_index=current_idx,
                    coefficient=float(m.group(2)),
                    dvpt2_state=m.group(3).strip(),
                )
            )
            continue
        m2 = cont_pattern.match(stripped)
        if m2 and current_idx is not None:
            out.append(
                GaussianVariationalStateDefinition(
                    variational_state_index=current_idx,
                    coefficient=float(m2.group(1)),
                    dvpt2_state=m2.group(2).strip(),
                )
            )
            continue
        if out:
            break
    return tuple(out)


def _find_final_fundamental_bands(lines: list[str]) -> tuple[GaussianFundamentalBand, ...]:
    try:
        start = _find_last_section_index(lines, "Vibrational Energies at Anharmonic Level")
    except ValueError:
        return ()
    band_start = None
    for i in range(start, len(lines)):
        if lines[i].strip() == "Fundamental Bands":
            band_start = i
            break
    if band_start is None:
        return ()
    out: list[GaussianFundamentalBand] = []
    pattern = re.compile(
        r"^\s*(?:(H|L)\s+)?(\d+)\(1\)\s+(\w+)\s+([\-0-9Dd.+]+)\s+([\-0-9Dd.+]+)\s+"
    )
    for i in range(band_start + 3, len(lines)):
        stripped = lines[i].rstrip()
        if not stripped:
            if out:
                break
            continue
        m = pattern.match(stripped)
        if not m:
            if out:
                break
            continue
        out.append(
            GaussianFundamentalBand(
                mode_index=int(m.group(2)),
                status=m.group(3),
                harmonic_cm=_to_float(m.group(4)),
                anharmonic_cm=_to_float(m.group(5)),
                overlap_flag=m.group(1),
            )
        )
    return tuple(out)


def _find_c1_matrix(lines: list[str], start: int) -> np.ndarray:
    row_pattern = re.compile(
        r"^\s*(\d+)\s+([XYZ])\s+([XYZ])\s+([\-0-9Dd.+]+)\s*$",
        re.IGNORECASE,
    )
    axis = {"X": 0, "Y": 1, "Z": 2}
    entries: list[tuple[int, int, int, float]] = []
    max_mode = 0
    for i in range(start, len(lines)):
        line = lines[i]
        m = row_pattern.match(line)
        if m:
            mode = int(m.group(1)) - 1
            ix = axis[m.group(2).upper()]
            jx = axis[m.group(3).upper()]
            entries.append((mode, ix, jx, _to_float(m.group(4))))
            max_mode = max(max_mode, mode + 1)
            continue
        if entries and "Dimensionless C_i^abc Matrix" in line:
            break
    if not entries:
        raise ValueError("Could not parse C_i^ab matrix.")
    c1 = np.zeros((max_mode, 3, 3), dtype=float)
    for mode, ix, jx, value in entries:
        c1[mode, ix, jx] = value
    return c1


def _find_c2_tensor(lines: list[str], start: int) -> np.ndarray:
    row_pattern = re.compile(
        r"^\s*(\d+)\s+([XYZ])\s+([XYZ])\s+([XYZ])\s+([\-0-9Dd.+]+)\s*$",
        re.IGNORECASE,
    )
    axis = {"X": 0, "Y": 1, "Z": 2}
    entries: list[tuple[int, int, int, int, float]] = []
    max_mode = 0
    for i in range(start, len(lines)):
        line = lines[i]
        m = row_pattern.match(line)
        if m:
            mode = int(m.group(1)) - 1
            ix = axis[m.group(2).upper()]
            jx = axis[m.group(3).upper()]
            kx = axis[m.group(4).upper()]
            entries.append((mode, ix, jx, kx, _to_float(m.group(5))))
            max_mode = max(max_mode, mode + 1)
            continue
        if entries and "Quartic Centrifugal Distortion Constants Tau Prime" in line:
            break
    if not entries:
        raise ValueError("Could not parse C_i^abc tensor.")
    c2 = np.zeros((max_mode, 3, 3, 3), dtype=float)
    for mode, ix, jx, kx, value in entries:
        c2[mode, ix, jx, kx] = value
    return c2


def _find_quadratic_force_constants(lines: list[str]) -> np.ndarray:
    indices = [i for i, line in enumerate(lines) if "QUADRATIC FORCE CONSTANTS IN NORMAL MODES" in line]
    if not indices:
        raise ValueError("Could not find quadratic force-constant section.")
    start = indices[-1]
    freq_map: dict[int, float] = {}
    pattern = re.compile(r"^\s*(\d+)\s+(\d+)\s+([\-0-9Dd.+]+)")
    for i in range(start + 8, len(lines)):
        line = lines[i]
        if not line.strip():
            if freq_map:
                break
            continue
        m = pattern.match(line)
        if m:
            i_idx = int(m.group(1))
            j_idx = int(m.group(2))
            if i_idx == j_idx:
                freq_map[i_idx] = _to_float(m.group(3))
            continue
        if freq_map and "CUBIC FORCE CONSTANTS IN NORMAL MODES" in line:
            break
    if not freq_map:
        raise ValueError("Could not parse anharmonic-order frequencies from quadratic force constants.")
    n_modes = max(freq_map)
    return np.array([freq_map[i] for i in range(1, n_modes + 1)], dtype=float)


def _find_cubic_force_constants(lines: list[str], n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    indices = [i for i, line in enumerate(lines) if "CUBIC FORCE CONSTANTS IN NORMAL MODES" in line]
    if not indices:
        raise ValueError("Could not find cubic force-constant section.")
    start = indices[-1]
    phi3_reduced = np.zeros((n_modes, n_modes, n_modes), dtype=float)
    phi3_raw = np.zeros((n_modes, n_modes, n_modes), dtype=float)
    pattern = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([\-0-9Dd.+]+)\s+([\-0-9Dd.+]+)\s+([\-0-9Dd.+]+)\s*$"
    )
    for i in range(start + 8, len(lines)):
        line = lines[i]
        if not line.strip():
            if np.any(phi3_reduced):
                break
            continue
        m = pattern.match(line)
        if m:
            idx = sorted((int(m.group(1)) - 1, int(m.group(2)) - 1, int(m.group(3)) - 1))
            value_reduced = _to_float(m.group(4))
            value_raw = _to_float(m.group(6))
            for p in {
                (idx[0], idx[1], idx[2]),
                (idx[0], idx[2], idx[1]),
                (idx[1], idx[0], idx[2]),
                (idx[1], idx[2], idx[0]),
                (idx[2], idx[0], idx[1]),
                (idx[2], idx[1], idx[0]),
            }:
                phi3_reduced[p] = value_reduced
                phi3_raw[p] = value_raw
            continue
        if np.any(phi3_reduced) and "QUARTIC FORCE CONSTANTS IN NORMAL MODES" in line:
            break
    if not np.any(phi3_reduced):
        raise ValueError("Could not parse cubic force constants.")
    return phi3_reduced, phi3_raw


def _find_quartic_force_constants(lines: list[str], n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    indices = [i for i, line in enumerate(lines) if "QUARTIC FORCE CONSTANTS IN NORMAL MODES" in line]
    if not indices:
        raise ValueError("Could not find quartic force-constant section.")
    start = indices[-1]
    phi4_reduced = np.zeros((n_modes, n_modes, n_modes, n_modes), dtype=float)
    phi4_raw = np.zeros((n_modes, n_modes, n_modes, n_modes), dtype=float)
    pattern = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\-0-9Dd.+]+)\s+([\-0-9Dd.+]+)\s+([\-0-9Dd.+]+)\s*$"
    )
    for i in range(start + 8, len(lines)):
        line = lines[i]
        if not line.strip():
            if np.any(phi4_reduced):
                break
            continue
        m = pattern.match(line)
        if m:
            idx = sorted((int(m.group(1)) - 1, int(m.group(2)) - 1, int(m.group(3)) - 1, int(m.group(4)) - 1))
            value_reduced = _to_float(m.group(5))
            value_raw = _to_float(m.group(7))
            for p in set(permutations(idx)):
                phi4_reduced[p] = value_reduced
                phi4_raw[p] = value_raw
            continue
        if np.any(phi4_reduced) and "Num. of 4th derivatives" in line:
            break
    if not np.any(phi4_reduced):
        raise ValueError("Could not parse quartic force constants.")
    return phi4_reduced, phi4_raw


def _parse_fchk_array(lines: list[str], label: str, kind: str) -> np.ndarray:
    for i, line in enumerate(lines):
        if line.startswith(label) and f" {kind}   N=" in line:
            n_vals = int(line.split("=")[-1])
            values: list[float] = []
            j = i + 1
            while len(values) < n_vals and j < len(lines):
                values.extend(float(tok.replace("D", "E")) for tok in lines[j].split())
                j += 1
            if len(values) != n_vals:
                raise ValueError(f"Could not parse full fchk array for {label}.")
            return np.array(values, dtype=float)
    raise ValueError(f"Could not find fchk array: {label}")


def _parse_fchk_scalar(lines: list[str], label: str, kind: str) -> int | float:
    for line in lines:
        if line.startswith(label) and f" {kind}" in line:
            return int(line.split()[-1]) if kind == "I" else float(line.split()[-1].replace("D", "E"))
    raise ValueError(f"Could not find fchk scalar: {label}")


def _parse_fchk_char_value(lines: list[str], label: str) -> str | None:
    for i, line in enumerate(lines):
        if line.startswith(label):
            parts = line.split()
            if "N=" in parts and i + 1 < len(lines):
                value = lines[i + 1].strip()
                return value or None
            tail = line[len(label) :].strip()
            return tail or None
    return None


def _normalize_point_group_label(label: str | None) -> str | None:
    if label is None:
        return None
    text = label.strip()
    if not text:
        return None
    up = text.upper()
    if up == "C*V":
        return "Cinfv"
    if up == "D*H":
        return "Dinfh"
    if up == "CINFV":
        return "Cinfv"
    if up == "DINFH":
        return "Dinfh"
    m = re.match(r"^([CDSOTI])(\d+)([A-Z].*)?$", text, re.IGNORECASE)
    if not m:
        return text
    head = m.group(1).upper()
    order = m.group(2)
    tail = (m.group(3) or "")
    tail = tail.replace("INF", "inf")
    if tail:
        tail = tail[0].lower() + tail[1:]
    return f"{head}{order}{tail}"


def parse_gaussian_harmonic_data(path: str | Path) -> GaussianHarmonicData:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    atomic_numbers, coords = _find_last_standard_orientation(lines)
    masses = _find_thermochemistry_masses(lines, len(atomic_numbers))
    moments, principal_axes, rot_ghz = _find_principal_axes(lines)
    freqs, red_masses, force_constants, modes = _find_last_harmonic_modes(lines)
    mode_symmetry_labels = _find_last_harmonic_mode_symmetry_labels(lines)
    mapping = _find_mode_equivalency(lines, freqs.size)
    return GaussianHarmonicData(
        atomic_numbers=atomic_numbers,
        masses_amu=masses,
        coords_std_ang=coords,
        principal_axes=principal_axes,
        moments_au=moments,
        rot_ghz=rot_ghz,
        frequencies_cm=freqs,
        reduced_masses_amu=red_masses,
        force_constants_mdyne_a=force_constants,
        normal_modes=modes,
        mode_symmetry_labels=mode_symmetry_labels,
        harmonic_to_anharmonic=mapping,
    )


def parse_gaussian_alpha_data(path: str | Path) -> GaussianAlphaData:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    alpha_idx_cm, alpha_cm, axis_labels_cm = _find_alpha_matrix(lines, "cm^-1")
    alpha_idx_mhz, alpha_mhz, axis_labels_mhz = _find_alpha_matrix(lines, "MHz")
    if alpha_idx_cm.shape != alpha_idx_mhz.shape or not np.array_equal(alpha_idx_cm, alpha_idx_mhz):
        raise ValueError("Alpha matrices in cm^-1 and MHz use inconsistent mode indexing.")
    if axis_labels_cm != axis_labels_mhz:
        raise ValueError("Alpha matrices in cm^-1 and MHz use inconsistent axis labels.")
    return GaussianAlphaData(
        mode_indices=alpha_idx_cm,
        alpha_cm=alpha_cm,
        alpha_mhz=alpha_mhz,
        axis_labels=axis_labels_cm,
    )


def parse_gaussian_quartic_benchmark(path: str | Path) -> GaussianQuarticBenchmark:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    alpha = parse_gaussian_alpha_data(path)
    tau_cm, tau_mhz = _find_tau_prime(lines)
    didq = _find_didq_block(lines)
    axes_guess = _find_spectroscopic_axes(lines)
    linear_rotdist = parse_gaussian_linear_rotdist_constants(path)

    is_linear = (
        linear_rotdist.d_mhz is not None
        and list(tau_cm.keys()) == ["cccc"]
    )
    if is_linear:
        kappa = 0.0
        delta = 0.0
        sigma = 0.0
    else:
        kappa, delta, sigma = _find_asymmetry(lines)

    a_mhz = {}
    a_pairs = [("DELTA", "J"), ("DELTA", "K"), ("DELTA", "JK"), ("delta", "J"), ("delta", "K")]
    for i, line in enumerate(lines):
        if "Constants in the Asymmetrically reduced Hamiltonian" in line:
            j = i + 3
            idx = 0
            while j < len(lines):
                parts = lines[j].split()
                if len(parts) >= 4 and idx < len(a_pairs) and tuple(parts[:2]) == a_pairs[idx]:
                    a_mhz[f"{parts[0]} {parts[1]}"] = _to_float(parts[-1])
                    idx += 1
                if idx == len(a_pairs):
                    break
                j += 1
            break

    s_mhz = {}
    s_pairs = [("D", "J"), ("D", "JK"), ("D", "K"), ("d", "1"), ("d", "2")]
    for i, line in enumerate(lines):
        if "Constants in the Symmetrically Reduced Hamiltonian" in line:
            j = i + 3
            idx = 0
            while j < len(lines):
                parts = lines[j].split()
                if len(parts) >= 4 and idx < len(s_pairs) and tuple(parts[:2]) == s_pairs[idx]:
                    s_mhz[f"{parts[0]} {parts[1]}"] = _to_float(parts[-1])
                    idx += 1
                if idx == len(s_pairs):
                    break
                j += 1
            break

    if is_linear:
        axes = axes_guess if axes_guess is not None else {"a": 0, "b": 1, "c": 2}
    else:
        axes = _infer_spectroscopic_axes_from_quartic_data(tau_cm, s_mhz, sigma, fallback=axes_guess)

    return GaussianQuarticBenchmark(
        alpha_mode_indices=alpha.mode_indices,
        alpha_cm=alpha.alpha_cm,
        alpha_mhz=alpha.alpha_mhz,
        alpha_axis_labels=alpha.axis_labels,
        tau_prime_cm=tau_cm,
        tau_prime_mhz=tau_mhz,
        a_reduction_mhz=a_mhz,
        s_reduction_mhz=s_mhz,
        sigma=sigma,
        kappa=kappa,
        delta=delta,
        didq_amu_sqrt_ang=didq,
        spectroscopic_axes=axes,
    )


def parse_gaussian_sextic_benchmark(path: str | Path) -> GaussianSexticBenchmark:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    phi_cart_cm, phi_cart_hz = _find_sextic_cartesian(lines)
    harmonic = parse_gaussian_harmonic_data(path)
    sextic_start = _find_last_section_index(lines, "Dump from SEXTIC")
    tau_start = _find_last_section_index(lines, "Quartic Centrifugal Distortion Constants Tau (in cm^-1)")
    c1_start = _find_last_section_index(lines, "Dimensionless C_i^ab Matrix")
    c2_start = _find_last_section_index(lines, "Dimensionless C_i^abc Matrix")
    if tau_start < sextic_start:
        raise ValueError("Could not locate sextic Tau block after 'Dump from SEXTIC'.")
    tau_cm = _find_tau_tensor(lines, tau_start)
    c1 = _find_c1_matrix(lines, c1_start)
    c2 = _find_c2_tensor(lines, c2_start)
    # L717/Sextic computes i0 = LsPrNM(i) when printing C_i^abc, but the
    # current write statement still uses storage index i. The raw log block is
    # therefore in storage order rather than printed-mode order.
    c2_reordered_to_print = None
    if harmonic.harmonic_to_anharmonic is not None:
        c2_reordered_to_print = c2[np.asarray(harmonic.harmonic_to_anharmonic, dtype=int)].copy()
    a_cm, a_hz = _find_sextic_reduction(
        lines,
        "Constants in the A reduced Hamiltonian",
        r"^\s*((?:Phi|phi)\s+[JjKk]+)\s*:\s*([\-0-9Dd.+]+)\s+([\-0-9Dd.+]+)\s*$",
    )
    s_cm, s_hz = _find_sextic_reduction(
        lines,
        "Constants in the S reduced Hamiltonian",
        r"^\s*((?:H\s+[JJKK]+|h\s+\d+))\s*:\s*([\-0-9Dd.+]+)\s+([\-0-9Dd.+]+)\s*$",
    )
    rho, mu, nu, lam = _find_sextic_auxiliary(lines)
    return GaussianSexticBenchmark(
        phi_cart_cm=phi_cart_cm,
        phi_cart_hz=phi_cart_hz,
        a_reduction_cm=a_cm,
        a_reduction_hz=a_hz,
        s_reduction_cm=s_cm,
        s_reduction_hz=s_hz,
        tau_cm=tau_cm,
        c1=c1,
        c2=c2,
        c2_reordered_to_print=c2_reordered_to_print,
        rho=rho,
        mu=mu,
        nu=nu,
        lam=lam,
    )


def parse_gaussian_linear_ltype_constants(path: str | Path) -> GaussianLinearLTypeConstants:
    lines = Path(path).read_text().splitlines()
    return _find_linear_ltype_constants(lines)


def parse_gaussian_linear_rotdist_constants(path: str | Path) -> GaussianLinearRotDistConstants:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return _find_linear_rotdist_constants(lines)


def parse_gaussian_anharmonic_force_data(path: str | Path) -> GaussianAnharmonicForceData:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    frequencies_cm = _find_quadratic_force_constants(lines)
    phi3_reduced_cm, phi3_raw_au = _find_cubic_force_constants(lines, frequencies_cm.size)
    phi4_reduced_cm, phi4_raw_au = _find_quartic_force_constants(lines, frequencies_cm.size)
    return GaussianAnharmonicForceData(
        frequencies_cm=frequencies_cm,
        phi3_reduced_cm=phi3_reduced_cm,
        phi3_raw_au=phi3_raw_au,
        phi4_reduced_cm=phi4_reduced_cm,
        phi4_raw_au=phi4_raw_au,
    )


def parse_gaussian_anharmonic_analysis(path: str | Path) -> GaussianAnharmonicAnalysis:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    entries, active_counts = _find_resonance_entries(lines)
    x_cor = _find_lower_triangular_matrix_after_header(lines, "Coriolis contributions to X Matrix (in cm^-1)")
    x_3rd = _find_lower_triangular_matrix_after_header(lines, "3rd Deriv. contributions to X Matrix (in cm^-1)")
    x_4th = _find_lower_triangular_matrix_after_header(lines, "4th Deriv. contributions to X Matrix (in cm^-1)")
    x_total = _find_lower_triangular_matrix_after_header(lines, "Total Anharmonic X Matrix (in cm^-1)")
    pt2_model = "unknown"
    for line in lines:
        m = re.search(r"PT2 model:\s*(.+?)\s*$", line)
        if m:
            pt2_model = m.group(1).strip()
    return GaussianAnharmonicAnalysis(
        pt2_model=pt2_model,
        x_coriolis_cm=x_cor,
        x_third_derivative_cm=x_3rd,
        x_fourth_derivative_cm=x_4th,
        x_total_cm=x_total,
        resonances=entries,
        active_resonance_counts=active_counts,
        variational_overlaps=_find_variational_overlaps(lines),
        variational_energies=_find_variational_energies(lines),
        variational_state_definitions=_find_variational_state_definitions(lines),
        fundamental_bands=_find_final_fundamental_bands(lines),
    )


def frequency_reorder_map(source_freq_cm: np.ndarray, target_freq_cm: np.ndarray, tol_cm: float = 5.0) -> np.ndarray:
    """Map source-mode ordering to target ordering by closest harmonic frequencies."""
    source = np.asarray(source_freq_cm, dtype=float)
    target = np.asarray(target_freq_cm, dtype=float)
    if source.shape != target.shape:
        raise ValueError("Source and target frequency lists must have the same shape.")

    mapping = np.full(source.shape[0], -1, dtype=int)
    used: set[int] = set()
    for i, nu in enumerate(source):
        diffs = np.abs(target - nu)
        order = np.argsort(diffs)
        pick = next((int(j) for j in order if int(j) not in used), None)
        if pick is None or diffs[pick] > tol_cm:
            raise ValueError(
                f"Could not match source frequency {nu:.6f} cm^-1 within {tol_cm:.3f} cm^-1."
            )
        mapping[i] = pick
        used.add(pick)
    return mapping


def reorder_cubic_force_constants(phi3_cm: np.ndarray, source_to_target: np.ndarray) -> np.ndarray:
    """Permute cubic force constants from source mode ordering to target ordering."""
    mapping = np.asarray(source_to_target, dtype=int)
    n_modes = mapping.size
    out = np.zeros_like(phi3_cm)
    for i in range(n_modes):
        for j in range(n_modes):
            for k in range(n_modes):
                out[mapping[i], mapping[j], mapping[k]] = phi3_cm[i, j, k]
    return out


def align_gaussian_cubic_force_constants(
    anh: GaussianAnharmonicForceData,
    target_freq_cm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align Gaussian cubic force constants to a target mode ordering by frequency.

    Important: Gaussian's anharmonic cubic block is already stored in the phase
    convention used internally by the sextic machinery. When these constants are
    imported into our reordered harmonic backend, only the *mode ordering* should
    be changed; the normal-mode sign flips used to align ``dIdQ``/``c1``/``c2``
    must not be applied again to the cubic tensor.
    """
    mapping = frequency_reorder_map(anh.frequencies_cm, np.asarray(target_freq_cm, dtype=float))
    phi3_reduced = reorder_cubic_force_constants(anh.phi3_reduced_cm, mapping)
    phi3_raw = reorder_cubic_force_constants(anh.phi3_raw_au, mapping)
    return mapping, phi3_reduced, phi3_raw


def apply_mode_signs_to_cubic_force_constants(phi3: np.ndarray, mode_signs: np.ndarray) -> np.ndarray:
    """Apply normal-mode sign flips to cubic force constants."""
    signs = np.asarray(mode_signs, dtype=float)
    if phi3.shape[0] != signs.size:
        raise ValueError("Mode-sign vector size must match the cubic-force tensor dimension.")
    out = np.zeros_like(phi3)
    for i in range(signs.size):
        for j in range(signs.size):
            for k in range(signs.size):
                out[i, j, k] = phi3[i, j, k] * signs[i] * signs[j] * signs[k]
    return out


def parse_gaussian_fchk_harmonic_data(path: str | Path) -> GaussianFchkHarmonicData:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    n_atoms = int(_parse_fchk_scalar(lines, "Number of atoms", "I"))
    n_modes = int(_parse_fchk_scalar(lines, "Number of Normal Modes", "I"))
    atomic_numbers = _parse_fchk_array(lines, "Atomic numbers", "I").astype(int)
    masses_amu = _parse_fchk_array(lines, "Real atomic weights", "R")
    coords_bohr = _parse_fchk_array(lines, "Current cartesian coordinates", "R").reshape(n_atoms, 3)
    vib_atmass_amu = _parse_fchk_array(lines, "Vib-AtMass", "R")
    vib_e2 = _parse_fchk_array(lines, "Vib-E2", "R")
    # Gaussian stores Vib-Modes mode-major in the formatted checkpoint.
    vib_modes = _parse_fchk_array(lines, "Vib-Modes", "R").reshape(n_modes, 3 * n_atoms).T
    hess_tri = _parse_fchk_array(lines, "Cartesian Force Constants", "R")
    point_group = _parse_fchk_char_value(lines, "Point Group")
    if point_group is not None:
        point_group = _normalize_point_group_label(point_group.replace("0", "").strip())
    dim = 3 * n_atoms
    cartesian_force_constants = np.zeros((dim, dim), dtype=float)
    p = 0
    for i in range(dim):
        for j in range(i + 1):
            cartesian_force_constants[i, j] = hess_tri[p]
            cartesian_force_constants[j, i] = hess_tri[p]
            p += 1
    return GaussianFchkHarmonicData(
        atomic_numbers=atomic_numbers,
        masses_amu=masses_amu,
        coords_bohr=coords_bohr,
        n_modes=n_modes,
        vib_atmass_amu=vib_atmass_amu,
        vib_e2=vib_e2,
        vib_modes=vib_modes,
        cartesian_force_constants=cartesian_force_constants,
        point_group=point_group,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse Gaussian/GDV VPT data from a log file.")
    ap.add_argument("log")
    args = ap.parse_args()

    harm = parse_gaussian_harmonic_data(args.log)
    quart = parse_gaussian_quartic_benchmark(args.log)
    sext = parse_gaussian_sextic_benchmark(args.log)
    anh = parse_gaussian_anharmonic_force_data(args.log)

    print("=== Harmonic data ===")
    print(f"n_atoms = {harm.atomic_numbers.size}")
    print(f"frequencies_cm = {harm.frequencies_cm}")
    print(f"reduced_masses_amu = {harm.reduced_masses_amu}")
    print(f"rot_ghz = {harm.rot_ghz}")
    print(f"moments_au = {harm.moments_au}")
    print("\n=== Quartic benchmark ===")
    print(f"tau_prime_mhz = {quart.tau_prime_mhz}")
    print(f"A_reduction_mhz = {quart.a_reduction_mhz}")
    print(f"S_reduction_mhz = {quart.s_reduction_mhz}")
    print(f"sigma = {quart.sigma}")
    print("\n=== Sextic benchmark ===")
    print(f"phi_cart_hz = {sext.phi_cart_hz}")
    print(f"A_sextic_hz = {sext.a_reduction_hz}")
    print(f"S_sextic_hz = {sext.s_reduction_hz}")
    print("\n=== Anharmonic force data ===")
    print(f"anharmonic_order_frequencies_cm = {anh.frequencies_cm}")


if __name__ == "__main__":
    main()
