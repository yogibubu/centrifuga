#!/usr/bin/env python3
"""RV3: Ro-Vibrational Van Vleck scaffold built on the current centrifugal backend.

This module does not try to finish the full RV3 program in one step. It fixes
the program architecture and the public API that Cedex can extend without
rewriting the existing harmonic, sextic, and linear backends.

Current live scope:
- geometry loading from xyz/log/fchk
- harmonic source loading from fchk
- order-2 harmonic model construction
- point-group and rotational-constant summary
- normal-mode irrep assignment
- quartic distortion from the harmonic route
- order-3 cubic loading from Gaussian log plus mode-order consistency checks
- order-3 sextic/H22 diagnostics through the existing backend
- Gaussian alpha parsing / mode filtering
- alpha from the harmonic model plus semi-diagonal cubic input

Current deferred scope:
- Der2/Der3/Der4 parsers (formats still to be fixed)
- general order-4 non-linear branch
- migration of the existing linear order-4 Aliev branch behind this API
- GVPT2 vibrational and ro-vibrational layers
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

from compare_gaussian_sextic import sextic_cubic_hierarchy_hz, sextic_h22_linear_candidate_hz, sextic_linear_source_formula_hz
from ceditt_gui import (
    _M4,
    _flip_handedness_abc,
    _flip_tau_last_two_axes,
    _norm_reduction,
    _norm_rep,
    _quartic_forward_constants,
    _quartic_reduced_3plus2_from_tau,
    _quartic_spectral_invariants_from_tau,
    _rotate_abc,
    _sextic_condition_metrics,
    _sextic_decomposition,
    _sextic_physical_subspace_residual,
    compute_T_over_B,
    compute_s111,
    quartic_transform_matrix,
    stability_metrics,
    transform_quartic_tensor,
    transform_sextic_tensor,
)
from distortion_workflow import compute_order2_quartic
from gaussian_vpt_parser import (
    align_gaussian_cubic_force_constants,
    frequency_reorder_map,
    parse_gaussian_alpha_data,
    parse_gaussian_anharmonic_force_data,
    parse_gaussian_fchk_harmonic_data,
    parse_gaussian_harmonic_data,
)
from h22_reference import project_h22_ceditt3_reference_path
from linear_dv_aliev_terms import (
    build_explicit_aliev_L_model,
    build_explicit_aliev_betas,
    build_explicit_aliev_dv_compactness_audit,
    build_explicit_aliev_dv_general_model,
    build_explicit_aliev_dv_legacy_compact_model,
    make_linear_aliev_explicit_inputs_from_mapping,
    resolve_linear_aliev_quartic_mode,
)
from rovib_distortion import (
    ANGSTROM_TO_BOHR,
    CMINV_TO_MHZ,
    HarmonicInertiaModel,
    Molecule,
    harmonic_inertia_model_from_geometry_hessian,
    inertia_tensor,
    read_xyz,
    rotational_constants,
)
from scripts.build_linear_aliev_payload_from_gaussian import build_payload as build_linear_aliev_payload
from symmetry_metadata import assign_normal_mode_irreps, point_group_from_geometry, rotor_type_for_symmetry, symbols_from_atomic_numbers
from vibrot_alpha import alpha_matrix_from_cubic_two_index_cm, read_cubic_two_index_matrix


RV3_ALLOWED_ORDERS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class RV3Source:
    path: str
    format: str = "auto"

    def resolved_path(self) -> Path:
        return Path(self.path).expanduser().resolve()


@dataclass(frozen=True)
class RV3Request:
    max_derivative_order: int
    geometry: RV3Source | None = None
    hessian: RV3Source | None = None
    cubic: RV3Source | None = None
    quartic: RV3Source | None = None
    alpha_log: RV3Source | None = None
    alpha_cubic_two_index: RV3Source | None = None
    alpha_excluded_modes: tuple[int, ...] = ()
    representation: str = "I"


@dataclass
class RV3GeometryStage:
    source_path: str
    source_format: str
    n_atoms: int
    symbols: list[str]
    point_group: str
    rotational_symmetry_number: int
    rotor_type: str
    moments_amu_a2: list[float]
    abc_mhz: list[float]


@dataclass
class RV3HarmonicStage:
    source_path: str
    source_format: str
    representation: str
    n_modes: int
    frequencies_cm: list[float]
    mode_irreps: list[str] | None
    rotor_limit: dict[str, Any]
    watson_a_mhz: dict[str, float]
    watson_s_mhz: dict[str, float]
    special_quartic_projection: dict[str, Any] | None
    linear_ltype_terms_present: bool


@dataclass
class RV3CubicStage:
    source_path: str
    source_format: str
    mode_reorder_map: list[int]
    phi3_reduced_shape: list[int]
    sextic_h22_linear_candidate_hz: dict[str, Any] | None
    sextic_cubic_hierarchy_hz: dict[str, Any]
    sextic_linear_source_formula_hz: dict[str, Any] | None
    notes: list[str] = field(default_factory=list)


@dataclass
class RV3QuarticH22Stage:
    fchk_path: str
    log_path: str
    projection_mhz: dict[str, float]


@dataclass
class RV3AlphaParserStage:
    source_path: str
    source_format: str
    axis_labels: list[str]
    excluded_modes: list[int]
    total_alpha_mhz: list[float]
    kept_alpha_mhz: list[float]
    removed_alpha_mhz: list[float]
    total_alpha_cm: list[float]
    kept_alpha_cm: list[float]
    removed_alpha_cm: list[float]
    mode_rows_mhz: list[dict[str, Any]]


@dataclass
class RV3AlphaInternalStage:
    source_path: str
    source_format: str
    cubic_matrix_shape: list[int]
    cubic_matrix_origin: str
    excluded_modes: list[int]
    alpha_total_sum_mhz: list[float]
    alpha_component_sums_mhz: dict[str, list[float]]
    projection_strategy: str | None
    rotor_limit: dict[str, Any] | None
    special_limit_summary_mhz: dict[str, float] | None
    benchmark_total_sum_mhz: list[float] | None
    benchmark_difference_mhz: list[float] | None
    mode_rows_mhz: list[dict[str, Any]]


@dataclass
class RV3QuarticStage:
    source_path: str
    source_format: str
    mode_reorder_map: list[int]
    phi4_reduced_shape: list[int]
    linear_branch_status: str
    linear_general_observable: dict[str, Any] | None = None
    linear_legacy_compact_observable: dict[str, Any] | None = None
    linear_compactness_audit: dict[str, Any] | None = None
    linear_optical_constant: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class RV3Result:
    request: dict[str, Any]
    geometry_stage: RV3GeometryStage
    harmonic_stage: RV3HarmonicStage | None
    cubic_stage: RV3CubicStage | None
    quartic_h22_stage: RV3QuarticH22Stage | None
    alpha_parser_stage: RV3AlphaParserStage | None
    alpha_internal_stage: RV3AlphaInternalStage | None
    quartic_stage: RV3QuarticStage | None
    pending_work: list[str]

    def to_jsonable(self) -> dict[str, Any]:
        return _sanitize_jsonable(asdict(self))


@dataclass(frozen=True)
class RV3ManualQuarticRequest:
    A_mhz: float
    B_mhz: float
    C_mhz: float
    rep_in: str
    reduction: str
    constants: list[float]


@dataclass
class RV3ManualQuarticResult:
    request: dict[str, Any]
    outputs: dict[str, dict[str, Any]]
    tau_input: list[float]
    spectral_invariants: dict[str, Any]
    reduced_3plus2: dict[str, Any]
    handedness_flip: dict[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        return _sanitize_jsonable(asdict(self))


@dataclass(frozen=True)
class RV3ManualSexticRequest:
    A_mhz: float
    B_mhz: float
    C_mhz: float
    rep_in: str
    reduction_in: str
    reduction_out: str
    constants: list[float]


@dataclass
class RV3ManualSexticResult:
    request: dict[str, Any]
    outputs: dict[str, dict[str, Any]]
    physical_subspace_input: dict[str, Any]
    handedness_flip: dict[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        return _sanitize_jsonable(asdict(self))


@dataclass(frozen=True)
class _LoadedGeometry:
    molecule: Molecule
    source_path: str
    source_format: str


@dataclass(frozen=True)
class _LoadedHessian:
    source_path: str
    source_format: str
    masses_amu: np.ndarray
    coords_ang: np.ndarray
    hessian: np.ndarray
    symbols: list[str]


def _parse_mode_selection(spec: str) -> set[int]:
    out: set[int] = set()
    text = spec.strip()
    if not text:
        return out
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            i = int(left.strip())
            j = int(right.strip())
            if i <= 0 or j <= 0:
                raise ValueError("Mode indices must be positive integers.")
            if j < i:
                i, j = j, i
            out.update(range(i, j + 1))
            continue
        idx = int(item)
        if idx <= 0:
            raise ValueError("Mode indices must be positive integers.")
        out.add(idx)
    return out


def _sanitize_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize_jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def _format_compact(obj: Any) -> str:
    return json.dumps(_sanitize_jsonable(obj), sort_keys=True)


def run_manual_quartic_transform(request: RV3ManualQuarticRequest) -> RV3ManualQuarticResult:
    rep_in = _norm_rep(request.rep_in)
    red = _norm_reduction(request.reduction)
    A = float(request.A_mhz)
    B = float(request.B_mhz)
    C = float(request.C_mhz)
    d_in = np.asarray(request.constants, dtype=float).reshape(5)
    rotor_type = rotor_type_for_symmetry(np.asarray([A, B, C], dtype=float), np.asarray([1.0, 2.0, 3.0], dtype=float))
    if rotor_type != "asymmetric":
        raise ValueError(
            "Manual quartic transform is reserved for asymmetric-top transforms. "
            "Use the harmonic RV3 route for exact symmetric-top or linear limits."
        )

    rep_outs = [rep for rep in ("I", "II", "III") if rep != rep_in]
    m_in = _M4(A, B, C, red)
    tau_in = np.linalg.pinv(m_in) @ d_in
    spec_in = _quartic_spectral_invariants_from_tau(tau_in)
    red5_in = _quartic_reduced_3plus2_from_tau(tau_in, A, B, C)
    outputs: dict[str, dict[str, Any]] = {}
    for rep_out in rep_outs:
        d_out = transform_quartic_tensor(d_in, A, B, C, rep_in, rep_out, red)
        A2, B2, C2 = _rotate_abc(A, B, C, rep_in, rep_out)
        tmat = quartic_transform_matrix(A, B, C, rep_in, rep_out, red, "tensor")
        metrics = stability_metrics(tmat, A, B, C)
        m_out = _M4(A2, B2, C2, red)
        tau_out = np.linalg.pinv(m_out) @ d_out
        outputs[rep_out] = {
            "A_mhz": A2,
            "B_mhz": B2,
            "C_mhz": C2,
            "constants": [float(x) for x in d_out],
            "tensor_roundtrip_max_error": float(np.max(np.abs(transform_quartic_tensor(d_out, A2, B2, C2, rep_out, rep_in, red) - d_in))),
            "stability_metrics": metrics,
            "s111": compute_s111(A2, B2, C2, d_out, red),
            "T_over_B": compute_T_over_B(B2, d_out),
            "spectral_invariants": spec_in if rep_out == rep_in else _quartic_spectral_invariants_from_tau(tau_out),
            "reduced_3plus2": _quartic_reduced_3plus2_from_tau(tau_out, A2, B2, C2),
        }
    flip_abc = _flip_handedness_abc(A, B, C)
    tau_flip = _flip_tau_last_two_axes(tau_in)
    d_flip = _quartic_forward_constants(red, tau_flip, *flip_abc)
    return RV3ManualQuarticResult(
        request={
            "A_mhz": A,
            "B_mhz": B,
            "C_mhz": C,
            "rep_in": rep_in,
            "reduction": red,
            "constants": [float(x) for x in d_in],
        },
        outputs=outputs,
        tau_input=[float(x) for x in tau_in],
        spectral_invariants=spec_in,
        reduced_3plus2=red5_in,
        handedness_flip={
            "A_mhz": flip_abc[0],
            "B_mhz": flip_abc[1],
            "C_mhz": flip_abc[2],
            "constants": [float(x) for x in d_flip],
        },
    )


def run_manual_sextic_transform(request: RV3ManualSexticRequest) -> RV3ManualSexticResult:
    rep_in = _norm_rep(request.rep_in)
    red_in = _norm_reduction(request.reduction_in)
    red_out = _norm_reduction(request.reduction_out)
    A = float(request.A_mhz)
    B = float(request.B_mhz)
    C = float(request.C_mhz)
    h_in = np.asarray(request.constants, dtype=float).reshape(7)
    rotor_type = rotor_type_for_symmetry(np.asarray([A, B, C], dtype=float), np.asarray([1.0, 2.0, 3.0], dtype=float))
    if rotor_type != "asymmetric":
        raise ValueError(
            "Manual sextic transform is reserved for asymmetric-top transforms. "
            "Use the harmonic RV3 route for exact symmetric-top or linear limits."
        )

    rep_outs = [rep for rep in ("I", "II", "III") if rep != rep_in]
    phys_in = _sextic_physical_subspace_residual(h_in, rep_in, red_in, A, B, C)
    dec_in = _sextic_decomposition(h_in, rep_in, red_in, A, B, C)
    outputs: dict[str, dict[str, Any]] = {}
    for rep_out in rep_outs:
        h_out = transform_sextic_tensor(h_in, A, B, C, rep_in, rep_out, red_in, red_out)
        A2, B2, C2 = _rotate_abc(A, B, C, rep_in, rep_out)
        outputs[rep_out] = {
            "A_mhz": A2,
            "B_mhz": B2,
            "C_mhz": C2,
            "constants": [float(x) for x in h_out],
            "physical_subspace_residual": _sextic_physical_subspace_residual(h_out, rep_out, red_out, A2, B2, C2),
            "roundtrip_max_error": float(np.max(np.abs(transform_sextic_tensor(h_out, A2, B2, C2, rep_out, rep_in, red_out, red_in) - h_in))),
            "decomposition": _sextic_decomposition(h_out, rep_out, red_out, A2, B2, C2),
            "condition_metrics": _sextic_condition_metrics(rep_in, rep_out),
        }
    flip_abc = _flip_handedness_abc(A, B, C)
    return RV3ManualSexticResult(
        request={
            "A_mhz": A,
            "B_mhz": B,
            "C_mhz": C,
            "rep_in": rep_in,
            "reduction_in": red_in,
            "reduction_out": red_out,
            "constants": [float(x) for x in h_in],
        },
        outputs=outputs,
        physical_subspace_input={
            "residual": phys_in,
            "decomposition": dec_in,
        },
        handedness_flip={
            "A_mhz": flip_abc[0],
            "B_mhz": flip_abc[1],
            "C_mhz": flip_abc[2],
            "note": (
                "The fixed-representation r<->l axis swap does not define an independently "
                "validated opposite-handed sextic target inside the current 5D transport."
            ),
        },
    )


def _infer_format(source: RV3Source, *, role: str) -> str:
    fmt = source.format.strip().lower()
    if fmt != "auto":
        return fmt
    suffix = source.resolved_path().suffix.lower()
    if suffix == ".xyz":
        return "xyz"
    if suffix in {".log", ".out"}:
        return "log"
    if suffix == ".fchk":
        return "fchk"
    if role == "hessian":
        return "der2"
    if role == "cubic":
        return "der3"
    if role == "quartic":
        return "der4"
    raise ValueError(f"Could not infer format for {role} source {source.path!r}.")


def _validate_request(request: RV3Request) -> None:
    if request.max_derivative_order not in RV3_ALLOWED_ORDERS:
        raise ValueError(f"max_derivative_order must be one of {RV3_ALLOWED_ORDERS}.")
    if request.max_derivative_order >= 2 and request.hessian is None:
        raise ValueError("Order >= 2 requires a Hessian source.")
    if request.max_derivative_order >= 3 and request.cubic is None:
        raise ValueError("Order >= 3 requires a cubic-derivative source.")
    if request.max_derivative_order >= 4 and request.quartic is None:
        raise ValueError("Order >= 4 requires a quartic-derivative source.")
    if request.geometry is None and request.hessian is None:
        raise ValueError("At least one geometry-bearing source must be provided.")


def _load_geometry(source: RV3Source) -> _LoadedGeometry:
    fmt = _infer_format(source, role="geometry")
    path = source.resolved_path()
    if fmt == "xyz":
        mol = read_xyz(path)
        return _LoadedGeometry(molecule=mol, source_path=str(path), source_format=fmt)
    if fmt == "log":
        harm = parse_gaussian_harmonic_data(path)
        mol = Molecule(
            symbols=symbols_from_atomic_numbers(harm.atomic_numbers),
            masses_amu=np.asarray(harm.masses_amu, dtype=float),
            coords_ang=np.asarray(harm.coords_std_ang, dtype=float),
        )
        return _LoadedGeometry(molecule=mol, source_path=str(path), source_format=fmt)
    if fmt == "fchk":
        harm = parse_gaussian_fchk_harmonic_data(path)
        mol = Molecule(
            symbols=symbols_from_atomic_numbers(harm.atomic_numbers),
            masses_amu=np.asarray(harm.masses_amu, dtype=float),
            coords_ang=np.asarray(harm.coords_bohr, dtype=float) / ANGSTROM_TO_BOHR,
        )
        return _LoadedGeometry(molecule=mol, source_path=str(path), source_format=fmt)
    raise ValueError(f"Unsupported geometry format {fmt!r}.")


def _load_hessian(source: RV3Source) -> _LoadedHessian:
    fmt = _infer_format(source, role="hessian")
    path = source.resolved_path()
    if fmt == "fchk":
        harm = parse_gaussian_fchk_harmonic_data(path)
        return _LoadedHessian(
            source_path=str(path),
            source_format=fmt,
            masses_amu=np.asarray(harm.masses_amu, dtype=float),
            coords_ang=np.asarray(harm.coords_bohr, dtype=float) / ANGSTROM_TO_BOHR,
            hessian=np.asarray(harm.cartesian_force_constants, dtype=float),
            symbols=symbols_from_atomic_numbers(harm.atomic_numbers),
        )
    if fmt == "der2":
        raise NotImplementedError("Der2 parsing is reserved for the RV3 follow-up implementation.")
    raise ValueError(f"Unsupported Hessian format {fmt!r}.")


def _load_anharmonic(source: RV3Source, *, role: str):
    fmt = _infer_format(source, role=role)
    path = source.resolved_path()
    if fmt == "log":
        return fmt, str(path), parse_gaussian_anharmonic_force_data(path)
    if fmt in {"der3", "der4"}:
        raise NotImplementedError(f"{fmt.upper()} parsing is reserved for the RV3 follow-up implementation.")
    raise ValueError(f"Unsupported {role} format {fmt!r}.")


def _reorder_quartic_force_constants(phi4: np.ndarray, source_to_target: np.ndarray) -> np.ndarray:
    mapping = np.asarray(source_to_target, dtype=int)
    out = np.zeros_like(phi4)
    n_modes = mapping.size
    for i in range(n_modes):
        for j in range(n_modes):
            for k in range(n_modes):
                for l in range(n_modes):
                    out[mapping[i], mapping[j], mapping[k], mapping[l]] = phi4[i, j, k, l]
    return out


def _reduce_cubic_to_two_index_matrix(phi3_reduced_cm: np.ndarray) -> np.ndarray:
    phi3 = np.asarray(phi3_reduced_cm, dtype=float)
    if phi3.ndim != 3 or not (phi3.shape[0] == phi3.shape[1] == phi3.shape[2]):
        raise ValueError(f"Unexpected reduced cubic tensor shape: {phi3.shape}")
    n_modes = phi3.shape[0]
    out = np.zeros((n_modes, n_modes), dtype=float)
    for i in range(n_modes):
        out[i, i] = phi3[i, i, i]
        for j in range(n_modes):
            if i == j:
                continue
            out[i, j] = phi3[i, i, j]
    return out


def _validate_symbol_sequence(expected: list[str], observed: list[str], *, context: str) -> None:
    if len(expected) != len(observed):
        raise ValueError(f"{context}: atom count mismatch ({len(expected)} vs {len(observed)}).")
    if [x.upper() for x in expected] != [x.upper() for x in observed]:
        raise ValueError(f"{context}: atomic ordering mismatch between sources.")


def _geometry_stage_from_molecule(loaded: _LoadedGeometry) -> RV3GeometryStage:
    mol = loaded.molecule
    i_tensor, _com = inertia_tensor(mol.masses_amu, mol.coords_ang)
    moments, abc_mhz, principal_axes = rotational_constants(i_tensor)
    coords_pa = (np.asarray(mol.coords_ang, dtype=float) - np.average(mol.coords_ang, axis=0, weights=mol.masses_amu)) @ principal_axes
    rotor_type = rotor_type_for_symmetry(abc_mhz, moments)
    sigma, point_group = point_group_from_geometry(list(mol.symbols), coords_pa, rotor_type)
    return RV3GeometryStage(
        source_path=loaded.source_path,
        source_format=loaded.source_format,
        n_atoms=mol.n_atoms,
        symbols=list(mol.symbols),
        point_group=point_group,
        rotational_symmetry_number=int(sigma),
        rotor_type=rotor_type,
        moments_amu_a2=[float(x) for x in moments],
        abc_mhz=[float(x) for x in abc_mhz],
    )


def _harmonic_stage_from_sources(
    *,
    geometry: _LoadedGeometry | None,
    hessian: _LoadedHessian,
    representation: str,
    point_group: str,
) -> tuple[HarmonicInertiaModel, RV3HarmonicStage]:
    if geometry is not None:
        _validate_symbol_sequence(geometry.molecule.symbols, hessian.symbols, context="Geometry/Hessian consistency")
        coords_ang = np.asarray(geometry.molecule.coords_ang, dtype=float)
    else:
        coords_ang = np.asarray(hessian.coords_ang, dtype=float)
    model = harmonic_inertia_model_from_geometry_hessian(
        hessian.masses_amu,
        coords_ang,
        hessian.hessian,
        representation=representation,
        symbols=hessian.symbols,
        point_group=point_group,
    )
    irreps = assign_normal_mode_irreps(model)
    quartic = compute_order2_quartic(model)
    stage = RV3HarmonicStage(
        source_path=hessian.source_path,
        source_format=hessian.source_format,
        representation=representation,
        n_modes=int(len(model.vib_freq_cm)),
        frequencies_cm=[float(x) for x in np.asarray(model.vib_freq_cm, dtype=float)],
        mode_irreps=None if irreps is None else [str(x) for x in irreps],
        rotor_limit=quartic["rotor_limit"],
        watson_a_mhz={str(k): float(v) for k, v in quartic["watson_a_mhz"].items()},
        watson_s_mhz={str(k): float(v) for k, v in quartic["watson_s_mhz"].items()},
        special_quartic_projection=quartic["special_quartic_projection"],
        linear_ltype_terms_present=quartic["linear_ltype_terms"] is not None,
    )
    return model, stage


def _axes_from_model(model: HarmonicInertiaModel) -> dict[str, int]:
    return {str(label): int(idx) for idx, label in enumerate(model.xyz_to_abc)}


def _cubic_stage_from_source(model: HarmonicInertiaModel, source: RV3Source) -> RV3CubicStage:
    fmt, path, anh = _load_anharmonic(source, role="cubic")
    mapping, phi3_reduced_cm, _phi3_raw = align_gaussian_cubic_force_constants(anh, np.asarray(model.vib_freq_cm, dtype=float))
    axes = _axes_from_model(model)
    rotor_type = rotor_type_for_symmetry(np.asarray(model.abc_mhz, dtype=float), np.asarray(model.moments_amu_a2, dtype=float))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"compare_gaussian_sextic")
        h22 = sextic_h22_linear_candidate_hz(model, axes)
        hierarchy = sextic_cubic_hierarchy_hz(model, phi3_reduced_cm, axes)
        linear_source = sextic_linear_source_formula_hz(model, phi3_reduced_cm) if rotor_type == "linear" else None
    notes = [
        "Mode ordering aligned by harmonic frequency matching.",
        "This stage validates cubic-source consistency with the harmonic normal-mode basis.",
    ]
    return RV3CubicStage(
        source_path=path,
        source_format=fmt,
        mode_reorder_map=[int(x) for x in mapping],
        phi3_reduced_shape=[int(x) for x in phi3_reduced_cm.shape],
        sextic_h22_linear_candidate_hz=h22,
        sextic_cubic_hierarchy_hz=hierarchy,
        sextic_linear_source_formula_hz=linear_source,
        notes=notes,
    )


def _alpha_parser_stage_from_log(
    source: RV3Source,
    *,
    excluded_modes: tuple[int, ...] | list[int] | set[int] = (),
) -> RV3AlphaParserStage:
    fmt = _infer_format(source, role="cubic")
    if fmt != "log":
        raise ValueError("Gaussian alpha parsing currently requires a Gaussian anharmonic log source.")
    path = str(source.resolved_path())
    bench = parse_gaussian_alpha_data(path)
    excluded = sorted(int(x) for x in excluded_modes)
    mode_indices = np.asarray(bench.mode_indices, dtype=int)
    n_modes = int(mode_indices.size)
    invalid = sorted(idx for idx in excluded if idx < 1 or idx > n_modes)
    if invalid:
        raise ValueError(f"Excluded mode indices out of range: {invalid}")
    keep_mask = np.array([idx not in excluded for idx in mode_indices], dtype=bool)
    alpha_mhz = np.asarray(bench.alpha_mhz, dtype=float)
    alpha_cm = np.asarray(bench.alpha_cm, dtype=float)
    total_mhz = np.sum(alpha_mhz, axis=0)
    kept_mhz = np.sum(alpha_mhz[keep_mask], axis=0) if np.any(keep_mask) else np.zeros(3, dtype=float)
    total_cm = np.sum(alpha_cm, axis=0)
    kept_cm = np.sum(alpha_cm[keep_mask], axis=0) if np.any(keep_mask) else np.zeros(3, dtype=float)
    mode_rows: list[dict[str, Any]] = []
    harm = parse_gaussian_harmonic_data(path).reordered_to_anharmonic()
    freqs = np.asarray(harm.frequencies_cm, dtype=float)
    for i, mode_idx in enumerate(mode_indices):
        row = alpha_mhz[i]
        mode_rows.append(
            {
                "mode": int(mode_idx),
                "frequency_cm": float(freqs[i]) if i < freqs.size else None,
                "status": "excluded" if int(mode_idx) in excluded else "kept",
                "values_mhz": [float(x) for x in row],
            }
        )
    return RV3AlphaParserStage(
        source_path=path,
        source_format=fmt,
        axis_labels=[str(x) for x in bench.axis_labels],
        excluded_modes=excluded,
        total_alpha_mhz=[float(x) for x in total_mhz],
        kept_alpha_mhz=[float(x) for x in kept_mhz],
        removed_alpha_mhz=[float(x) for x in (total_mhz - kept_mhz)],
        total_alpha_cm=[float(x) for x in total_cm],
        kept_alpha_cm=[float(x) for x in kept_cm],
        removed_alpha_cm=[float(x) for x in (total_cm - kept_cm)],
        mode_rows_mhz=mode_rows,
    )


def _alpha_internal_stage_from_sources(
    model: HarmonicInertiaModel,
    *,
    cubic_source: RV3Source | None,
    alpha_cubic_two_index: RV3Source | None,
    alpha_log_source: RV3Source | None,
    excluded_modes: tuple[int, ...] | list[int] | set[int] = (),
) -> RV3AlphaInternalStage | None:
    if alpha_cubic_two_index is None and cubic_source is None:
        return None
    n_modes = int(np.abs(np.asarray(model.vib_freq_cm, dtype=float)).size)
    excluded = sorted(int(x) for x in excluded_modes)
    invalid = sorted(idx for idx in excluded if idx < 1 or idx > n_modes)
    if invalid:
        raise ValueError(f"Excluded mode indices out of range: {invalid}")

    origin = ""
    source_path = ""
    source_format = ""
    if alpha_cubic_two_index is not None:
        source_path = str(alpha_cubic_two_index.resolved_path())
        source_format = _infer_format(alpha_cubic_two_index, role="cubic")
        mat = read_cubic_two_index_matrix(source_path, n_modes)
        origin = "provided_2index"
    else:
        assert cubic_source is not None
        fmt, path, anh = _load_anharmonic(cubic_source, role="cubic")
        _mapping, phi3_reduced_cm, _phi3_raw = align_gaussian_cubic_force_constants(anh, np.asarray(model.vib_freq_cm, dtype=float))
        mat = _reduce_cubic_to_two_index_matrix(phi3_reduced_cm)
        source_path = path
        source_format = fmt
        origin = "derived_from_cubic_log"

    alpha = alpha_matrix_from_cubic_two_index_cm(model, mat, excluded_modes=set(excluded))
    total_cm = np.asarray(alpha["alpha_total_cm_abc"], dtype=float)
    total_mhz = total_cm * CMINV_TO_MHZ
    cor_mhz = np.asarray(alpha["alpha_coriolis_cm_abc"], dtype=float) * CMINV_TO_MHZ
    inertia_mhz = np.asarray(alpha["alpha_inertia_cm_abc"], dtype=float) * CMINV_TO_MHZ
    anh_mhz = np.asarray(alpha["alpha_anharmonic_cm_abc"], dtype=float) * CMINV_TO_MHZ
    anh_diag_mhz = np.asarray(alpha["alpha_anharmonic_diagonal_cm_abc"], dtype=float) * CMINV_TO_MHZ
    anh_sd_mhz = np.asarray(alpha["alpha_anharmonic_semidiagonal_cm_abc"], dtype=float) * CMINV_TO_MHZ

    special_limit_summary_mhz = None
    rotor_limit = alpha.get("rotor_limit")
    if rotor_limit is not None and rotor_limit["is_special_limit"]:
        if rotor_limit["kind"] == "linear" and "alpha_linear_cm" in alpha:
            alpha_lin_mhz = np.asarray(alpha["alpha_linear_cm"], dtype=float) * CMINV_TO_MHZ
            special_limit_summary_mhz = {"linear_perpendicular_sum_mhz": float(np.sum(alpha_lin_mhz))}
        elif "alpha_axial_cm" in alpha:
            axial = alpha["alpha_axial_cm"]
            par_mhz = np.asarray(axial["parallel"], dtype=float) * CMINV_TO_MHZ
            perp_mhz = np.asarray(axial["perpendicular"], dtype=float) * CMINV_TO_MHZ
            special_limit_summary_mhz = {
                "parallel_sum_mhz": float(np.sum(par_mhz)),
                "perpendicular_sum_mhz": float(np.sum(perp_mhz)),
            }

    benchmark_total_sum_mhz = None
    benchmark_difference_mhz = None
    bench_source = alpha_log_source if alpha_log_source is not None else cubic_source
    if bench_source is not None and _infer_format(bench_source, role="cubic") == "log":
        bench = parse_gaussian_alpha_data(str(bench_source.resolved_path()))
        bench_sum_mhz = np.sum(np.asarray(bench.alpha_mhz, dtype=float), axis=0)
        benchmark_total_sum_mhz = [float(x) for x in bench_sum_mhz]
        benchmark_difference_mhz = [float(x) for x in (np.sum(total_mhz, axis=0) - bench_sum_mhz)]

    mode_rows = []
    for i in range(n_modes):
        mode_rows.append(
            {
                "mode": i + 1,
                "frequency_cm": float(np.asarray(model.vib_freq_cm, dtype=float)[i]),
                "status": "excluded" if (i + 1) in excluded else "kept",
                "total_mhz": [float(x) for x in total_mhz[i]],
                "coriolis_mhz": [float(x) for x in cor_mhz[i]],
                "inertia_mhz": [float(x) for x in inertia_mhz[i]],
                "anharmonic_diag_mhz": [float(x) for x in anh_diag_mhz[i]],
                "anharmonic_semidiagonal_mhz": [float(x) for x in anh_sd_mhz[i]],
            }
        )
    return RV3AlphaInternalStage(
        source_path=source_path,
        source_format=source_format,
        cubic_matrix_shape=[int(x) for x in mat.shape],
        cubic_matrix_origin=origin,
        excluded_modes=excluded,
        alpha_total_sum_mhz=[float(x) for x in np.sum(total_mhz, axis=0)],
        alpha_component_sums_mhz={
            "coriolis": [float(x) for x in np.sum(cor_mhz, axis=0)],
            "inertia": [float(x) for x in np.sum(inertia_mhz, axis=0)],
            "anharmonic": [float(x) for x in np.sum(anh_mhz, axis=0)],
            "anharmonic_diag": [float(x) for x in np.sum(anh_diag_mhz, axis=0)],
            "anharmonic_semidiagonal": [float(x) for x in np.sum(anh_sd_mhz, axis=0)],
        },
        projection_strategy=None if alpha.get("projection_strategy") is None else str(alpha["projection_strategy"]),
        rotor_limit=alpha.get("rotor_limit"),
        special_limit_summary_mhz=special_limit_summary_mhz,
        benchmark_total_sum_mhz=benchmark_total_sum_mhz,
        benchmark_difference_mhz=benchmark_difference_mhz,
        mode_rows_mhz=mode_rows,
    )


def _quartic_h22_stage_from_sources(
    *,
    harmonic_fchk_path: str | None,
    benchmark_log_source: RV3Source | None,
) -> RV3QuarticH22Stage | None:
    if harmonic_fchk_path is None or benchmark_log_source is None:
        return None
    if _infer_format(benchmark_log_source, role="quartic") != "log":
        return None
    log_path = str(benchmark_log_source.resolved_path())
    try:
        proj = project_h22_ceditt3_reference_path(harmonic_fchk_path, log_path)
    except Exception:
        return None
    return RV3QuarticH22Stage(
        fchk_path=str(harmonic_fchk_path),
        log_path=log_path,
        projection_mhz={str(k): float(v) for k, v in proj.items()},
    )


def _quartic_stage_from_source(
    model: HarmonicInertiaModel,
    source: RV3Source,
    *,
    harmonic_fchk_path: str | None,
) -> RV3QuarticStage:
    fmt, path, anh = _load_anharmonic(source, role="quartic")
    mapping = frequency_reorder_map(anh.frequencies_cm, np.asarray(model.vib_freq_cm, dtype=float))
    phi4_reduced_cm = _reorder_quartic_force_constants(np.asarray(anh.phi4_reduced_cm, dtype=float), mapping)
    rotor_type = rotor_type_for_symmetry(np.asarray(model.abc_mhz, dtype=float), np.asarray(model.moments_amu_a2, dtype=float))
    notes = [
        "Mode ordering aligned by harmonic frequency matching.",
        "Quartic source consistency is validated against the internally built harmonic basis.",
    ]
    branch_status = "nonlinear_order4_pending"
    linear_general_observable = None
    linear_legacy_compact_observable = None
    linear_compactness_audit = None
    linear_optical_constant = None
    if rotor_type == "linear":
        if harmonic_fchk_path is None:
            raise ValueError("Linear order-4 RV3 branch currently requires the harmonic Hessian source to be an fchk.")
        payload = build_linear_aliev_payload(
            fchk_path=harmonic_fchk_path,
            log_path=path,
            zeta_reduction="pair_offdiag",
            force_constant_source="reduced",
            pair_seed_source="gaussian_qe_source",
        )
        quartic_mode = resolve_linear_aliev_quartic_mode(
            k4_parallel=payload.get("k4_parallel"),
            k4_reduced=payload.get("k4_reduced"),
            quartic_mode="auto",
        )
        inputs = make_linear_aliev_explicit_inputs_from_mapping(payload, quartic_mode=quartic_mode)
        betas = build_explicit_aliev_betas(inputs)
        general_model = build_explicit_aliev_dv_general_model(inputs)
        legacy_model = build_explicit_aliev_dv_legacy_compact_model(inputs)
        audit = build_explicit_aliev_dv_compactness_audit(inputs)
        l_model = build_explicit_aliev_L_model(inputs)
        n_parallel = len(inputs.omega_parallel)
        ground_state = tuple(0 for _ in general_model.mode_kinds)
        linear_general_observable = {
            "quartic_mode": quartic_mode,
            "mode_kinds": list(general_model.mode_kinds),
            "ground_state_dv_cm": float(sp.N(general_model.value_for_state(ground_state))),
            "beta_parallel_mode_terms_cm": {
                str(idx): float(sp.N(val)) for idx, val in general_model.beta_parallel_mode_terms.items()
            },
            "beta_parallel_pair_terms_cm": {
                f"{i},{j}": float(sp.N(val)) for (i, j), val in general_model.beta_parallel_pair_terms.items()
            },
            "beta_perpendicular_cm": {
                str(idx): float(sp.N(val)) for idx, val in general_model.beta_perpendicular.items()
            },
        }
        linear_legacy_compact_observable = {
            "ground_state_dv_cm": float(sp.N(legacy_model.value_for_state(ground_state))),
            "beta_parallel_cm": {str(idx): float(sp.N(val)) for idx, val in betas.beta_parallel.items()},
            "beta_perpendicular_cm": {str(idx): float(sp.N(val)) for idx, val in betas.beta_perpendicular.items()},
        }
        linear_compactness_audit = {
            "max_parallel_mode_term_cm": float(sp.N(audit.max_parallel_mode_term)),
            "max_parallel_pair_term_cm": float(sp.N(audit.max_parallel_pair_term)),
            "pair_to_mode_ratio": float(sp.N(audit.pair_to_mode_ratio)),
        }
        linear_optical_constant = {
            "L_cm": float(sp.N(l_model.value)),
            "shared_offset_cm": float(sp.N(l_model.shared_offset)),
            "mode_contributions_cm": [float(sp.N(x)) for x in l_model.mode_contributions],
            "mode_contributions_with_shared_offset_cm": [
                float(sp.N(x)) for x in l_model.mode_contributions_with_shared_offset
            ],
        }
        branch_status = "linear_order4_general_branch_live"
        notes.append(
            "For linear molecules RV3 now calls the existing general/decompacted linear-Aliev observable branch."
        )
    return RV3QuarticStage(
        source_path=path,
        source_format=fmt,
        mode_reorder_map=[int(x) for x in mapping],
        phi4_reduced_shape=[int(x) for x in phi4_reduced_cm.shape],
        linear_branch_status=branch_status,
        linear_general_observable=linear_general_observable,
        linear_legacy_compact_observable=linear_legacy_compact_observable,
        linear_compactness_audit=linear_compactness_audit,
        linear_optical_constant=linear_optical_constant,
        notes=notes,
    )


def run_rv3(request: RV3Request) -> RV3Result:
    _validate_request(request)
    geometry_loaded = _load_geometry(request.geometry) if request.geometry is not None else None
    if geometry_loaded is None and request.hessian is not None:
        geometry_loaded = _load_geometry(request.hessian)
    if geometry_loaded is None:
        raise ValueError("RV3 could not obtain geometry from the provided sources.")

    geometry_stage = _geometry_stage_from_molecule(geometry_loaded)

    harmonic_stage = None
    cubic_stage = None
    quartic_h22_stage = None
    alpha_parser_stage = None
    alpha_internal_stage = None
    quartic_stage = None
    pending_work = [
        "Der2/Der3/Der4 external formats are not wired yet.",
        "GVPT2 vibrational and ro-vibrational layers are still pending.",
    ]

    if request.max_derivative_order >= 2:
        assert request.hessian is not None
        hessian_loaded = _load_hessian(request.hessian)
        model, harmonic_stage = _harmonic_stage_from_sources(
            geometry=geometry_loaded,
            hessian=hessian_loaded,
            representation=request.representation,
            point_group=geometry_stage.point_group,
        )
        benchmark_log_source = request.quartic or request.cubic or request.alpha_log
        quartic_h22_stage = _quartic_h22_stage_from_sources(
            harmonic_fchk_path=hessian_loaded.source_path if hessian_loaded.source_format == "fchk" else None,
            benchmark_log_source=benchmark_log_source,
        )
        if request.max_derivative_order >= 3:
            assert request.cubic is not None
            cubic_stage = _cubic_stage_from_source(model, request.cubic)
            alpha_log_source = request.alpha_log if request.alpha_log is not None else request.cubic
            if alpha_log_source is not None:
                try:
                    alpha_parser_stage = _alpha_parser_stage_from_log(
                        alpha_log_source,
                        excluded_modes=request.alpha_excluded_modes,
                    )
                except Exception:
                    alpha_parser_stage = None
            alpha_internal_stage = _alpha_internal_stage_from_sources(
                model,
                cubic_source=request.cubic,
                alpha_cubic_two_index=request.alpha_cubic_two_index,
                alpha_log_source=alpha_log_source,
                excluded_modes=request.alpha_excluded_modes,
            )
        if request.max_derivative_order >= 4:
            assert request.quartic is not None
            quartic_stage = _quartic_stage_from_source(
                model,
                request.quartic,
                harmonic_fchk_path=hessian_loaded.source_path if hessian_loaded.source_format == "fchk" else None,
            )

    return RV3Result(
        request={
            "max_derivative_order": int(request.max_derivative_order),
            "geometry": None if request.geometry is None else asdict(request.geometry),
            "hessian": None if request.hessian is None else asdict(request.hessian),
            "cubic": None if request.cubic is None else asdict(request.cubic),
            "quartic": None if request.quartic is None else asdict(request.quartic),
            "alpha_log": None if request.alpha_log is None else asdict(request.alpha_log),
            "alpha_cubic_two_index": None if request.alpha_cubic_two_index is None else asdict(request.alpha_cubic_two_index),
            "alpha_excluded_modes": [int(x) for x in request.alpha_excluded_modes],
            "representation": request.representation,
        },
        geometry_stage=geometry_stage,
        harmonic_stage=harmonic_stage,
        cubic_stage=cubic_stage,
        quartic_h22_stage=quartic_h22_stage,
        alpha_parser_stage=alpha_parser_stage,
        alpha_internal_stage=alpha_internal_stage,
        quartic_stage=quartic_stage,
        pending_work=pending_work,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manual-quartic-json",
        help="Run the manual quartic transform route from a JSON file and exit.",
    )
    ap.add_argument(
        "--manual-sextic-json",
        help="Run the manual sextic transform route from a JSON file and exit.",
    )
    ap.add_argument("--max-derivative-order", type=int, choices=RV3_ALLOWED_ORDERS)
    ap.add_argument("--geometry", help="Geometry source path (xyz/log/fchk).")
    ap.add_argument("--geometry-format", default="auto")
    ap.add_argument("--hessian", help="Hessian source path (fchk or Der2).")
    ap.add_argument("--hessian-format", default="auto")
    ap.add_argument("--cubic", help="Cubic source path (log or Der3).")
    ap.add_argument("--cubic-format", default="auto")
    ap.add_argument("--quartic", help="Quartic source path (log or Der4).")
    ap.add_argument("--quartic-format", default="auto")
    ap.add_argument("--alpha-log", help="Optional Gaussian anharmonic log for alpha parsing/benchmarking.")
    ap.add_argument("--alpha-log-format", default="auto")
    ap.add_argument("--alpha-cubic-two-index", help="Optional semi-diagonal cubic 2-index matrix for the internal alpha route.")
    ap.add_argument("--alpha-cubic-two-index-format", default="auto")
    ap.add_argument(
        "--alpha-excluded-modes",
        default="",
        help="Optional 1-based alpha mode exclusion list, e.g. 1,3-5.",
    )
    ap.add_argument("--representation", default="I", choices=("I", "II", "III"))
    ap.add_argument("--integrated-report", action="store_true", help="Emit the integrated vibro-rotational report instead of the short summary.")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of a human summary.")
    return ap


def _format_human_summary(result: RV3Result) -> str:
    lines = [
        "RV3 run summary",
        f"  order: {result.request['max_derivative_order']}",
        f"  geometry: {result.geometry_stage.source_format} -> {result.geometry_stage.point_group} / {result.geometry_stage.rotor_type}",
        f"  rotational constants (MHz): {result.geometry_stage.abc_mhz}",
    ]
    if result.harmonic_stage is not None:
        lines.append(f"  harmonic modes: {result.harmonic_stage.n_modes} in representation {result.harmonic_stage.representation}")
    if result.quartic_h22_stage is not None:
        lines.append(f"  quartic H22 reference (MHz): {result.quartic_h22_stage.projection_mhz}")
    if result.cubic_stage is not None:
        lines.append(f"  cubic source aligned with mapping: {result.cubic_stage.mode_reorder_map}")
    if result.alpha_parser_stage is not None:
        lines.append(f"  alpha parser total (MHz): {result.alpha_parser_stage.total_alpha_mhz}")
    if result.alpha_internal_stage is not None:
        lines.append(f"  alpha internal total (MHz): {result.alpha_internal_stage.alpha_total_sum_mhz}")
    if result.quartic_stage is not None:
        lines.append(f"  quartic stage: {result.quartic_stage.linear_branch_status}")
        if result.quartic_stage.linear_compactness_audit is not None:
            ratio = result.quartic_stage.linear_compactness_audit["pair_to_mode_ratio"]
            lines.append(f"  linear compactness ratio: {ratio}")
    return "\n".join(lines)


def build_rv3_integrated_report(result: RV3Result) -> str:
    lines = [
        "Integrated Vibro-Rotational Analysis",
        f"geometry={result.geometry_stage.source_path}",
        f"point_group={result.geometry_stage.point_group}; rotor_type={result.geometry_stage.rotor_type}; sigma={result.geometry_stage.rotational_symmetry_number}",
        f"ABC_MHz=({result.geometry_stage.abc_mhz[0]}, {result.geometry_stage.abc_mhz[1]}, {result.geometry_stage.abc_mhz[2]})",
    ]
    if result.harmonic_stage is not None:
        lines.extend(
            [
                "",
                "[Standard Quartics]",
                f"representation={result.harmonic_stage.representation}",
                f"n_modes={result.harmonic_stage.n_modes}",
                "watson_S_mhz="
                + ", ".join(f"{k}={v}" for k, v in result.harmonic_stage.watson_s_mhz.items()),
            ]
        )
    if result.quartic_h22_stage is not None:
        lines.extend(
            [
                "",
                "[Validated Quartic H22]",
                f"fchk={result.quartic_h22_stage.fchk_path}",
                f"log={result.quartic_h22_stage.log_path}",
                f"projection_mhz={_format_compact(result.quartic_h22_stage.projection_mhz)}",
            ]
        )
    if result.alpha_parser_stage is not None:
        lines.extend(
            [
                "",
                "[Gaussian Alpha Parser]",
                f"log={result.alpha_parser_stage.source_path}",
                f"excluded_modes={result.alpha_parser_stage.excluded_modes or '(none)'}",
                f"total_alpha_mhz={_format_compact(result.alpha_parser_stage.total_alpha_mhz)}",
                f"kept_alpha_mhz={_format_compact(result.alpha_parser_stage.kept_alpha_mhz)}",
                f"removed_alpha_mhz={_format_compact(result.alpha_parser_stage.removed_alpha_mhz)}",
            ]
        )
    if result.alpha_internal_stage is not None:
        lines.extend(
            [
                "",
                "[Alpha From Harmonic + Cubic]",
                f"source={result.alpha_internal_stage.source_path}",
                f"cubic_origin={result.alpha_internal_stage.cubic_matrix_origin}",
                f"excluded_modes={result.alpha_internal_stage.excluded_modes or '(none)'}",
                f"alpha_total_sum_mhz={_format_compact(result.alpha_internal_stage.alpha_total_sum_mhz)}",
                f"alpha_component_sums_mhz={_format_compact(result.alpha_internal_stage.alpha_component_sums_mhz)}",
            ]
        )
        if result.alpha_internal_stage.special_limit_summary_mhz is not None:
            lines.append(f"special_limit_summary_mhz={_format_compact(result.alpha_internal_stage.special_limit_summary_mhz)}")
        if result.alpha_internal_stage.benchmark_total_sum_mhz is not None:
            lines.append(f"gaussian_alpha_benchmark_mhz={_format_compact(result.alpha_internal_stage.benchmark_total_sum_mhz)}")
            lines.append(f"gaussian_alpha_difference_mhz={_format_compact(result.alpha_internal_stage.benchmark_difference_mhz)}")
    if result.cubic_stage is not None:
        lines.extend(
            [
                "",
                "[Sextic H22 Linear Diagnostic]",
                f"source={result.cubic_stage.source_path}",
                f"mode_reorder_map={result.cubic_stage.mode_reorder_map}",
                f"h22_linear_candidate_hz={_format_compact(result.cubic_stage.sextic_h22_linear_candidate_hz)}",
                "",
                "[Harmonic/Cubic Sextic Hierarchy]",
                f"hierarchy_hz={_format_compact(result.cubic_stage.sextic_cubic_hierarchy_hz)}",
            ]
        )
    if result.quartic_stage is not None:
        lines.extend(
            [
                "",
                "[Linear Order-4]",
                f"status={result.quartic_stage.linear_branch_status}",
            ]
        )
        if result.quartic_stage.linear_general_observable is not None:
            lines.append(f"general_observable={_format_compact(result.quartic_stage.linear_general_observable)}")
        if result.quartic_stage.linear_legacy_compact_observable is not None:
            lines.append(f"legacy_compact={_format_compact(result.quartic_stage.linear_legacy_compact_observable)}")
        if result.quartic_stage.linear_compactness_audit is not None:
            lines.append(f"compactness_audit={_format_compact(result.quartic_stage.linear_compactness_audit)}")
        if result.quartic_stage.linear_optical_constant is not None:
            lines.append(f"optical_constant={_format_compact(result.quartic_stage.linear_optical_constant)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = _build_arg_parser()
    ns = ap.parse_args(argv)
    if ns.manual_quartic_json and ns.manual_sextic_json:
        raise ValueError("Choose only one of --manual-quartic-json or --manual-sextic-json.")
    if ns.manual_quartic_json:
        payload = json.loads(Path(ns.manual_quartic_json).read_text())
        result = run_manual_quartic_transform(RV3ManualQuarticRequest(**payload))
        print(json.dumps(result.to_jsonable(), indent=2, sort_keys=True))
        return 0
    if ns.manual_sextic_json:
        payload = json.loads(Path(ns.manual_sextic_json).read_text())
        result = run_manual_sextic_transform(RV3ManualSexticRequest(**payload))
        print(json.dumps(result.to_jsonable(), indent=2, sort_keys=True))
        return 0
    if ns.max_derivative_order is None:
        raise ValueError("--max-derivative-order is required unless a manual transform JSON route is selected.")
    request = RV3Request(
        max_derivative_order=ns.max_derivative_order,
        geometry=None if ns.geometry is None else RV3Source(ns.geometry, ns.geometry_format),
        hessian=None if ns.hessian is None else RV3Source(ns.hessian, ns.hessian_format),
        cubic=None if ns.cubic is None else RV3Source(ns.cubic, ns.cubic_format),
        quartic=None if ns.quartic is None else RV3Source(ns.quartic, ns.quartic_format),
        alpha_log=None if ns.alpha_log is None else RV3Source(ns.alpha_log, ns.alpha_log_format),
        alpha_cubic_two_index=(
            None
            if ns.alpha_cubic_two_index is None
            else RV3Source(ns.alpha_cubic_two_index, ns.alpha_cubic_two_index_format)
        ),
        alpha_excluded_modes=tuple(sorted(_parse_mode_selection(ns.alpha_excluded_modes))),
        representation=ns.representation,
    )
    result = run_rv3(request)
    if ns.json:
        print(json.dumps(result.to_jsonable(), indent=2, sort_keys=True))
    elif ns.integrated_report:
        print(build_rv3_integrated_report(result))
    else:
        print(_format_human_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
