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
from distortion_workflow import compute_order2_quartic
from gaussian_vpt_parser import (
    align_gaussian_cubic_force_constants,
    frequency_reorder_map,
    parse_gaussian_anharmonic_force_data,
    parse_gaussian_fchk_harmonic_data,
    parse_gaussian_harmonic_data,
)
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
    HarmonicInertiaModel,
    Molecule,
    harmonic_inertia_model_from_geometry_hessian,
    inertia_tensor,
    read_xyz,
    rotational_constants,
)
from scripts.build_linear_aliev_payload_from_gaussian import build_payload as build_linear_aliev_payload
from symmetry_metadata import assign_normal_mode_irreps, point_group_from_geometry, rotor_type_for_symmetry, symbols_from_atomic_numbers


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
    quartic_stage: RV3QuarticStage | None
    pending_work: list[str]

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
        if request.max_derivative_order >= 3:
            assert request.cubic is not None
            cubic_stage = _cubic_stage_from_source(model, request.cubic)
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
            "representation": request.representation,
        },
        geometry_stage=geometry_stage,
        harmonic_stage=harmonic_stage,
        cubic_stage=cubic_stage,
        quartic_stage=quartic_stage,
        pending_work=pending_work,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-derivative-order", type=int, required=True, choices=RV3_ALLOWED_ORDERS)
    ap.add_argument("--geometry", help="Geometry source path (xyz/log/fchk).")
    ap.add_argument("--geometry-format", default="auto")
    ap.add_argument("--hessian", help="Hessian source path (fchk or Der2).")
    ap.add_argument("--hessian-format", default="auto")
    ap.add_argument("--cubic", help="Cubic source path (log or Der3).")
    ap.add_argument("--cubic-format", default="auto")
    ap.add_argument("--quartic", help="Quartic source path (log or Der4).")
    ap.add_argument("--quartic-format", default="auto")
    ap.add_argument("--representation", default="I", choices=("I", "II", "III"))
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
    if result.cubic_stage is not None:
        lines.append(f"  cubic source aligned with mapping: {result.cubic_stage.mode_reorder_map}")
    if result.quartic_stage is not None:
        lines.append(f"  quartic stage: {result.quartic_stage.linear_branch_status}")
        if result.quartic_stage.linear_compactness_audit is not None:
            ratio = result.quartic_stage.linear_compactness_audit["pair_to_mode_ratio"]
            lines.append(f"  linear compactness ratio: {ratio}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = _build_arg_parser()
    ns = ap.parse_args(argv)
    request = RV3Request(
        max_derivative_order=ns.max_derivative_order,
        geometry=None if ns.geometry is None else RV3Source(ns.geometry, ns.geometry_format),
        hessian=None if ns.hessian is None else RV3Source(ns.hessian, ns.hessian_format),
        cubic=None if ns.cubic is None else RV3Source(ns.cubic, ns.cubic_format),
        quartic=None if ns.quartic is None else RV3Source(ns.quartic, ns.quartic_format),
        representation=ns.representation,
    )
    result = run_rv3(request)
    if ns.json:
        print(json.dumps(result.to_jsonable(), indent=2, sort_keys=True))
    else:
        print(_format_human_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
