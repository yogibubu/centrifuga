#!/usr/bin/env python3
"""Audit mode-resolved alpha channels on the local Gaussian benchmark set.

This script is intentionally diagnostic. It compares the baseline VPT2-like
alpha model against the current quasiparticle-filtered surrogate driven by the
local Gaussian DVPT2/GVPT2 data.

Interpretation used in the printed summaries:

- ``Coriolis`` and ``Inertia`` depend only on the harmonic model;
- ``Anharm`` is the cubic-force contribution and therefore requires the
  anharmonic force field, including both diagonal and semi-diagonal
  ``phi_iii`` / ``phi_iij`` ingredients.

Current default benchmark policy:
- exclude PH3, because the local ``ph3.log`` is not a stable minimum
  (imaginary harmonic frequencies).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_vpt_parser import parse_gaussian_anharmonic_force_data, parse_gaussian_fchk_harmonic_data
from harmonic_convention import align_cubic_to_harmonic_model, build_default_harmonic_model
from rovib_distortion import ANGSTROM_TO_BOHR
from vibrot_alpha import alpha_matrix_from_harmonic_and_cubic_cm, alpha_matrix_from_mixed_gaussian_sources


DEFAULT_SPECIES = (
    "h2o",
    "h2s",
    "hfo",
    "h2co",
    "h2cs",
    "nh3",
    "f2co",
    "bf3",
    "nf3",
    "c2h2",
    "hccd",
)

AXIS_LABELS = ("a", "b", "c")


def _load_alpha_case(
    harmonic_base: Path,
    anharmonic_base: Path,
    analysis_base: Path,
    species: str,
) -> dict[str, object]:
    fchk_path = harmonic_base / f"{species}.fchk"
    log_path = anharmonic_base / f"{species}.log"
    analysis_log_path = analysis_base / f"{species}.log"
    fchk = parse_gaussian_fchk_harmonic_data(fchk_path)
    anh = parse_gaussian_anharmonic_force_data(log_path)
    model, _meta = build_default_harmonic_model(
        fchk.masses_amu,
        fchk.coords_bohr * (1.0 / ANGSTROM_TO_BOHR),
        fchk.cartesian_force_constants,
    )
    cubic = align_cubic_to_harmonic_model(anh, model.vib_freq_cm)
    alpha = alpha_matrix_from_harmonic_and_cubic_cm(model, cubic.reduced_cm)
    qp = alpha_matrix_from_mixed_gaussian_sources(
        fchk_path,
        log_path,
        analysis_log_path=analysis_log_path,
    )
    return {
        "model": model,
        "alpha": alpha,
        "qp": qp["alpha"],
        "report": qp["report"],
        "source_paths": {
            "harmonic_fchk": fchk_path,
            "anharmonic_log": log_path,
            "analysis_log": analysis_log_path,
        },
    }


def _load_alpha_case_from_paths(
    harmonic_fchk_path: Path,
    anharmonic_log_path: Path,
    analysis_log_path: Path | None,
    label: str,
) -> dict[str, object]:
    fchk = parse_gaussian_fchk_harmonic_data(harmonic_fchk_path)
    anh = parse_gaussian_anharmonic_force_data(anharmonic_log_path)
    model, _meta = build_default_harmonic_model(
        fchk.masses_amu,
        fchk.coords_bohr * (1.0 / ANGSTROM_TO_BOHR),
        fchk.cartesian_force_constants,
    )
    cubic = align_cubic_to_harmonic_model(anh, model.vib_freq_cm)
    alpha = alpha_matrix_from_harmonic_and_cubic_cm(model, cubic.reduced_cm)
    qp = alpha_matrix_from_mixed_gaussian_sources(
        harmonic_fchk_path,
        anharmonic_log_path,
        analysis_log_path=analysis_log_path,
    )
    return {
        "label": label,
        "model": model,
        "alpha": alpha,
        "qp": qp["alpha"],
        "report": qp.get("report"),
        "source_paths": {
            "harmonic_fchk": harmonic_fchk_path,
            "anharmonic_log": anharmonic_log_path,
            "analysis_log": analysis_log_path,
        },
    }


def _classify_mode(total_row: np.ndarray, cor_row: np.ndarray, inertia_row: np.ndarray, anh_row: np.ndarray) -> tuple[str, dict[str, float]]:
    comps = {
        "Cor": float(np.max(np.abs(cor_row))),
        "In": float(np.max(np.abs(inertia_row))),
        "Anh": float(np.max(np.abs(anh_row))),
    }
    dominant = max(comps, key=comps.get)
    second = sorted(comps.values(), reverse=True)[1]
    if comps[dominant] <= 1.0e-30:
        return "zero", comps
    if second <= 0.6 * comps[dominant]:
        return dominant, comps
    return f"mixed->{dominant}", comps


def _fmt_triplet(vals: np.ndarray) -> str:
    arr = np.asarray(vals, dtype=float)
    return "(" + ", ".join(f"{x:+8.4f}" for x in arr) + ")"


def _print_delta_vib_summary(
    total_abc: np.ndarray,
    cor_abc: np.ndarray,
    inertia_abc: np.ndarray,
    anh_abc: np.ndarray,
    anh_diag_abc: np.ndarray,
    anh_semidiag_abc: np.ndarray,
    top: int,
) -> None:
    delta_total = 0.5 * np.sum(total_abc, axis=0)
    delta_cor = 0.5 * np.sum(cor_abc, axis=0)
    delta_inertia = 0.5 * np.sum(inertia_abc, axis=0)
    delta_anh = 0.5 * np.sum(anh_abc, axis=0)
    delta_anh_diag = 0.5 * np.sum(anh_diag_abc, axis=0)
    delta_anh_semidiag = 0.5 * np.sum(anh_semidiag_abc, axis=0)
    print("Delta_vib summary (same units as alpha output):")
    print(f"  total    abc={_fmt_triplet(delta_total)}")
    print(f"  coriolis abc={_fmt_triplet(delta_cor)}   [harmonic-only input]")
    print(f"  inertia  abc={_fmt_triplet(delta_inertia)}   [harmonic-only input]")
    print(f"  anharm   abc={_fmt_triplet(delta_anh)}   [requires cubic force constants]")
    print(f"  anharm(diag phi_iii) abc={_fmt_triplet(delta_anh_diag)}")
    print(f"  anharm(semi-diag phi_iij) abc={_fmt_triplet(delta_anh_semidiag)}")
    for axis_idx, axis in enumerate(AXIS_LABELS):
        ranked = sorted(
            ((mode + 1, 0.5 * float(total_abc[mode, axis_idx])) for mode in range(total_abc.shape[0])),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        top_terms = ", ".join(f"m{mode}={value:+.4f}" for mode, value in ranked[: min(top, len(ranked))])
        print(f"  dominant Delta_vib({axis}) terms: {top_terms}")


def run_audit(harmonic_base: Path, anharmonic_base: Path, analysis_base: Path, species_list: list[str], top: int) -> int:
    for species in species_list:
        case = _load_alpha_case(harmonic_base, anharmonic_base, analysis_base, species)
        case["label"] = species
        _print_case(case, top)
    return 0


def _print_case(case: dict[str, object], top: int) -> None:
        case_label = str(case.get("label", "case"))
        model = case["model"]
        alpha = case["alpha"]
        alpha_qp = case["qp"]
        report = case["report"]
        source_paths = case["source_paths"]
        total = np.asarray(alpha["alpha_total_cm_abc"], dtype=float)
        cor = np.asarray(alpha["alpha_coriolis_cm_abc"], dtype=float)
        inertia = np.asarray(alpha["alpha_inertia_cm_abc"], dtype=float)
        anh = np.asarray(alpha["alpha_anharmonic_cm_abc"], dtype=float)
        anh_diag = np.asarray(alpha["alpha_anharmonic_diagonal_cm_abc"], dtype=float)
        anh_semidiag = np.asarray(alpha["alpha_anharmonic_semidiagonal_cm_abc"], dtype=float)
        total_qp = np.asarray(alpha_qp["alpha_total_cm_abc"], dtype=float)
        rotor = (alpha.get("rotor_limit") or {}).get("kind", "unknown")

        rows: list[tuple[int, float, float, str, dict[str, float]]] = []
        counts: dict[str, int] = {}
        for idx in range(total.shape[0]):
            score = float(np.max(np.abs(total[idx])))
            mode_label, comps = _classify_mode(total[idx], cor[idx], inertia[idx], anh[idx])
            counts[mode_label] = counts.get(mode_label, 0) + 1
            rows.append((idx + 1, float(model.vib_freq_cm[idx]), score, mode_label, comps))

        rows.sort(key=lambda item: item[2], reverse=True)
        print(f"=== {case_label} ({rotor}) ===")
        plan = alpha_qp.get("anharmonic_filter_plan")
        if plan is not None:
            print(
                "qp_filter:",
                f"class2={plan.class2_modes or ()}",
                f"class3={plan.class3_modes or ()}",
                f"disabled_pairs={plan.disable_semidiagonal_pairs or ()}",
            )
        print(
            "sources:",
            f"harmonic_fchk={source_paths['harmonic_fchk']}",
            f"anharmonic_log={source_paths['anharmonic_log']}",
            f"analysis_log={source_paths['analysis_log']}",
        )
        print("input structure: Coriolis/Inertia = harmonic-only; Anharm = cubic-force contribution (phi_iii, phi_iij)")
        _print_delta_vib_summary(total, cor, inertia, anh, anh_diag, anh_semidiag, top)
        print("class_counts:", ", ".join(f"{key}={counts[key]}" for key in sorted(counts)))
        for mode, freq, score, label, comps in rows[: min(top, len(rows))]:
            qp_score = float(np.max(np.abs(total_qp[mode - 1])))
            delta = qp_score - score
            assigned = "n/a" if report is None else report.diagnostics[mode - 1].assigned_class
            print(
                f"mode={mode:>2d}  w={freq:>9.2f} cm^-1  "
                f"max|alpha|={score:>9.4f}  qp={qp_score:>9.4f}  dqp={delta:>+9.4f}  "
                f"class={label:<10s}  three-class={assigned:<3s}  "
                f"Cor={comps['Cor']:>8.4f}  In={comps['In']:>8.4f}  Anh={comps['Anh']:>8.4f}"
            )
            print(f"  alpha_total_abc    = {_fmt_triplet(total[mode - 1])}")
            print(f"  alpha_coriolis_abc = {_fmt_triplet(cor[mode - 1])}   [harmonic-only]")
            print(f"  alpha_inertia_abc  = {_fmt_triplet(inertia[mode - 1])}   [harmonic-only]")
            print(f"  alpha_anharm_abc   = {_fmt_triplet(anh[mode - 1])}   [from cubic phi_iii / phi_iij]")
            print(f"  alpha_anh_diag_abc = {_fmt_triplet(anh_diag[mode - 1])}   [diagonal anharmonicity via phi_iii]")
            print(f"  alpha_anh_sd_abc   = {_fmt_triplet(anh_semidiag[mode - 1])}   [mode coupling via phi_iij]")
            print(f"  1/2 alpha -> Delta_vib abc = {_fmt_triplet(0.5 * total[mode - 1])}")
        print()


def run_single_case(
    harmonic_fchk_path: Path,
    anharmonic_log_path: Path,
    analysis_log_path: Path | None,
    top: int,
    label: str | None = None,
) -> int:
    case = _load_alpha_case_from_paths(
        harmonic_fchk_path,
        anharmonic_log_path,
        analysis_log_path,
        label or harmonic_fchk_path.stem,
    )
    _print_case(case, top)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gaussian-dir",
        default=str(Path(__file__).resolve().parents[1] / "gaussian"),
        help="Default directory for harmonic, anharmonic, and analysis Gaussian files.",
    )
    ap.add_argument(
        "--harmonic-gaussian-dir",
        default=None,
        help="Directory providing the harmonic .fchk files for Coriolis/Inertia and the harmonic model.",
    )
    ap.add_argument(
        "--anharmonic-gaussian-dir",
        default=None,
        help="Directory providing the anharmonic .log files used to read cubic force constants.",
    )
    ap.add_argument(
        "--analysis-gaussian-dir",
        default=None,
        help="Directory providing the DVPT2/GVPT2 analysis .log files used for the quasiparticle filter.",
    )
    ap.add_argument("--harmonic-fchk", default=None, help="Explicit harmonic .fchk path for single-case mode.")
    ap.add_argument("--anharmonic-log", default=None, help="Explicit anharmonic .log path for single-case mode.")
    ap.add_argument(
        "--analysis-log",
        default=None,
        help="Explicit DVPT2/GVPT2 analysis .log path for single-case mode. Defaults to --anharmonic-log.",
    )
    ap.add_argument("--label", default=None, help="Optional label printed for single-case mode.")
    ap.add_argument(
        "--species",
        nargs="+",
        default=list(DEFAULT_SPECIES),
        help="Species tags without extension. Default excludes PH3 until a valid minimum is regenerated.",
    )
    ap.add_argument("--top", type=int, default=5, help="How many highest-|alpha| modes to print per species.")
    args = ap.parse_args()
    if bool(args.harmonic_fchk) != bool(args.anharmonic_log):
        raise SystemExit("Provide both --harmonic-fchk and --anharmonic-log for single-case mode.")
    if args.harmonic_fchk and args.anharmonic_log:
        analysis_log = Path(args.analysis_log) if args.analysis_log else Path(args.anharmonic_log)
        return run_single_case(
            Path(args.harmonic_fchk),
            Path(args.anharmonic_log),
            analysis_log,
            int(args.top),
            label=args.label,
        )
    default_base = Path(args.gaussian_dir)
    harmonic_base = Path(args.harmonic_gaussian_dir) if args.harmonic_gaussian_dir else default_base
    anharmonic_base = Path(args.anharmonic_gaussian_dir) if args.anharmonic_gaussian_dir else default_base
    analysis_base = Path(args.analysis_gaussian_dir) if args.analysis_gaussian_dir else anharmonic_base
    return run_audit(harmonic_base, anharmonic_base, analysis_base, list(args.species), int(args.top))


if __name__ == "__main__":
    raise SystemExit(main())
