#!/usr/bin/env python3

from pathlib import Path

import numpy as np

from anharmonic_partition import (
    ThreeClassAnalysis,
    ThreeClassModeDiagnostic,
    classify_three_classes_cartesian,
)
from gaussian_vpt_parser import (
    parse_gaussian_anharmonic_analysis,
    parse_gaussian_anharmonic_force_data,
    parse_gaussian_fchk_harmonic_data,
)
from harmonic_convention import align_cubic_to_harmonic_model, build_default_harmonic_model
from rovib_distortion import ANGSTROM_TO_BOHR
from vibrot_alpha import (
    alpha_matrix_from_mixed_gaussian_sources,
    alpha_matrix_from_gaussian_quasiparticle_model,
    alpha_matrix_from_harmonic_and_cubic_cm,
    alpha_matrix_from_harmonic_and_cubic_with_report,
)


def _build(name: str):
    fchk = parse_gaussian_fchk_harmonic_data(Path(f"{name}.fchk"))
    model, meta = build_default_harmonic_model(
        fchk.masses_amu,
        fchk.coords_bohr * (1.0 / ANGSTROM_TO_BOHR),
        fchk.cartesian_force_constants,
    )
    return model, meta


def test_alpha_effective_frequencies_only_change_anharmonic_channel() -> None:
    model, _meta = _build("h2o")
    anh = parse_gaussian_anharmonic_force_data(Path("h2o.log"))
    cubic = align_cubic_to_harmonic_model(anh, model.vib_freq_cm)

    base = alpha_matrix_from_harmonic_and_cubic_cm(model, cubic.reduced_cm)
    eff = np.abs(np.asarray(model.vib_freq_cm, dtype=float)).copy()
    eff[0] *= 0.9
    shifted = alpha_matrix_from_harmonic_and_cubic_cm(
        model,
        cubic.reduced_cm,
        effective_frequencies_cm=eff,
    )

    assert np.allclose(base["alpha_coriolis_cm"], shifted["alpha_coriolis_cm"])
    assert np.allclose(base["alpha_inertia_cm"], shifted["alpha_inertia_cm"])
    assert not np.allclose(base["alpha_anharmonic_cm"], shifted["alpha_anharmonic_cm"])
    assert np.allclose(
        shifted["alpha_anharmonic_cm"],
        shifted["alpha_anharmonic_diagonal_cm"] + shifted["alpha_anharmonic_semidiagonal_cm"],
    )
    assert np.allclose(shifted["anharmonic_reference_frequencies_cm"], eff)


def test_alpha_report_wrapper_matches_manual_quasiparticle_filter_plan() -> None:
    model, _meta = _build("h2co")
    anh = parse_gaussian_anharmonic_force_data(Path("h2co.log"))
    cubic = align_cubic_to_harmonic_model(anh, model.vib_freq_cm)
    n_modes = len(model.vib_freq_cm)

    diagnostics: list[ThreeClassModeDiagnostic] = []
    for idx in range(n_modes):
        mode = idx + 1
        assigned = "I"
        omega_gvpt2 = None
        reasons = ("perturbative_default",)
        partners: tuple[int, ...] = ()
        if mode == 1:
            assigned = "II"
            omega_gvpt2 = float(abs(model.vib_freq_cm[idx]) * 0.92)
            reasons = ("large_direct_diagonal_self_cubic",)
        elif mode == 2:
            assigned = "III"
            partners = (1,)
            reasons = ("strong_resonant_mixing",)
        diagnostics.append(
            ThreeClassModeDiagnostic(
                mode_index=mode,
                mode_label=f"mode_{mode}",
                assigned_class=assigned,
                y_diag_cm=0.0,
                z_diag_cm=0.0,
                x_diag_cm=0.0,
                self_cubic_shift_cm=0.0,
                quartic_diag_term_cm=0.0,
                direct_fraction=0.0,
                active_resonance_count=0,
                low_overlap=None,
                resonance_partner_modes=partners,
                omega_dvpt2_cm=float(abs(model.vib_freq_cm[idx])),
                omega_gvpt2_cm=omega_gvpt2,
                qp_shift_cm=None if omega_gvpt2 is None else omega_gvpt2 - float(abs(model.vib_freq_cm[idx])),
                reasons=reasons,
            )
        )
    report = ThreeClassAnalysis(
        y_direct_cm=np.zeros((n_modes, n_modes), dtype=float),
        z_indirect_cm=np.zeros((n_modes, n_modes), dtype=float),
        diagnostics=tuple(diagnostics),
        unreliable_semidiagonal_pairs=((1, 2),),
    )

    eff = np.abs(np.asarray(model.vib_freq_cm, dtype=float)).copy()
    eff[0] *= 0.92
    manual = alpha_matrix_from_harmonic_and_cubic_cm(
        model,
        cubic.reduced_cm,
        effective_frequencies_cm=eff,
        disabled_semidiagonal_pairs=((1, 2),),
    )
    wrapped = alpha_matrix_from_harmonic_and_cubic_with_report(model, cubic.reduced_cm, report)

    assert np.allclose(wrapped["alpha_total_cm"], manual["alpha_total_cm"])
    assert np.allclose(wrapped["alpha_anharmonic_cm"], manual["alpha_anharmonic_cm"])
    assert np.allclose(wrapped["anharmonic_reference_frequencies_cm"], eff)
    plan = wrapped["anharmonic_filter_plan"]
    assert plan.class2_modes == (1,)
    assert plan.class3_modes == (2,)
    assert plan.disable_semidiagonal_pairs == ((1, 2),)


def test_alpha_gaussian_quasiparticle_entrypoint_matches_manual_pipeline() -> None:
    model, convention = _build("h2co")
    force = parse_gaussian_anharmonic_force_data(Path("h2co.log"))
    analysis = parse_gaussian_anharmonic_analysis(Path("h2co.log"))
    cubic = align_cubic_to_harmonic_model(force, model.vib_freq_cm)
    report = classify_three_classes_cartesian(force, analysis, class2_max_frequency_cm=4000.0)
    manual = alpha_matrix_from_harmonic_and_cubic_with_report(model, cubic.reduced_cm, report)

    got = alpha_matrix_from_gaussian_quasiparticle_model(
        Path("h2co.fchk"),
        Path("h2co.log"),
        classify_kwargs={"class2_max_frequency_cm": 4000.0},
    )

    assert np.allclose(got["alpha"]["alpha_total_cm"], manual["alpha_total_cm"])
    assert np.allclose(got["alpha"]["alpha_anharmonic_cm"], manual["alpha_anharmonic_cm"])
    assert got["harmonic_convention"] == convention
    assert got["report"].unreliable_semidiagonal_pairs == report.unreliable_semidiagonal_pairs


def test_alpha_mixed_sources_matches_same_source_baseline_when_files_coincide() -> None:
    model, _meta = _build("h2co")
    force = parse_gaussian_anharmonic_force_data(Path("h2co.log"))
    cubic = align_cubic_to_harmonic_model(force, model.vib_freq_cm)
    manual = alpha_matrix_from_harmonic_and_cubic_cm(model, cubic.reduced_cm)

    got = alpha_matrix_from_mixed_gaussian_sources(
        Path("h2co.fchk"),
        Path("h2co.log"),
    )

    assert np.allclose(got["alpha"]["alpha_total_cm"], manual["alpha_total_cm"])
    assert np.allclose(got["alpha"]["alpha_coriolis_cm"], manual["alpha_coriolis_cm"])
    assert np.allclose(got["alpha"]["alpha_inertia_cm"], manual["alpha_inertia_cm"])
    assert np.allclose(got["alpha"]["alpha_anharmonic_cm"], manual["alpha_anharmonic_cm"])
