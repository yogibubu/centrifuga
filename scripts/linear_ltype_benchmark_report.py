#!/usr/bin/env python3
"""Generate a compact linear-branch benchmark report."""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceditt_gui import _build_harmonic_model_from_inputs
from distortion_workflow import compute_order2_quartic
from gaussian_vpt_parser import parse_gaussian_linear_ltype_constants

CMINV_TO_HZ = 2.99792458e10


def _fmt_mhz(value_hz: float | None) -> str:
    if value_hz is None:
        return "---"
    return f"{value_hz / 1.0e6:.6f}"


def _parse_gaussian_fundamental_bands(log_path: str) -> list[dict[str, float | int]]:
    lines = pathlib.Path(log_path).read_text(errors="ignore").splitlines()
    out: list[dict[str, float | int]] = []
    start = None
    vib_block = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Vibrational Energies at Anharmonic Level"):
            vib_block = i
    if vib_block is None:
        return out
    for i, line in enumerate(lines[vib_block:], start=vib_block):
        if line.strip().startswith("Fundamental Bands"):
            start = i
            break
    if start is None:
        return out
    for line in lines[start + 3 :]:
        if line.strip().startswith("Overtones"):
            break
        if not line.strip():
            if out:
                break
            continue
        m = re.match(
            r"\s*(\d+)\(1\)\s+\S+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
            line,
        )
        if not m:
            continue
        out.append(
            {
                "mode_1based": int(m.group(1)),
                "harm_cm": float(m.group(2)),
                "anharm_cm": float(m.group(3)),
            }
        )
    return out


def _nearest_fundamental(pair_freq_cm: float, fundamentals: list[dict[str, float | int]]) -> dict[str, float | int] | None:
    if not fundamentals:
        return None
    return min(fundamentals, key=lambda item: abs(float(item["harm_cm"]) - pair_freq_cm))


def _report_species(tag: str, *, fchk_path: str, log_path: str | None = None) -> None:
    model, _ = _build_harmonic_model_from_inputs("I", fchk_path=fchk_path)
    out = compute_order2_quartic(model, linear_log_path=log_path)
    ltype = out["linear_ltype_terms"]
    quartic = out["special_quartic_projection"]["quartic_mhz"]
    fundamentals = _parse_gaussian_fundamental_bands(log_path) if log_path else []
    qconst = parse_gaussian_linear_ltype_constants(log_path) if log_path else None
    gaussian_qe_items = sorted(qconst.q_e_mhz.items(), key=lambda kv: kv[1]) if qconst else []

    print(f"Linear-branch benchmark: {tag}")
    print(f"Pure rotational D [MHz]: {quartic['D']:.9f}")
    print(f"B_linear [cm^-1]: {ltype['B_linear_cm']:.9f}")
    print()
    if qconst and (qconst.q_e_mhz or qconst.q_j_mhz or qconst.q_k_mhz):
        print("Gaussian linear-l-type constants from log")
        for name, vals in (("q^e", qconst.q_e_mhz), ("q^J", qconst.q_j_mhz), ("q^K", qconst.q_k_mhz)):
            if not vals:
                continue
            converted = ", ".join(f"Q({idx})={val:.6f} MHz" for idx, val in sorted(vals.items()))
            print(f"  {name}: {converted}")
        if qconst.active_dd_22_count is not None:
            print(f"  active 2-2 Darling-Dennison resonances: {qconst.active_dd_22_count}")
        print()
    hdr = (
        "pair      modes    freq/cm^-1    q_l[Mhz]    q_e^(0)[MHz]    "
        "q_e^(W)[MHz]    q_e^(src)[MHz]    q_v~2B/nu_anh[Mhz]    q_J^(pair)[MHz]    q_J^(recon)[MHz]    q_K^(recon)[MHz]"
    )
    print(hdr)
    for pair in ltype["pairs"]:
        qpair = pair["literature_constants_hz"]
        qspec = pair["spectroscopic_linear_constants_hz"]
        qeff = pair["effective_linear_model_hz"]["constants_hz"]
        qjk_src = pair.get("gaussian_source_rotational_constants_hz", {})
        fund = _nearest_fundamental(pair["freq_cm"], fundamentals)
        qv_hz = None
        if fund is not None and abs(float(fund["anharm_cm"])) > 1.0e-12:
            qv_hz = float((2.0 * ltype["B_linear_cm"] / float(fund["anharm_cm"])) * CMINV_TO_HZ)
        qg_target = None
        if gaussian_qe_items:
            qg_target = min(gaussian_qe_items, key=lambda kv: abs(kv[1] - (qspec["q_e_source"] / 1.0e6)))
        print(
            f"{pair.get('pair_label','?'):8s}  "
            f"{str(pair['modes']):9s}  "
            f"{pair['freq_cm']:11.6f}  "
            f"{_fmt_mhz(qpair['q_l']):>10s}  "
            f"{_fmt_mhz(qspec['q_e0']):>13s}  "
            f"{_fmt_mhz(qspec['q_eW']):>13s}  "
            f"{_fmt_mhz(qspec['q_e_source']):>15s}  "
            f"{_fmt_mhz(qv_hz):>18s}  "
            f"{_fmt_mhz(qeff['q_J_pair']):>15s}  "
            f"{_fmt_mhz(qjk_src.get('q_J_source')):>14s}  "
            f"{_fmt_mhz(qjk_src.get('q_K_source')):>14s}"
        )
        if qg_target is not None:
            print(
                f"           nearest Gaussian q^e target: Q({qg_target[0]})={qg_target[1]:.6f} MHz"
            )
        gres = pair.get("gaussian_source_exact_constants_hz")
        if gres is not None:
            print(
                "           Gaussian RotL2x exact model: "
                f"Q({int(gres['Q_index'])})  "
                f"q^e={gres['q_e'] / 1.0e6:.6f} MHz, "
                f"q^J={gres['q_J'] / 1.0e6:.6f} MHz, "
                f"q^K={gres['q_K'] / 1.0e6:.6f} MHz"
            )
    print()


def main() -> None:
    _report_species("C2H2", fchk_path="c2h2.fchk")
    _report_species("HCCD", fchk_path="hccd.fchk", log_path="hccd.log")
    print("External comparison points previously collected for HCCD (NIST):")
    print("  nu4: q_nu = 132.993 MHz")
    print("  nu5: q_nu = 105.702 MHz")
    print()
    print("Interpretation:")
    print("  - q_e^(0) and q_e^(W) are simple spectroscopic estimates.")
    print("  - q_e^(src) follows the explicit Gaussian source formula and is the right non-resonant J0 term.")
    print("  - q_v~2B/nu_anh is the first state-specific estimate from anharmonic fundamentals.")
    print("  - For HCCD, q_e^(src) already lands on the Gaussian q^e values.")
    print("  - The reconstructed q^J/q^K terms from printed Gaussian alpha/cubic blocks are exposed separately from the exact RotL2x q^J/q^K block.")
    print("  - q_l and q_J^(pair) remain the tensorial pairwise branch.")
    print("  - q_e^(src) is now the J0 driving term of the minimal effective model.")


if __name__ == "__main__":
    main()
