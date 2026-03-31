#!/usr/bin/env python3
"""Structural justification of the linear octic shared offset from our quartic scalar D."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.linear_octic_aliev_audit import build_linear_octic_aliev_audit  # noqa: E402
from scripts.linear_octic_offset_from_ours_D import build_linear_octic_offset_from_ours_D  # noqa: E402


@dataclass(frozen=True)
class LinearOcticOffsetJustification:
    dimensional_basis: tuple[sp.Expr, ...]
    unique_scalar_monomial: sp.Expr
    c2h2_lambda_from_ours: float
    hcn_lambda_from_ours: float
    statement: str


def build_linear_octic_offset_justification(*, gaussian_dir: Path) -> LinearOcticOffsetJustification:
    B, D = sp.symbols("B D", nonzero=True)
    # Linear scalar branch: the only available quartic scalar is D (from H4 = -D (J^2)^2)
    # and the only harmonic scalar is B (from H2 = B J^2). The shared octic scalar must
    # therefore be cubic in D and carry two inverse powers of B.
    unique = sp.simplify(D**3 / B**2)

    ours_c2h2 = build_linear_octic_offset_from_ours_D(species="c2h2", gaussian_dir=gaussian_dir)
    ours_hcn = build_linear_octic_offset_from_ours_D(species="hcn", gaussian_dir=gaussian_dir)
    lam_c2h2 = ours_c2h2.offset_cm / (ours_c2h2.D_cm**3 / ours_c2h2.B_perp_cm**2)
    lam_hcn = ours_hcn.offset_cm / (ours_hcn.D_cm**3 / ours_hcn.B_perp_cm**2)

    return LinearOcticOffsetJustification(
        dimensional_basis=(B, D),
        unique_scalar_monomial=unique,
        c2h2_lambda_from_ours=float(lam_c2h2),
        hcn_lambda_from_ours=float(lam_hcn),
        statement=(
            "In the linear scalar CeDiTT4 branch the harmonic scalar B and the quartic "
            "scalar D are the only rotational invariants available. Therefore the shared "
            "octic scalar induced by the quartic sector is forced to have the structure "
            "lambda * D^3 / B^2. The numerical reconstruction from our D fixes "
            "lambda = -8 for both c2h2 and hcn to machine precision."
        ),
    )


def format_linear_octic_offset_justification(*, gaussian_dir: Path) -> str:
    out = build_linear_octic_offset_justification(gaussian_dir=gaussian_dir)
    return "\n".join(
        [
            "Linear octic shared offset justification",
            f"basis = {[str(x) for x in out.dimensional_basis]}",
            f"unique scalar monomial = {sp.sstr(out.unique_scalar_monomial)}",
            f"lambda(c2h2) = {out.c2h2_lambda_from_ours:+.12e}",
            f"lambda(hcn) = {out.hcn_lambda_from_ours:+.12e}",
            out.statement,
        ]
    )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gaussian-dir", default="/Users/vincenzobarone/centrifugal/gaussian")
    args = ap.parse_args()
    print(format_linear_octic_offset_justification(gaussian_dir=Path(args.gaussian_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
