#!/usr/bin/env python3
"""General driver for the linear-molecule Aliev `D_v` / `L` model.

The input is a JSON file with the explicit linear-molecule data:

- `B`
- `D_J`
- `omega_parallel`
- `omega_perpendicular`
- `coriolis_nt` (or legacy alias `zeta_nt`)
- `bxx_parallel`
- `k3_parallel`
- `k3_perp_pair`

and optionally either

- `k4_parallel` for the full four-index quartic field, or
- `k4_reduced` for the reduced three-index field interpreted as `k_(ii jk)`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linear_dv_aliev_terms import (
    build_explicit_aliev_L_model,
    build_explicit_aliev_betas,
    build_explicit_aliev_dv_compact_model_if_safe,
    build_explicit_aliev_dv_general_model,
    build_explicit_aliev_dv_legacy_compact_model,
    make_linear_aliev_explicit_inputs_from_mapping,
    resolve_linear_aliev_quartic_mode,
)


def _format_quanta(text: str, n_modes: int) -> tuple[float, ...]:
    values = tuple(float(chunk.strip()) for chunk in text.split(",") if chunk.strip())
    if len(values) != n_modes:
        raise ValueError(f"Expected {n_modes} quanta, got {len(values)}.")
    return values


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_json", help="JSON file with explicit linear-Aliev inputs.")
    ap.add_argument(
        "--quartic-mode",
        choices=("auto", "full", "general_4index", "reduced", "reduced_input", "none"),
        default="auto",
        help="How to interpret quartic input: general 4-index, reduced-input 3-index, none, or auto-detect.",
    )
    ap.add_argument(
        "--observable-branch",
        choices=("general", "legacy_compact", "compact_if_safe"),
        default="general",
        help="Observable branch: general decompacted, legacy compact, or compact only if the pair sector is negligible.",
    )
    ap.add_argument(
        "--compactness-tolerance",
        type=float,
        default=1.0,
        help="Used only with --observable-branch compact_if_safe.",
    )
    ap.add_argument(
        "--state",
        help="Comma-separated vibrational quanta in mode order: all parallel modes, then perpendicular modes.",
    )
    args = ap.parse_args()

    payload = json.loads(Path(args.input_json).read_text())
    mode = resolve_linear_aliev_quartic_mode(
        k4_parallel=payload.get("k4_parallel"),
        k4_reduced=payload.get("k4_reduced"),
        quartic_mode=args.quartic_mode,
    )
    inputs = make_linear_aliev_explicit_inputs_from_mapping(payload, quartic_mode=mode)
    betas = build_explicit_aliev_betas(inputs)
    if args.observable_branch == "general":
        dv_model = build_explicit_aliev_dv_general_model(inputs)
    elif args.observable_branch == "legacy_compact":
        dv_model = build_explicit_aliev_dv_legacy_compact_model(inputs)
    else:
        dv_model = build_explicit_aliev_dv_compact_model_if_safe(inputs, tolerance=args.compactness_tolerance)

    print(f"quartic_mode = {mode}")
    print(f"observable_branch = {args.observable_branch}")
    print(f"n_parallel = {len(inputs.omega_parallel)}")
    print(f"n_perpendicular = {len(inputs.omega_perpendicular)}")
    print()
    print("[beta_parallel]")
    for idx in range(len(inputs.omega_parallel)):
        print(f"beta_n[{idx}] = {betas.beta_parallel[idx]}")
    print()
    print("[beta_perpendicular]")
    for idx in range(len(inputs.omega_perpendicular)):
        print(f"beta_t[{idx}] = {betas.beta_perpendicular[idx]}")
    print()
    if args.state:
        quanta = _format_quanta(args.state, len(dv_model.mode_kinds))
        print(f"D_v(state={quanta}) = {dv_model.value_for_state(quanta)}")
    else:
        print("D_v(state) available via --state q0,q1,...")
    try:
        l_model = build_explicit_aliev_L_model(inputs)
    except ValueError:
        print("L = unavailable (quartic input absent)")
    else:
        print(f"L = {l_model.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
