#!/usr/bin/env python3
"""Inspect the raw 2x2 Coriolis blocks for linear-molecule degenerate pairs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceditt_gui import _build_harmonic_model_from_inputs  # noqa: E402
from compare_gaussian_sextic import _classify_rotor_limit, _degenerate_mode_metadata  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fchk", default="/Users/vincenzobarone/centrifugal/gaussian/c2h2.fchk")
    args = ap.parse_args()

    model, _ = _build_harmonic_model_from_inputs("I", fchk_path=args.fchk)
    rotor = _classify_rotor_limit(np.asarray(model.abc_mhz, dtype=float), np.asarray(model.moments_amu_a2, dtype=float))
    pair_meta = _degenerate_mode_metadata(model, rotor)
    zeta = np.asarray(model.coriolis_zeta_pairs_xyz, dtype=float)
    deg_axes = tuple(rotor.get("degenerate_axes", ()))
    axis_map = {label: idx for idx, label in enumerate(model.xyz_to_abc)}
    deg_axis_indices = [axis_map[str(ax)] for ax in deg_axes]
    deg_modes = {int(x) for meta in pair_meta for x in meta["pair"]}
    parallel = [idx for idx in range(len(model.vib_freq_cm)) if idx not in deg_modes]

    print("=== Zeta Tensor Audit ===")
    print(f"fchk = {args.fchk}")
    print(f"xyz_to_abc = {model.xyz_to_abc}")
    print(f"symmetry_axis = {rotor.get('symmetry_axis')}")
    print(f"degenerate_axes = {deg_axes}")
    print(f"parallel_modes = {parallel}")

    for meta in pair_meta:
        i, j = (int(x) for x in meta["pair"])
        print("")
        print(
            f"pair = {(i, j)}  freq = {meta['freq_cm']:.6f} cm^-1  "
            f"dominant_pair_axis = {meta['dominant_coriolis_axis']}  "
            f"|zeta_pair| = {meta['dominant_coriolis_abs']:.6f}"
        )
        for p in parallel:
            block = np.array(
                [
                    [zeta[deg_axis_indices[0], p, i], zeta[deg_axis_indices[0], p, j]],
                    [zeta[deg_axis_indices[1], p, i], zeta[deg_axis_indices[1], p, j]],
                ],
                dtype=float,
            )
            offdiag = 0.5 * (block[0, 1] + block[1, 0])
            diag = 0.5 * (block[0, 0] + block[1, 1])
            skew = 0.5 * (block[0, 1] - block[1, 0])
            asym_diag = 0.5 * (block[0, 0] - block[1, 1])
            print(f"  parallel mode {p}:")
            print(f"    block = {block.tolist()}")
            print(f"    pair_offdiag = {offdiag:.12e}")
            print(f"    pair_diag    = {diag:.12e}")
            print(f"    offdiag_skew = {skew:.12e}")
            print(f"    diag_asym    = {asym_diag:.12e}")
            print(f"    frob_norm    = {float(np.linalg.norm(block)):.12e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
