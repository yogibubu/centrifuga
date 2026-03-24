#!/usr/bin/env python3
"""Print H22 D and intrinsic-share D_N diagnostics for small benchmarks."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceditt_gui import _h22_from_fchk


SPECIES = ("h2o", "h2s", "h2co", "h2cs")


def main() -> None:
    print("species      D                 D_N               ||tau_std||_F        ||tau(H22)||_F      ||tau(N)||_F")
    for species in SPECIES:
        res = _h22_from_fchk(str(ROOT / f"{species}.fchk"), "I", "S")
        d = res["d_diagnostic"]
        dn = res["dn_diagnostic"]
        print(
            f"{species:<10} "
            f"{float(d['D']): .10e}  "
            f"{float(dn['D_N']): .10e}  "
            f"{float(d['tau_std_fro']): .10e}  "
            f"{float(d['tau_h22_fro']): .10e}  "
            f"{float(dn['tau_intrinsic_fro']): .10e}"
        )


if __name__ == "__main__":
    main()
