#!/usr/bin/env python3
"""Compare the current quartic-channel implementation against Paper 2 references.

This script is intentionally diagnostic. It makes the current gap explicit
between the validated H22 benchmark and the still provisional H12H30/H30H30
working references collected in ``paper2_quartic_benchmark_note.md``.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from channel_contributions import parse_model, channel_contributions, to_watson_khz
from gaussian_vpt_parser import parse_gaussian_quartic_benchmark
from h22_reference import project_h22_ceditt3_reference_path


NOTE_PATH = REPO_ROOT / "paper2_quartic_benchmark_note.md"
CHANNELS = ("H22", "H12H30", "H30H30")
WATSON_KEYS = ("DJ", "DJK", "DK", "d1", "d2")
CHANNEL_STATUS = {
    "H22": "validated benchmark",
    "H12H30": "working reference",
    "H30H30": "working reference",
}

def parse_benchmark_note(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    text = path.read_text()
    species_blocks = re.split(r"^##\s+", text, flags=re.MULTILINE)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for block in species_blocks[1:]:
        lines = block.strip().splitlines()
        species = lines[0].strip()
        species_data: dict[str, dict[str, float]] = {}
        for channel in CHANNELS:
            match = re.search(rf"^{channel}\s*=\s*(\{{.*\}})$", block, flags=re.MULTILINE)
            if not match:
                raise ValueError(f"Missing {channel} entry for {species} in {path}")
            species_data[channel] = ast.literal_eval(match.group(1))
        out[species] = species_data
    return out


def compute_current_values(species: str, benchmark_species: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    base = REPO_ROOT / species.lower()
    model, phi3, meta = parse_model(base.with_suffix(".fchk"), base.with_suffix(".log"))
    quart = parse_gaussian_quartic_benchmark(base.with_suffix(".log"))
    taus = channel_contributions(model, phi3, phi3_reduced_cm=meta.get("phi3_reduced_cm"))
    axes = quart.spectroscopic_axes
    out = {
        channel: to_watson_khz(
            taus[channel],
            abc_mhz=model.abc_mhz,
            reduction="S",
            spectroscopic_axes=axes,
        )
        for channel in CHANNELS
    }
    out["H22"] = project_h22_ceditt3_reference_path(base.with_suffix(".fchk"), base.with_suffix(".log"))
    return out


def fmt(value: float) -> str:
    return f"{value: .6e}"


def main() -> None:
    benchmark = parse_benchmark_note(NOTE_PATH)
    print("Paper 2 quartic reference audit")
    print("NOTE: H22 is a validated benchmark; H12H30/H30H30 are provisional working references.\n")

    for species in ("H2O", "H2S", "H2CO", "H2CS"):
        current = compute_current_values(species, benchmark[species])
        print(f"=== {species} ===")
        for channel in CHANNELS:
            print(f"{channel} [{CHANNEL_STATUS[channel]}]")
            max_abs_delta = 0.0
            for key in WATSON_KEYS:
                ref = benchmark[species][channel][key]
                got = current[channel][key]
                delta = got - ref
                max_abs_delta = max(max_abs_delta, abs(delta))
                ratio = float("inf") if ref == 0.0 and got != 0.0 else (got / ref if ref != 0.0 else 1.0)
                print(
                    f"  {key:<3} ref={fmt(ref)}  got={fmt(got)}  "
                    f"delta={fmt(delta)}  ratio={fmt(ratio)}"
                )
            print(f"  max|delta| = {fmt(max_abs_delta)}")
        print()


if __name__ == "__main__":
    main()
