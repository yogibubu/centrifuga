#!/usr/bin/env python3
"""Inspect what the current scaffold-level H30H30 fit fails to reproduce."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.h30h30_scaffold_fit import NOTE_PATH, SCAFFOLDS, SPECIES, WATSON_KEYS, parse_benchmark_note, scaffold_vectors


def main() -> None:
    benchmark = parse_benchmark_note(NOTE_PATH)
    rows = []
    rhs = []
    per_species = {}
    for species in SPECIES:
        vecs = scaffold_vectors(species)
        per_species[species] = vecs
        ref = benchmark[species]["H30H30"]
        for idx, key in enumerate(WATSON_KEYS):
            rows.append([vecs[name][idx] for name in SCAFFOLDS])
            rhs.append(ref[key])
    A = np.array(rows, dtype=float)
    b = np.array(rhs, dtype=float)
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    print("H30H30 residual audit after scaffold-level fit")
    print("coefficients =", dict(zip(SCAFFOLDS, coeffs)))

    for species in SPECIES:
        vecs = per_species[species]
        fitted = sum(coeffs[i] * vecs[name] for i, name in enumerate(SCAFFOLDS))
        ref = np.array([benchmark[species]["H30H30"][key] for key in WATSON_KEYS], dtype=float)
        delta = fitted - ref
        print(f"\n{species}")
        print("  residual =", dict(zip(WATSON_KEYS, delta)))
        print("  residual norm =", float(np.linalg.norm(delta)))
        dominant = WATSON_KEYS[int(np.argmax(np.abs(delta)))]
        print("  dominant residual component =", dominant)


if __name__ == "__main__":
    main()
