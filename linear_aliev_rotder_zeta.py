#!/usr/bin/env python3
"""Rotational-derivative-style zeta builder for linear Aliev workflows.

This module does not identify Gaussian pairwise Coriolis blocks with the Aliev
zeta object used in the bending seed.  Instead, it builds a canonical
degenerate-pair basis from geometry-side first-level rotational tensors and
then exposes the resulting canonical Coriolis vectors and their quadratic
invariants.  These outputs are the correct starting point for a future
rotational-derivative zeta reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compare_gaussian_sextic import (
    _c1_from_mu1,
    _classify_rotor_limit,
    _degenerate_mode_metadata,
    _special_alpha_context,
)
from rovib_distortion import coriolis_zeta_tensor


@dataclass(frozen=True)
class LinearAlievRotderZeta:
    """Canonicalized linear-molecule zeta-building data.

    Attributes
    ----------
    symmetry_axis_index
        Cartesian symmetry-axis index in the current model representation.
    degenerate_axis_indices
        The two perpendicular Cartesian-axis indices of the degenerate plane.
    parallel_indices
        Indices of non-degenerate parallel modes.
    perpendicular_pairs
        Degenerate pair indices.
    canonical_pair_rotations
        2x2 orthogonal rotations used to canonicalize each degenerate pair.
    canonical_coriolis_xyz
        Pairwise Coriolis tensor recomputed on the canonicalized mode basis.
    zeta_pair_vectors
        For each parallel mode and degenerate pair, the two canonical
        components along the symmetry-axis Coriolis channel.
    zeta_seed_gram
        Invariant pair seed metric built as dot products of the canonical
        two-component vectors.
    """

    symmetry_axis_index: int
    degenerate_axis_indices: tuple[int, int]
    parallel_indices: tuple[int, ...]
    perpendicular_pairs: tuple[tuple[int, int], ...]
    canonical_pair_rotations: tuple[tuple[tuple[float, float], tuple[float, float]], ...]
    canonical_coriolis_xyz: np.ndarray
    zeta_pair_vectors: np.ndarray
    zeta_seed_gram: np.ndarray


def _rotate_mode_pairs(vib_vecs_mw_pa: np.ndarray, pairs: list[tuple[int, int]], rots: list[np.ndarray]) -> np.ndarray:
    out = np.array(vib_vecs_mw_pa, dtype=float, copy=True)
    for (i, j), rot in zip(pairs, rots, strict=True):
        out[:, [i, j]] = out[:, [i, j]] @ rot
    return out


def _rotate_mode_axis3_tensor(tensor: np.ndarray, pairs: list[tuple[int, int]], rots: list[np.ndarray]) -> np.ndarray:
    out = np.array(tensor, dtype=float, copy=True)
    for (i, j), rot in zip(pairs, rots, strict=True):
        out[:, :, [i, j]] = np.einsum("ab,xyb->xya", rot, out[:, :, [i, j]], optimize=True)
    return out


def build_linear_aliev_rotder_zeta(model) -> LinearAlievRotderZeta:
    """Build canonical pairwise zeta data from geometry and normal modes.

    The construction is:
    1. classify the linear rotor and identify degenerate bending pairs;
    2. build the first-level geometry tensor `c1` from `mu1=d(I^-1)/dQ`;
    3. use the resulting tensorial features to canonicalize each degenerate
       pair inside its 2D subspace;
    4. rotate the harmonic modes and `mu1`-derived tensors into that canonical
       basis;
    5. recompute the pairwise Coriolis tensor on the canonicalized mode basis;
    6. expose the canonical two-component vectors and the quadratic invariant
       Gram matrix needed by the future bending-seed reconstruction.
    """

    rotor_limit = _classify_rotor_limit(np.asarray(model.abc_mhz, dtype=float), np.asarray(model.moments_amu_a2, dtype=float))
    if rotor_limit.get("kind") != "linear":
        raise ValueError("Rotational-derivative zeta builder currently supports only linear molecules.")

    pair_meta = _degenerate_mode_metadata(model, rotor_limit)
    if not pair_meta:
        raise ValueError("No degenerate perpendicular pairs detected for linear zeta builder.")

    deg_modes = {int(x) for meta in pair_meta for x in meta["pair"]}
    parallel_indices = tuple(idx for idx in range(len(model.vib_freq_cm)) if idx not in deg_modes)
    pair_list = [tuple(int(x) for x in meta["pair"]) for meta in pair_meta]

    c1 = _c1_from_mu1(model)
    special_ctx = _special_alpha_context(model, rotor_limit, c1=c1)
    pair_rotation_map = {
        tuple(int(x) for x in meta["pair"]): np.asarray(meta["rotation_2x2"], dtype=float)
        for meta in special_ctx.get("canonical_pair_data", ())
    }
    rots = [pair_rotation_map.get(pair, np.eye(2, dtype=float)) for pair in pair_list]

    vib_canonical = _rotate_mode_pairs(np.asarray(model.vib_vecs_mw_pa, dtype=float), pair_list, rots)
    canonical_coriolis_xyz = coriolis_zeta_tensor(vib_canonical)

    axis_map = {label: idx for idx, label in enumerate(model.xyz_to_abc)}
    sym_lbl = rotor_limit.get("symmetry_axis")
    deg_lbls = tuple(rotor_limit.get("degenerate_axes", ()))
    if sym_lbl is None or len(deg_lbls) != 2:
        raise ValueError("Linear zeta builder requires one symmetry axis and two degenerate perpendicular axes.")
    sym_idx = int(axis_map[str(sym_lbl)])
    deg_axis_indices = (int(axis_map[str(deg_lbls[0])]), int(axis_map[str(deg_lbls[1])]))

    zeta_pair_vectors = np.zeros((len(parallel_indices), len(pair_list), 2), dtype=float)
    for n_pos, n in enumerate(parallel_indices):
        for t_pos, (i, j) in enumerate(pair_list):
            zeta_pair_vectors[n_pos, t_pos, 0] = float(canonical_coriolis_xyz[sym_idx, n, i])
            zeta_pair_vectors[n_pos, t_pos, 1] = float(canonical_coriolis_xyz[sym_idx, n, j])

    zeta_seed_gram = np.zeros((len(pair_list), len(parallel_indices), len(parallel_indices)), dtype=float)
    for t_pos in range(len(pair_list)):
        for n_pos, _n in enumerate(parallel_indices):
            vn = zeta_pair_vectors[n_pos, t_pos, :]
            for np_pos, _np in enumerate(parallel_indices):
                vnp = zeta_pair_vectors[np_pos, t_pos, :]
                zeta_seed_gram[t_pos, n_pos, np_pos] = float(np.dot(vn, vnp))

    return LinearAlievRotderZeta(
        symmetry_axis_index=sym_idx,
        degenerate_axis_indices=deg_axis_indices,
        parallel_indices=parallel_indices,
        perpendicular_pairs=tuple(pair_list),
        canonical_pair_rotations=tuple(tuple(tuple(float(x) for x in row) for row in rot) for rot in rots),
        canonical_coriolis_xyz=canonical_coriolis_xyz,
        zeta_pair_vectors=zeta_pair_vectors,
        zeta_seed_gram=zeta_seed_gram,
    )
