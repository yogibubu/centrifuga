#!/usr/bin/env python3
"""Deterministic Gaussian-side alignment helpers for quartic/sextic workflows."""

from __future__ import annotations

from dataclasses import dataclass

from compare_gaussian_sextic import _aligned_model_from_gaussian


DEFAULT_GAUSSIAN_ALIGNMENT_MAX_DIDQ_ERR = 1.0e-3


@dataclass(frozen=True)
class GaussianAlignedModel:
    model: object
    representation: str
    order: tuple[int, ...]
    signs: tuple[float, ...]
    didq_err: float

    @property
    def is_reliable(self) -> bool:
        return self.didq_err <= DEFAULT_GAUSSIAN_ALIGNMENT_MAX_DIDQ_ERR


def load_gaussian_aligned_model(fchk_path: str, log_path: str) -> GaussianAlignedModel:
    """Return the deterministic Gaussian-aligned harmonic model from ``fchk`` + ``log``.

    The alignment is derived from the data already present in the Gaussian
    output: representation choice, mode reorder, and mode-sign convention are
    fixed by the `dIdQ` block comparison, not by per-molecule fitting against
    final quartic constants.
    """
    model, rep, order, signs, didq_err = _aligned_model_from_gaussian(fchk_path, log_path)
    return GaussianAlignedModel(
        model=model,
        representation=str(rep),
        order=tuple(int(x) for x in order.tolist()),
        signs=tuple(float(x) for x in signs.tolist()),
        didq_err=float(didq_err),
    )
