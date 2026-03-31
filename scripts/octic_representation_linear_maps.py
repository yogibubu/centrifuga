#!/usr/bin/env python3
"""Exact linear representation maps for the octic rotational branch.

This file packages the exact operator-basis results into explicit linear maps
and Moore-Penrose pseudoinverses.
"""

from __future__ import annotations

import sympy as sp

from scripts.derive_octic_operator_basis import asymmetric_top_monomial_labels


def octic_cartesian_vector() -> sp.Matrix:
    """Return the 15-component cartesian coefficient vector."""
    return sp.Matrix([sp.Symbol(label) for label in asymmetric_top_monomial_labels()])


def octic_symmetric_top_from_cartesian_matrix(axis: str = "a") -> sp.Matrix:
    """Return the exact 5x15 linear map cartesian -> symmetric-top."""
    axis = axis.lower()
    if axis != "a":
        raise NotImplementedError("Only the a-axis case is derived explicitly.")
    rows = []
    labels = asymmetric_top_monomial_labels()
    index = {label: i for i, label in enumerate(labels)}

    def row(spec: dict[str, sp.Expr]) -> sp.Matrix:
        vec = [sp.Integer(0)] * len(labels)
        for key, val in spec.items():
            vec[index[key]] = val
        return sp.Matrix([vec])

    rows.append(row({"c_080": 1}))
    rows.append(row({"c_080": -4, "c_260": 1}))
    rows.append(row({"c_080": 6, "c_260": -3, "c_440": 1}))
    rows.append(row({"c_080": -4, "c_260": 3, "c_440": -2, "c_620": 1}))
    rows.append(row({"c_080": 1, "c_260": -1, "c_440": 1, "c_620": -1, "c_800": 1}))
    return sp.Matrix.vstack(*rows)


def octic_linear_from_cartesian_matrix(axis: str = "a") -> sp.Matrix:
    """Return the exact 1x15 linear map cartesian -> linear."""
    axis = axis.lower()
    if axis != "a":
        raise NotImplementedError("Only the a-axis case is derived explicitly.")
    labels = asymmetric_top_monomial_labels()
    vec = [sp.Integer(0)] * len(labels)
    vec[labels.index("c_080")] = 1
    return sp.Matrix([vec])


def octic_symmetric_top_pseudoinverse(axis: str = "a") -> sp.Matrix:
    """Return the exact Moore-Penrose right inverse for the 5x15 map."""
    M = octic_symmetric_top_from_cartesian_matrix(axis)
    return sp.simplify(M.T * (M * M.T).inv())


def octic_linear_pseudoinverse(axis: str = "a") -> sp.Matrix:
    """Return the exact Moore-Penrose right inverse for the 1x15 extractor."""
    M = octic_linear_from_cartesian_matrix(axis)
    return sp.simplify(M.T * (M * M.T).inv())


def octic_linear_image_vector(axis: str = "a") -> sp.Matrix:
    """Return the exact 15-vector spanning the linear image."""
    axis = axis.lower()
    if axis != "a":
        raise NotImplementedError("Only the a-axis case is derived explicitly.")
    labels = asymmetric_top_monomial_labels()
    coeffs = {label: sp.Integer(0) for label in labels}
    coeffs["c_080"] = 1
    coeffs["c_062"] = 4
    coeffs["c_044"] = 6
    coeffs["c_026"] = 4
    coeffs["c_008"] = 1
    return sp.Matrix([coeffs[label] for label in labels])


def octic_linear_embedding_matrix(axis: str = "a") -> sp.Matrix:
    """Return the exact 15x1 embedding of the linear image."""
    return octic_linear_image_vector(axis)


def octic_linear_embedding_pseudoinverse(axis: str = "a") -> sp.Matrix:
    """Return the exact Moore-Penrose left inverse for the linear image embedding."""
    E = octic_linear_embedding_matrix(axis)
    return sp.simplify((E.T * E).inv() * E.T)


def octic_symmetric_top_image_basis(axis: str = "a") -> sp.Matrix:
    """Return a 15x5 basis matrix for the symmetric-top image in cartesian space."""
    return octic_symmetric_top_pseudoinverse(axis)


def main() -> None:
    M = octic_symmetric_top_from_cartesian_matrix("a")
    print("sym map shape:", M.shape)
    print("linear map shape:", octic_linear_from_cartesian_matrix("a").shape)


if __name__ == "__main__":
    main()
