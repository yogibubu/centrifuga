import sympy as sp

from derive_sextic_transform import invariant_sigma_basis, monomial_list, poly_vec


def test_fixed_representation_handedness_flip_does_not_preserve_validated_sextic_5d_subspace() -> None:
    ja, jb, jc = sp.symbols("Ja Jb Jc")
    sigma_ops = invariant_sigma_basis()
    swap_bc = {ja: ja, jb: jc, jc: jb}

    sigma_swapped = [sp.expand(op.subs(swap_bc)) for op in sigma_ops]
    mons = monomial_list(sigma_ops + sigma_swapped)
    sigma_mat = sp.Matrix.hstack(*[poly_vec(op, mons) for op in sigma_ops])
    sigma_swapped_mat = sp.Matrix.hstack(*[poly_vec(op, mons) for op in sigma_swapped])

    # The validated sextic transport uses a 5D invariant subspace under cyclic
    # right-handed relabelings. A fixed-representation handedness flip (b<->c)
    # generates additional directions, so the joined span is larger than 5D.
    joined_rank = sp.Matrix.hstack(sigma_mat, sigma_swapped_mat).rank()

    assert sigma_mat.rank() == 5
    assert joined_rank > 5
    assert joined_rank == 9
