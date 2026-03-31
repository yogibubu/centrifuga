from scripts.paper1_octic_axial_projection import paper1_octic_axial_projection


def test_axial_projection_exposes_five_coefficients() -> None:
    out = paper1_octic_axial_projection()
    assert tuple(out["axial_coefficients"].keys()) == ("X0", "X1", "X2", "X3", "X4")


def test_linear_limit_is_x0() -> None:
    out = paper1_octic_axial_projection()
    assert out["linear_constant_L"] == out["axial_coefficients"]["X0"]


def test_x0_matches_ceditt4_style_combination() -> None:
    out = paper1_octic_axial_projection()
    total = out["cartesian_vector_15"]
    # Stable orth/cart order: c008,c026,c044,c062,c080,... so X0 is the
    # Moore-Penrose projection of the axial image and contains the pure
    # transverse components only.
    assert out["axial_coefficients"]["X0"].has(total[0], total[1], total[2], total[3], total[4])
