from gaussian_vpt_parser import parse_gaussian_linear_ltype_constants


def test_parse_gaussian_linear_ltype_constants_hccd() -> None:
    vals = parse_gaussian_linear_ltype_constants("hccd.log")
    assert abs(vals.q_e_mhz[5] - 101.45950720474154) < 1.0e-9
    assert abs(vals.q_e_mhz[7] - 139.93963652553992) < 1.0e-9
    assert abs(vals.q_j_mhz[5] - 0.00501490300389281) < 1.0e-12
    assert abs(vals.q_j_mhz[7] + 0.00504295090573441) < 1.0e-12
    assert abs(vals.q_k_mhz[5] + 0.005076692972587601) < 1.0e-12
    assert abs(vals.q_k_mhz[7] - 0.004956682199317115) < 1.0e-12
    assert vals.active_dd_22_count == 13
