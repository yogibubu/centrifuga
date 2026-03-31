from pathlib import Path

from ceditt_gui import App, _abc_delta_info, _harmonization_status, _point_group_from_xyz_file


ROOT = Path(__file__).resolve().parents[2]
H2S_XYZ = ROOT / "data/gaussian/linear_cases/h2s.xyz"


class DummyVar:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


def _dummy_app() -> App:
    app = object.__new__(App)
    app.vars = {
        "A": DummyVar(),
        "B": DummyVar(),
        "C": DummyVar(),
        "q_symm_xyz": DummyVar(),
        "q_h22_xyz": DummyVar(),
        "s_symm_xyz": DummyVar(),
    }
    return app


def test_xyz_reference_populates_abc_fields() -> None:
    meta = _point_group_from_xyz_file(str(H2S_XYZ))
    abc = tuple(float(x) for x in meta["abc_mhz_from_xyz"])

    app = _dummy_app()
    app._apply_xyz_abc_reference(meta)
    assert abs(float(app.vars["A"].get()) - abc[0]) < 1.0e-3
    assert abs(float(app.vars["B"].get()) - abc[1]) < 1.0e-3
    assert abs(float(app.vars["C"].get()) - abc[2]) < 1.0e-3


def test_read_abc_prefers_xyz_reference_over_manual_values() -> None:
    meta = _point_group_from_xyz_file(str(H2S_XYZ))
    abc = tuple(float(x) for x in meta["abc_mhz_from_xyz"])

    app = _dummy_app()
    app.vars["A"].set("1")
    app.vars["B"].set("2")
    app.vars["C"].set("3")
    app.vars["q_symm_xyz"].set(str(H2S_XYZ))
    read_abc = app._read_abc()
    assert read_abc == abc


def test_harmonize_model_abc_with_xyz_returns_delta() -> None:
    app = _dummy_app()
    app.vars["q_h22_xyz"].set(str(H2S_XYZ))
    info = app._harmonize_model_abc_with_xyz((307610.67, 262602.14, 141665.04), "q_h22_xyz")
    assert info is not None
    assert info["xyz_path"] == str(H2S_XYZ)
    assert info["max_delta_abc_mhz"] > 0.0


def test_harmonization_status_thresholds() -> None:
    assert _harmonization_status(0.5) == "OK"
    assert _harmonization_status(10.0) == "CHECK"
    assert _harmonization_status(80.0) == "WARNING"


def test_abc_delta_info_reports_status() -> None:
    info = _abc_delta_info((1.0, 2.0, 3.0), (1.2, 2.0, 3.0))
    assert info["status"] == "OK"
    info = _abc_delta_info((1.0, 2.0, 3.0), (20.0, 2.0, 3.0))
    assert info["status"] == "CHECK"
    info = _abc_delta_info((1.0, 2.0, 3.0), (100.0, 2.0, 3.0))
    assert info["status"] == "WARNING"


def test_read_abc_can_use_sextic_xyz_reference() -> None:
    meta = _point_group_from_xyz_file(str(H2S_XYZ))
    abc = tuple(float(x) for x in meta["abc_mhz_from_xyz"])

    app = _dummy_app()
    app.vars["A"].set("9")
    app.vars["B"].set("8")
    app.vars["C"].set("7")
    app.vars["s_symm_xyz"].set(str(H2S_XYZ))
    read_abc = app._read_abc("s_symm_xyz")
    assert read_abc == abc
