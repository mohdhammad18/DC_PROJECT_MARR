"""Smoke tests that execute app.py headlessly with Streamlit's AppTest runner."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = 120


def _button(at, label):
    return next(b for b in at.sidebar.button if label in b.label)


def test_app_starts_and_shows_a_finished_default_simulation():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    assert any("EKF position error" in m.label for m in at.metric)
    assert at.session_state["sim"].done


def test_controls_reset_step_and_run_complete():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    _button(at, "Reset").click().run()
    assert not at.exception, at.exception
    assert at.session_state["sim"].step_index == 0
    _button(at, "Step").click().run()
    assert at.session_state["sim"].step_index == 1
    _button(at, "Run complete").click().run()
    assert not at.exception, at.exception
    assert at.session_state["sim"].done


def test_start_plays_the_simulation_from_zero_to_the_end():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.sidebar.slider[0].set_value(100)  # large frames so the test stays quick
    at.run()
    _button(at, "Start").click().run()
    assert not at.exception, at.exception
    sim = at.session_state["sim"]
    assert sim.done and sim.step_index == sim.n_steps  # a fresh run, played to completion
    assert at.session_state["playing"] is False


def test_resume_button_plays_an_unfinished_simulation():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    _button(at, "Reset").click().run()  # unfinished simulation -> Resume is enabled
    sim = at.session_state["sim"]
    assert not sim.done and not _button(at, "Resume").disabled
    at.sidebar.slider[0].set_value(100)
    _button(at, "Resume").click().run()
    assert at.session_state["sim"].done  # resume played the simulation to the end


@pytest.mark.parametrize(
    "view",
    ["Error analysis", "EKF internals", "Coordinate frames", "Sensor data", "Scenario comparison", "Animated replay"],
)
def test_every_view_renders_without_errors(view):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.radio(key="ui.view").set_value(view).run()
    assert not at.exception, (view, at.exception)


def test_gps_toggle_and_invalid_config_are_handled():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.session_state["cfg.gps.enabled"] = False
    at.session_state["cfg.simulation.boundary_xmin"] = 50.0  # xmin > xmax -> invalid
    at.run()
    assert not at.exception, at.exception
    assert any("Boundary" in e.value for e in at.sidebar.error)
    assert _button(at, "Run complete").disabled
