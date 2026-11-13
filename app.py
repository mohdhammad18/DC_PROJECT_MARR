"""Robot Localization Simulator - Streamlit entry point.

Run from the repository root with::

    streamlit run app.py

This file only wires the UI to the simulation engine in ``src/``; all numerical
work happens there and is independent of Streamlit.
"""

from __future__ import annotations

import time

import streamlit as st

from src.config import ConfigError, SimConfig
from src.simulation import Simulation
from visualization import dashboard
from visualization.controls import config_from_state, render_config_sidebar

st.set_page_config(page_title="Robot Localization Simulator", page_icon="🤖", layout="wide")

VIEWS = [
    "Trajectory comparison",
    "Error analysis",
    "EKF internals",
    "Coordinate frames",
    "Sensor data",
    "Scenario comparison",
    "Animated replay",
]
PLAYBACK_DELAY_S = 0.1  # seconds between playback frames


# ------------------------------------------------------------- simulation state
def _new_simulation(cfg: SimConfig) -> Simulation:
    sim = Simulation(cfg)
    st.session_state["sim"] = sim
    st.session_state["sim_cfg"] = cfg.to_dict()
    return sim


def _fresh_simulation_from_sidebar() -> Simulation | None:
    """Build a simulation from the current sidebar values (callbacks run before the script)."""
    try:
        return _new_simulation(config_from_state())
    except ConfigError:
        return None  # the sidebar already shows the validation error


# Button callbacks: they fire exactly once per click, before the script reruns.
def _on_start() -> None:
    if _fresh_simulation_from_sidebar() is not None:
        st.session_state["playing"] = True


def _on_pause_resume() -> None:
    sim = st.session_state.get("sim")
    if sim is not None and not sim.done:
        st.session_state["playing"] = not st.session_state.get("playing", False)


def _on_reset() -> None:
    if _fresh_simulation_from_sidebar() is not None:
        st.session_state["playing"] = False


def _on_step() -> None:
    sim = st.session_state.get("sim")
    st.session_state["playing"] = False
    if sim is not None and not sim.done:
        sim.step()


def _on_run_complete() -> None:
    sim = _fresh_simulation_from_sidebar()
    if sim is not None:
        sim.run()
    st.session_state["playing"] = False


def _controls(cfg: SimConfig | None) -> None:
    """Sidebar simulation controls. Every button runs real simulation code."""
    ss = st.session_state
    sim: Simulation | None = ss.get("sim")
    playing = bool(ss.get("playing", False))
    can_run = cfg is not None

    st.sidebar.header("Simulation controls")
    ss["steps_per_frame"] = st.sidebar.slider(
        "Playback speed (steps per frame)", 1, 100, ss.get("steps_per_frame", 10),
        help="How many simulation steps are advanced between screen updates while playing.",
    )
    c1, c2 = st.sidebar.columns(2)
    c1.button("▶ Start", on_click=_on_start, disabled=not can_run, width="stretch",
              help="Build a fresh simulation from the current settings and play it from t = 0.")
    c2.button("⏸ Pause" if playing else "⏯ Resume", on_click=_on_pause_resume,
              disabled=sim is None or sim.done, width="stretch",
              help="Pause or resume the running playback.")
    c3, c4 = st.sidebar.columns(2)
    c3.button("↺ Reset", on_click=_on_reset, disabled=not can_run, width="stretch",
              help="Return to t = 0 with the current settings (nothing is run).")
    c4.button("Step ▸", on_click=_on_step, disabled=sim is None or sim.done, width="stretch",
              help="Advance exactly one time step.")
    st.sidebar.button("Run complete simulation", on_click=_on_run_complete, disabled=not can_run,
                      width="stretch", type="primary",
                      help="Rebuild from the current settings and run to the end.")


def main() -> None:
    ss = st.session_state
    cfg, cfg_error = render_config_sidebar()
    st.title("🤖 Robot Localization Simulator")
    st.caption(
        "Planar 2D differential-drive robot · wheel odometry + IMU + intermittent GPS fused with an "
        "Extended Kalman Filter. (Planar only: no 6-DoF localization.)"
    )

    # First visit: run the default configuration once so the dashboard shows real data.
    if "sim" not in ss and cfg is not None:
        _new_simulation(cfg).run()
        ss["playing"] = False

    _controls(cfg)

    sim: Simulation | None = ss.get("sim")
    if sim is None:
        st.error(cfg_error or "No valid configuration.")
        return

    # Advance playback before drawing so the display shows the newest state.
    if ss.get("playing") and not sim.done:
        sim.run(ss.get("steps_per_frame", 10))
    if sim.done:
        ss["playing"] = False

    if cfg is not None and ss.get("sim_cfg") != cfg.to_dict():
        st.info("Sidebar settings differ from the simulation shown. Use **Start**, **Reset** or "
                "**Run complete simulation** to apply them.")
    elif cfg is None:
        st.error("The sidebar configuration is invalid; showing the last simulation. See the sidebar for details.")

    df = sim.to_dataframe()
    dashboard.render_overview(sim, df)
    st.divider()

    view = st.radio("View", VIEWS, horizontal=True, key="ui.view", label_visibility="collapsed")
    if view == "Trajectory comparison":
        dashboard.render_trajectory(sim, df)
    elif view == "Error analysis":
        dashboard.render_error_analysis(df)
    elif view == "EKF internals":
        dashboard.render_ekf_internals(sim, df)
    elif view == "Coordinate frames":
        dashboard.render_coordinate_frames(sim, df)
    elif view == "Sensor data":
        dashboard.render_sensor_data(df)
    elif view == "Scenario comparison":
        dashboard.render_scenarios(cfg, cfg_error)
    else:
        dashboard.render_animation(sim, df)

    if ss.get("playing") and not sim.done:
        time.sleep(PLAYBACK_DELAY_S)
        st.rerun()


main()
