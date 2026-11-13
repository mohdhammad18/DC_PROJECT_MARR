"""Sidebar configuration widgets and session-state helpers for the Streamlit app.

Every configuration value lives in ``st.session_state`` under ``cfg.<section>.<name>``
so that widgets keep their values across reruns and scenario presets can overwrite them.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import TRAJECTORY_NAMES, ConfigError, SimConfig, load_config
from src.odometry import INTEGRATION_METHODS
from src.scenarios import SCENARIOS, scenario_config

PREFIX = "cfg."

TRAJECTORY_LABELS = {
    "straight": "Straight line",
    "square": "Square",
    "circle": "Circle",
    "figure_eight": "Figure eight",
    "custom": "Custom motion sequence",
}


def _key(section: str, name: str) -> str:
    return f"{PREFIX}{section}.{name}"


def apply_config(cfg: SimConfig) -> None:
    """Write a configuration into session state (use inside widget callbacks or before widgets)."""
    for section, values in cfg.to_dict().items():
        for name, value in values.items():
            st.session_state[_key(section, name)] = value


def init_state() -> None:
    """Load defaults from YAML on the first run of a session."""
    if f"{PREFIX}initialised" not in st.session_state:
        apply_config(load_config())
        st.session_state[f"{PREFIX}initialised"] = True
    # Streamlit drops the state of widgets that are not rendered in a run (e.g. trajectory
    # parameters hidden for another trajectory). Re-assigning keeps every setting alive.
    for key in [k for k in st.session_state.keys() if isinstance(k, str) and k.startswith(PREFIX)]:
        st.session_state[key] = st.session_state[key]
    # The defaults are the "full sensor fusion" setup; preselect that preset in the picker.
    st.session_state.setdefault("ui.preset", "full_fusion")


def config_from_state() -> SimConfig:
    """Assemble and validate a :class:`SimConfig` from the widget values."""
    data: dict[str, dict] = {}
    for key, value in st.session_state.items():
        if isinstance(key, str) and key.startswith(PREFIX) and key != f"{PREFIX}initialised":
            section, name = key[len(PREFIX):].split(".", 1)
            data.setdefault(section, {})[name] = value
    return SimConfig.from_dict(data).validate()


def _apply_defaults() -> None:
    apply_config(load_config())


def _apply_preset() -> None:
    key = st.session_state["ui.preset"]
    try:
        apply_config(scenario_config(key, config_from_state()))
    except ConfigError as exc:  # keep the user's (invalid) values; report instead of hiding
        st.session_state["ui.preset_error"] = str(exc)
    else:
        st.session_state.pop("ui.preset_error", None)


def _num(label: str, section: str, name: str, min_value: float, max_value: float | None = None,
         step: float = 0.01, fmt: str = "%.3f", help: str | None = None, disabled: bool = False) -> None:
    st.number_input(label, min_value=min_value, max_value=max_value, step=step, format=fmt,
                    key=_key(section, name), help=help, disabled=disabled)


def render_config_sidebar() -> tuple[SimConfig | None, str | None]:
    """Draw all configuration widgets. Returns ``(config, None)`` or ``(None, error_message)``."""
    init_state()
    ss = st.session_state

    st.sidebar.header("Scenario presets")
    st.sidebar.selectbox("Preset", list(SCENARIOS), format_func=lambda k: SCENARIOS[k].title, key="ui.preset")
    st.sidebar.caption(SCENARIOS[ss["ui.preset"]].description)
    c1, c2 = st.sidebar.columns(2)
    c1.button("Apply preset", on_click=_apply_preset, width="stretch",
              help="Overwrites the sensor settings below; trajectory, duration, dt and seed are kept.")
    c2.button("Defaults", on_click=_apply_defaults, width="stretch", help="Restore config/default_config.yaml.")
    if "ui.preset_error" in ss:
        st.sidebar.error(ss["ui.preset_error"])

    with st.sidebar.expander("Robot configuration", expanded=False):
        _num("Wheel radius r [m]", "robot", "wheel_radius", 0.001, 5.0, 0.005)
        _num("Wheel separation b [m]", "robot", "wheel_separation", 0.01, 10.0, 0.01)
        _num("Initial X [m]", "robot", "initial_x", -1000.0, 1000.0, 0.5, "%.2f")
        _num("Initial Y [m]", "robot", "initial_y", -1000.0, 1000.0, 0.5, "%.2f")
        _num("Initial yaw ψ₀ [rad]", "robot", "initial_yaw", -3.1416, 3.1416, 0.05, "%.3f")

    with st.sidebar.expander("Trajectory and time", expanded=False):
        st.selectbox("Trajectory", TRAJECTORY_NAMES, format_func=TRAJECTORY_LABELS.get, key=_key("trajectory", "name"))
        name = ss[_key("trajectory", "name")]
        if name in ("straight", "circle", "figure_eight", "square"):
            _num("Speed [m/s]", "trajectory", "speed", 0.01, 5.0, 0.05, "%.2f")
        if name == "square":
            _num("Side length [m]", "trajectory", "side_length", 0.1, 500.0, 0.5, "%.2f")
            _num("Corner turn rate [rad/s]", "trajectory", "turn_rate", 0.01, 10.0, 0.05, "%.2f")
        if name in ("circle", "figure_eight"):
            _num("Radius [m]", "trajectory", "radius", 0.1, 500.0, 0.5, "%.2f")
        if name == "custom":
            st.caption("Constant (v, ω) segments; the robot stops after the last one.")
            seg_key = _key("trajectory", "custom_segments")
            edited = st.data_editor(
                pd.DataFrame(ss[seg_key], columns=["duration", "v", "omega"]), num_rows="dynamic",
                key="ui.custom_editor", hide_index=True, width="stretch",
            )
            ss[seg_key] = [
                {k: float(r[k]) for k in ("duration", "v", "omega")}
                for _, r in edited.dropna().iterrows()
            ]
        _num("Time step dt [s]", "simulation", "dt", 0.001, 1.0, 0.005, "%.3f")
        _num("Duration [s]", "simulation", "duration", 0.1, 5000.0, 5.0, "%.1f")

    with st.sidebar.expander("Sensor configuration", expanded=False):
        st.markdown("**Wheel odometry**")
        _num("Wheel-speed noise σ [rad/s]", "odometry", "wheel_noise_std", 0.0, 50.0, 0.05)
        _num("Encoder scale error (right wheel)", "odometry", "scale_error", -0.5, 0.5, 0.005, "%.3f",
             help="Fractional calibration error; creates systematic drift.")
        st.markdown("**IMU**")
        st.toggle("Use gyro yaw rate (prediction + odometry)", key=_key("imu", "enabled"),
                  help="Off: yaw rate is derived from the wheels only.")
        imu_on = ss[_key("imu", "enabled")]
        _num("Gyro noise σ [rad/s]", "imu", "gyro_noise_std", 0.0, 5.0, 0.005, "%.4f", disabled=not imu_on)
        _num("Gyro bias [rad/s]", "imu", "gyro_bias", -1.0, 1.0, 0.001, "%.4f", disabled=not imu_on)
        _num("Bias random walk [rad/s/√s]", "imu", "bias_random_walk_std", 0.0, 1.0, 0.0005, "%.4f", disabled=not imu_on)
        if not imu_on:  # heading correction needs the IMU; keep the settings consistent
            ss[_key("imu", "heading_correction")] = False
        st.toggle("Optional IMU heading correction in EKF", key=_key("imu", "heading_correction"), disabled=not imu_on)
        head_on = imu_on and ss[_key("imu", "heading_correction")]
        _num("Heading noise σ [rad]", "imu", "heading_noise_std", 0.0, 3.0, 0.01, "%.3f", disabled=not head_on)
        _num("Heading rate [Hz]", "imu", "heading_rate_hz", 0.01, 1000.0, 0.5, "%.2f", disabled=not head_on)
        st.markdown("**GPS**")
        st.toggle("GPS enabled", key=_key("gps", "enabled"), help="Off: GPS-denied localization.")
        gps_on = ss[_key("gps", "enabled")]
        _num("GPS position noise σ [m]", "gps", "noise_std", 0.0, 100.0, 0.1, "%.2f", disabled=not gps_on)
        _num("GPS update rate [Hz]", "gps", "rate_hz", 0.01, 1000.0, 0.25, "%.2f", disabled=not gps_on)
        _num("Fix dropout probability", "gps", "dropout_probability", 0.0, 1.0, 0.05, "%.2f", disabled=not gps_on)
        st.markdown("**Randomness**")
        st.number_input("Random seed", min_value=0, step=1, key=_key("simulation", "seed"))

    with st.sidebar.expander("EKF tuning", expanded=False):
        st.selectbox("Motion-model integration", INTEGRATION_METHODS, key=_key("ekf", "integration"),
                     help="euler is the model from the brief; midpoint is second-order accurate.")
        _num("Process noise q_pos [m/√s]", "ekf", "q_pos", 0.0, 10.0, 0.01, "%.3f", help="Q = diag(q²) · dt")
        _num("Process noise q_yaw [rad/√s]", "ekf", "q_yaw", 0.0, 10.0, 0.005, "%.3f")
        _num("Initial position σ [m]", "ekf", "p0_pos_std", 0.001, 100.0, 0.05, "%.3f")
        _num("Initial yaw σ [rad]", "ekf", "p0_yaw_std", 0.001, 3.2, 0.01, "%.3f")
        st.toggle("Match R to sensor noise", key=_key("ekf", "match_sensor_noise"),
                  help="On: R is built from the GPS / heading noise settings. Off: use the values below "
                       "(mismatched R shows over- or under-confidence).")
        manual = not ss[_key("ekf", "match_sensor_noise")]
        _num("GPS R σ [m]", "ekf", "gps_r_std", 0.001, 100.0, 0.1, "%.3f", disabled=not manual)
        _num("Heading R σ [rad]", "ekf", "heading_r_std", 0.001, 3.0, 0.01, "%.3f", disabled=not manual)

    with st.sidebar.expander("Environment boundary", expanded=False):
        st.toggle("Show rectangular boundary", key=_key("simulation", "boundary_enabled"))
        b_on = ss[_key("simulation", "boundary_enabled")]
        _num("X min [m]", "simulation", "boundary_xmin", -10000.0, 10000.0, 1.0, "%.1f", disabled=not b_on)
        _num("X max [m]", "simulation", "boundary_xmax", -10000.0, 10000.0, 1.0, "%.1f", disabled=not b_on)
        _num("Y min [m]", "simulation", "boundary_ymin", -10000.0, 10000.0, 1.0, "%.1f", disabled=not b_on)
        _num("Y max [m]", "simulation", "boundary_ymax", -10000.0, 10000.0, 1.0, "%.1f", disabled=not b_on)

    try:
        cfg = config_from_state()
    except ConfigError as exc:
        st.sidebar.error(str(exc))
        return None, str(exc)
    st.sidebar.download_button("Download configuration (YAML)", cfg.to_yaml().encode("utf-8"),
                               file_name="simulation_config.yaml", mime="text/yaml", width="stretch")
    return cfg, None
