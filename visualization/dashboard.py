"""Streamlit panels (render functions) for the dashboard. All data comes from the simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import SimConfig
from src.coordinate_frames import global_to_local, local_to_global, rotation_matrix
from src.metrics import compare_estimators
from src.scenarios import SCENARIOS, run_scenarios
from src.sensors import GPS_DISABLED, GPS_DROPOUT, GPS_FIX, GPS_IDLE
from src.simulation import GPS_UPDATE_FAILED, Simulation

from .robot_animation import animation_figure
from .sensor_plots import (
    covariance_figure,
    gps_availability_figure,
    innovation_figure,
    position_error_figure,
    scenario_errors_figure,
    scenario_trajectories_figure,
    sensor_signals_figure,
    yaw_error_figure,
)
from .trajectory import boundary_from_settings, coordinate_frame_figure, trajectory_figure

_GPS_STATUS_TEXT = {
    GPS_FIX: "Fix received",
    GPS_DROPOUT: "Fix lost (dropout)",
    GPS_IDLE: "No fix scheduled",
    GPS_DISABLED: "GPS disabled",
    GPS_UPDATE_FAILED: "Update rejected",
    "initial": "Not started",
}


def show_plot(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})


def show_table(df: pd.DataFrame, **kwargs) -> None:
    st.dataframe(df, width="stretch", **kwargs)


def _fmt_state(x: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"x [m]": [x[0]], "y [m]": [x[1]], "ψ [deg]": [np.degrees(x[2])]}).round(4)


# ------------------------------------------------------------------ A. Overview
def render_overview(sim: Simulation, df: pd.DataFrame) -> None:
    cfg = sim.config
    row = df.iloc[-1]
    st.progress(
        min(1.0, sim.step_index / max(sim.n_steps, 1)),
        text=f"Simulation time {row.t:.2f} s of {cfg.simulation.duration:.1f} s  "
        f"(step {sim.step_index}/{sim.n_steps})" + ("  - finished" if sim.done else ""),
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Simulation time", f"{row.t:.2f} s")
    c2.metric("True position", f"({row.x_true:.2f}, {row.y_true:.2f}) m")
    c3.metric("EKF heading estimate", f"{np.degrees(row.psi_ekf):.1f}°",
              delta=f"true {np.degrees(row.psi_true):.1f}°", delta_color="off")
    c4.metric("GPS (this step)", _GPS_STATUS_TEXT.get(row.gps_status, row.gps_status))
    last_fix = df.t[df.gps_x.notna()]
    c5.metric("Last GPS fix", f"{last_fix.iloc[-1]:.1f} s" if len(last_fix) else "none yet")

    e1, e2, e3 = st.columns(3)
    e1.metric("EKF position error", f"{row.err_ekf:.3f} m",
              delta=f"{row.err_ekf - row.err_odo:+.3f} m vs odometry", delta_color="inverse")
    e2.metric("Odometry-only position error", f"{row.err_odo:.3f} m")
    e3.metric("EKF yaw error", f"{np.degrees(abs(row.yaw_err_ekf)):.2f}°",
              delta=f"odometry {np.degrees(abs(row.yaw_err_odo)):.2f}°", delta_color="off")

    with st.expander("Robot and run configuration"):
        r = cfg.robot
        st.table(
            pd.DataFrame(
                {
                    "Parameter": [
                        "Wheel radius r", "Wheel separation b", "Initial pose (x, y, ψ)", "Trajectory",
                        "Time step dt", "Duration", "Random seed", "GPS", "IMU yaw rate in prediction",
                    ],
                    "Value": [
                        f"{r.wheel_radius:.3f} m", f"{r.wheel_separation:.3f} m",
                        f"({r.initial_x:.2f} m, {r.initial_y:.2f} m, {r.initial_yaw:.2f} rad)",
                        cfg.trajectory.name, f"{cfg.simulation.dt:.3f} s", f"{cfg.simulation.duration:.1f} s",
                        str(cfg.simulation.seed),
                        f"{cfg.gps.rate_hz:.2f} Hz, σ={cfg.gps.noise_std:.2f} m, dropout {cfg.gps.dropout_probability:.0%}"
                        if cfg.gps.enabled else "disabled",
                        "yes (gyro)" if cfg.imu.enabled else "no (wheel-derived ω)",
                    ],
                }
            ).set_index("Parameter")
        )
    for msg in sim.warnings[-5:]:
        st.warning(msg)
    n_out = sim.boundary_violations()
    if n_out:
        st.warning(f"The ground-truth path leaves the environment boundary at {n_out} time steps.")


# ------------------------------------------------------- F. Trajectory comparison
def render_trajectory(sim: Simulation, df: pd.DataFrame) -> None:
    c1, c2, c3 = st.columns(3)
    show_gps = c1.checkbox("Show GPS measurements", value=True, key="ui.show_gps")
    show_bound = c2.checkbox("Show environment boundary", value=True, key="ui.show_boundary")
    show_ell = c3.checkbox("Show EKF 2σ position ellipse", value=False, key="ui.show_ellipse")
    boundary = boundary_from_settings(sim.config.simulation) if show_bound else None
    P_xy = sim.ekf.covariance[:2, :2]
    show_plot(
        trajectory_figure(df, None, boundary, show_gps, show_ell, P_xy),
        key="fig_trajectory",
    )
    st.caption(
        "Green: ground truth · orange dotted: odometry-only (wheel + gyro, no GPS) · blue: EKF fusion · "
        "purple ×: GPS fixes. Scroll to zoom, drag to pan, hover for values."
    )


# --------------------------------------------------------------- G. Error analysis
def render_error_analysis(df: pd.DataFrame) -> None:
    table = compare_estimators(df)
    st.subheader("Performance metrics")
    show_table(table.style.format("{:.4f}"))
    rmse_odo, rmse_ekf = table.iloc[0, 1], table.iloc[1, 1]
    if rmse_odo > 1e-9:
        gain = 100.0 * (rmse_odo - rmse_ekf) / rmse_odo
        verb = "lower" if gain >= 0 else "higher"
        st.caption(f"EKF RMSE is {abs(gain):.1f} % {verb} than odometry-only for this run.")
    show_plot(position_error_figure(df), "fig_pos_err")
    show_plot(yaw_error_figure(df), "fig_yaw_err")
    if st.checkbox("Show covariance evolution", value=False, key="ui.show_cov"):
        show_plot(covariance_figure(df), "fig_cov")


# ------------------------------------------------------------ H. EKF internals
def _sigma_row(P: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {"σ_x [m]": [np.sqrt(P[0, 0])], "σ_y [m]": [np.sqrt(P[1, 1])], "σ_ψ [deg]": [np.degrees(np.sqrt(P[2, 2]))]}
    ).round(4)


def render_ekf_internals(sim: Simulation, df: pd.DataFrame) -> None:
    records = sim.ekf_records
    if len(records) < 2:
        st.info("Run at least one simulation step to inspect the EKF.")
        return
    gps_steps = [r.step for r in records if r is not None and r.gps_update is not None]
    only_gps = st.checkbox(
        "Only step through GPS updates", value=bool(gps_steps), disabled=not gps_steps, key="ui.ekf_only_gps"
    )
    options = gps_steps if (only_gps and gps_steps) else list(range(1, len(records)))
    if len(options) == 1:
        step = options[0]
        st.caption(f"Only one step available: {step}")
    else:
        key = "ui.ekf_step"
        if key in st.session_state and st.session_state[key] not in options:
            # Snap a stale value (after reset / filter change) to the nearest valid step.
            st.session_state[key] = min(options, key=lambda s: abs(s - st.session_state[key]))
        step = st.select_slider("Simulation step", options=options, value=options[-1], key=key)
    rec = records[step]
    assert rec is not None
    pred, up = rec.prediction, rec.gps_update
    row = df.iloc[step]
    st.markdown(f"**Step {step}, t = {rec.time:.2f} s** - GPS status: *{_GPS_STATUS_TEXT.get(row.gps_status, row.gps_status)}*")

    st.markdown("#### 1. Prediction step")
    st.write(
        "The EKF propagates the previous estimate with the measured speed *v* and yaw rate *ω* "
        "(wheel odometry for *v*, gyro for *ω* when the IMU is enabled) and inflates the covariance: "
        "$P^- = F P F^\\top + Q$."
    )
    a, b = st.columns(2)
    a.write(f"Inputs: v = {pred.v:.4f} m/s, ω = {pred.omega:.4f} rad/s, Δt = {pred.dt:.3f} s")
    a.write("Previous state → predicted state x⁻")
    a.dataframe(pd.concat([_fmt_state(pred.x_prior), _fmt_state(pred.x_pred)], keys=["before", "predicted x⁻"]).droplevel(1))
    b.write("Position / yaw uncertainty (1σ): before → predicted P⁻")
    b.dataframe(pd.concat([_sigma_row(pred.P_prior), _sigma_row(pred.P_pred)], keys=["before", "predicted P⁻"]).droplevel(1))

    st.markdown("#### 2. GPS measurement arrival")
    if up is None:
        st.info(
            f"No GPS correction was applied at this step ({_GPS_STATUS_TEXT.get(row.gps_status, row.gps_status).lower()}). "
            "The corrected state equals the predicted state."
        )
    else:
        true_xy = df.loc[step, ["x_true", "y_true"]].to_numpy(dtype=float)
        st.write(
            f"A GPS fix z = ({up.z[0]:.3f}, {up.z[1]:.3f}) m arrived (global frame). "
            f"Its actual error versus the ground truth is {np.linalg.norm(up.z - true_xy):.3f} m."
        )

        st.markdown("#### 3. Innovation")
        st.write("The innovation is the difference between the measurement and what the prediction expects: "
                 "$r = z - H x^-$, with innovation covariance $S = H P^- H^\\top + R$.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Innovation r_x", f"{up.innovation[0]:+.3f} m")
        c2.metric("Innovation r_y", f"{up.innovation[1]:+.3f} m")
        c3.metric("NIS  rᵀS⁻¹r", f"{up.nis:.2f}", help="≈ 2 on average for a consistent 2-D measurement (χ² with 2 DoF).")

        st.markdown("#### 4. Kalman gain")
        st.write("$K = P^- H^\\top S^{-1}$ weighs the innovation by the relative uncertainty of prediction and GPS. "
                 "A gain near 1 trusts the GPS; near 0 trusts the prediction.")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("K[x, x]", f"{up.K[0, 0]:.3f}")
        g2.metric("K[y, y]", f"{up.K[1, 1]:.3f}")
        g3.metric("K[ψ, x]", f"{up.K[2, 0]:.3f}", help="How much an x innovation corrects the yaw estimate (via correlations in P⁻).")
        g4.metric("K[ψ, y]", f"{up.K[2, 1]:.3f}")

        st.markdown("#### 5. State correction")
        st.write("$\\hat x = x^- + K r$ and the covariance shrinks (Joseph form): "
                 "$P = (I-KH)P^-(I-KH)^\\top + K R K^\\top$.")
        a, b = st.columns(2)
        a.write("Predicted → corrected state")
        a.dataframe(pd.concat([_fmt_state(up.x_pred), _fmt_state(up.x_corrected)], keys=["predicted x⁻", "corrected x̂"]).droplevel(1))
        b.write("Uncertainty (1σ): predicted → updated")
        b.dataframe(pd.concat([_sigma_row(up.P_pred), _sigma_row(up.P_updated)], keys=["P⁻", "P updated"]).droplevel(1))
        err_before = np.linalg.norm(up.x_pred[:2] - true_xy)
        err_after = np.linalg.norm(up.x_corrected[:2] - true_xy)
        st.write(f"Position error versus truth: {err_before:.3f} m before → {err_after:.3f} m after the update.")

    if rec.heading_update is not None:
        hu = rec.heading_update
        st.info(
            f"An IMU heading correction was also applied: wrapped yaw innovation "
            f"{np.degrees(hu.innovation[0]):+.2f}°, gain K[ψ] = {hu.K[2, 0]:.3f}."
        )

    with st.expander("Detailed matrices and equations"):
        st.latex(r"F=\begin{bmatrix}1&0&-v\sin\psi\,\Delta t\\0&1&v\cos\psi\,\Delta t\\0&0&1\end{bmatrix},\quad "
                 r"H=\begin{bmatrix}1&0&0\\0&1&0\end{bmatrix}")

        def mat(name: str, m: np.ndarray) -> None:
            st.write(name)
            st.dataframe(pd.DataFrame(m).round(6))

        mat("F (state-transition Jacobian)", pred.F)
        mat("Q (process noise, per step)", pred.Q)
        mat("P⁻ (predicted covariance)", pred.P_pred)
        if up is not None:
            mat("H", up.H)
            mat("R (GPS noise covariance)", up.R)
            mat("S (innovation covariance)", up.S)
            mat("K (Kalman gain)", up.K)
            mat("P (updated covariance)", up.P_updated)


# ------------------------------------------------ E. Coordinate frame visualisation
def render_coordinate_frames(sim: Simulation, df: pd.DataFrame) -> None:
    st.write(
        "The **global frame** (X_G, Y_G) is fixed to the world. The **local frame** (X_L, Y_L) is attached to the "
        "robot: X_L points along the heading ψ, Y_L to the robot's left. Motion measured by the wheels is "
        "expressed in the local frame and must be rotated by R(ψ) before it can be added to a global position."
    )
    st.latex(r"R(\psi)=\begin{bmatrix}\cos\psi&-\sin\psi\\ \sin\psi&\cos\psi\end{bmatrix},\qquad "
             r"p_G = p_{robot,G} + R(\psi)\,p_L,\qquad p_L = R(\psi)^\top\,(p_G-p_{robot,G})")
    use_sim = st.checkbox("Use the simulation's current true pose", value=False, key="ui.cf_use_sim")
    last = df.iloc[-1]
    c1, c2, c3 = st.columns(3)
    if use_sim:
        rx, ry, psi = float(last.x_true), float(last.y_true), float(last.psi_true)
        c1.write(f"Robot position: ({rx:.2f}, {ry:.2f}) m")
        c2.write(f"Heading ψ: {np.degrees(psi):.1f}°")
    else:
        rx = c1.slider("Robot X in global frame [m]", -3.0, 6.0, 2.0, 0.1, key="ui.cf_rx")
        ry = c1.slider("Robot Y in global frame [m]", -3.0, 6.0, 1.0, 0.1, key="ui.cf_ry")
        psi = np.radians(c2.slider("Heading ψ [deg]", -180, 180, 35, 1, key="ui.cf_psi"))
    lx = c3.slider("Local movement vector: forward (X_L) [m]", -2.0, 3.0, 1.5, 0.1, key="ui.cf_lx")
    ly = c3.slider("Local movement vector: left (Y_L) [m]", -2.0, 3.0, 0.7, 0.1, key="ui.cf_ly")

    pL = np.array([lx, ly])
    pG = local_to_global(pL, [rx, ry], psi)
    back = global_to_local(pG, [rx, ry], psi)
    show_plot(coordinate_frame_figure((rx, ry), psi, (lx, ly)), "fig_frames")
    a, b = st.columns(2)
    a.write("**R(ψ)**")
    a.dataframe(pd.DataFrame(rotation_matrix(psi), columns=["X_L → ", "Y_L → "], index=["X_G", "Y_G"]).round(4))
    b.write(
        f"p_L = ({pL[0]:.3f}, {pL[1]:.3f}) m  →  p_G = ({pG[0]:.3f}, {pG[1]:.3f}) m  \n"
        f"Inverse check: R(ψ)ᵀ(p_G − p_robot,G) = ({back[0]:.3f}, {back[1]:.3f}) m"
    )


# ------------------------------------------------------------------ sensor data
def render_sensor_data(df: pd.DataFrame) -> None:
    show_plot(sensor_signals_figure(df), "fig_sensors")
    show_plot(gps_availability_figure(df), "fig_gps_avail")
    if df.gps_innov_x.notna().any():
        show_plot(innovation_figure(df), "fig_innov")
    with st.expander("Raw simulation data"):
        show_table(df)
        st.download_button(
            "Download simulation data (CSV)", df.to_csv(index=False).encode("utf-8"),
            file_name="simulation_data.csv", mime="text/csv",
        )


# ------------------------------------------------------------------- scenarios
def render_scenarios(base_cfg: SimConfig | None, config_error: str | None) -> None:
    st.write(
        "Each scenario applies a set of sensor settings on top of the current sidebar configuration "
        "(trajectory, duration, time step and seed are shared) and runs a complete simulation."
    )
    titles = {k: s.title for k, s in SCENARIOS.items()}
    chosen = st.multiselect(
        "Scenarios to compare", list(SCENARIOS), default=list(SCENARIOS),
        format_func=lambda k: titles[k], key="ui.scenario_select",
    )
    for k in chosen:
        st.caption(f"**{titles[k]}** - {SCENARIOS[k].description}")
    if st.button("Run selected scenarios", type="primary", disabled=base_cfg is None or not chosen):
        with st.spinner("Running scenarios..."):
            st.session_state["scenario_sims"] = run_scenarios(chosen, base_cfg)
            st.session_state["scenario_base"] = base_cfg.to_dict()
    if base_cfg is None:
        st.error(config_error or "Invalid configuration.")
    sims = st.session_state.get("scenario_sims")
    if not sims:
        st.info("Press **Run selected scenarios** to generate the comparison.")
        return
    if st.session_state.get("scenario_base") != (base_cfg.to_dict() if base_cfg else None):
        st.info("These results were produced with an earlier sidebar configuration. Re-run to refresh.")

    rows = []
    for k, s in sims.items():
        t = compare_estimators(s.to_dataframe())
        rows.append({
            "Scenario": titles[k],
            "Odometry RMSE [m]": t.iloc[0, 1], "EKF RMSE [m]": t.iloc[1, 1],
            "Odometry final [m]": t.iloc[0, 3], "EKF final [m]": t.iloc[1, 3],
            "Odometry max [m]": t.iloc[0, 2], "EKF max [m]": t.iloc[1, 2],
            "Odometry yaw [deg]": t.iloc[0, 4], "EKF yaw [deg]": t.iloc[1, 4],
        })
    summary = pd.DataFrame(rows).set_index("Scenario")
    show_table(summary.style.format("{:.3f}"))
    which = st.radio("Estimator to display", ["ekf", "odo"], horizontal=True, key="ui.scn_est",
                     format_func=lambda e: "EKF estimate" if e == "ekf" else "Odometry-only estimate")
    show_plot(scenario_trajectories_figure(sims, titles, which), "fig_scn_traj")
    show_plot(scenario_errors_figure(sims, titles, which), "fig_scn_err")


# -------------------------------------------------------------------- animation
def render_animation(sim: Simulation, df: pd.DataFrame) -> None:
    if len(df) < 2:
        st.info("Run at least one simulation step to create an animation.")
        return
    st.write("Replay of the simulation data produced so far. Playback runs in the browser.")
    show_plot(animation_figure(df, boundary_from_settings(sim.config.simulation)), "fig_anim")
