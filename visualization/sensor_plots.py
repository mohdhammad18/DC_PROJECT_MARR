"""Plotly figures for errors, covariance, sensor signals and GPS/innovation diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.sensors import GPS_DISABLED, GPS_DROPOUT

from .trajectory import COLOR_EKF, COLOR_GPS, COLOR_ODOM, COLOR_TRUTH, title_and_legend_on_top

_LAYOUT = dict(template="plotly_white", uirevision="plots")


def _legend_top(fig: go.Figure) -> None:
    """Title top-left, legend (may wrap to several rows) between the title and the plot."""
    title_and_legend_on_top(fig)


def _gps_fix_marks(fig: go.Figure, d: pd.DataFrame, row: int | None = None, col: int | None = None) -> None:
    """Vertical dotted markers at times where a GPS correction was applied."""
    for t in d.t[d.gps_innov_x.notna()]:
        kw = {"row": row, "col": col} if row else {}
        fig.add_vline(x=t, line=dict(color=COLOR_GPS, width=0.5, dash="dot"), opacity=0.35, **kw)


def position_error_figure(df: pd.DataFrame, step: int | None = None, mark_gps: bool = True) -> go.Figure:
    d = df if step is None else df.iloc[: step + 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.t, y=d.err_odo, name="Odometry-only", line=dict(color=COLOR_ODOM, width=2)))
    fig.add_trace(go.Scatter(x=d.t, y=d.err_ekf, name="EKF fused", line=dict(color=COLOR_EKF, width=2)))
    if mark_gps and len(d) < 4000:
        _gps_fix_marks(fig, d)
    fig.update_layout(
        title="Position error  e_p = √((x_true − x_est)² + (y_true − y_est)²)",
        xaxis_title="Time [s]", yaxis_title="Position error [m]", height=360, **_LAYOUT,
    )
    _legend_top(fig)
    return fig


def yaw_error_figure(df: pd.DataFrame, step: int | None = None) -> go.Figure:
    d = df if step is None else df.iloc[: step + 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.t, y=np.degrees(d.yaw_err_odo), name="Odometry-only", line=dict(color=COLOR_ODOM, width=2)))
    fig.add_trace(go.Scatter(x=d.t, y=np.degrees(d.yaw_err_ekf), name="EKF fused", line=dict(color=COLOR_EKF, width=2)))
    fig.update_layout(
        title="Yaw error  wrap(ψ_true − ψ_est)",
        xaxis_title="Time [s]", yaxis_title="Yaw error [deg]", height=360, **_LAYOUT,
    )
    _legend_top(fig)
    return fig


def covariance_figure(df: pd.DataFrame, step: int | None = None) -> go.Figure:
    """EKF standard deviations sqrt(P_xx), sqrt(P_yy), sqrt(P_psipsi) over time."""
    d = df if step is None else df.iloc[: step + 1]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=d.t, y=d.sigma_x, name="σ_x [m]", line=dict(color="#1d4ed8")), secondary_y=False)
    fig.add_trace(go.Scatter(x=d.t, y=d.sigma_y, name="σ_y [m]", line=dict(color="#0891b2")), secondary_y=False)
    fig.add_trace(
        go.Scatter(x=d.t, y=np.degrees(d.sigma_psi), name="σ_ψ [deg]", line=dict(color="#be185d", dash="dash")),
        secondary_y=True,
    )
    fig.update_layout(title="EKF covariance evolution (square roots of the diagonal of P)", height=360, **_LAYOUT)
    fig.update_xaxes(title_text="Time [s]")
    fig.update_yaxes(title_text="Position std [m]", secondary_y=False)
    fig.update_yaxes(title_text="Yaw std [deg]", secondary_y=True)
    _legend_top(fig)
    return fig


def sensor_signals_figure(df: pd.DataFrame, step: int | None = None) -> go.Figure:
    """Measured vs true forward speed and yaw rate."""
    d = df if step is None else df.iloc[: step + 1]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Forward speed v", "Yaw rate ω"))
    fig.add_trace(go.Scatter(x=d.t, y=d.v_meas, name="v wheel odometry", line=dict(color=COLOR_ODOM, width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.t, y=d.v_true, name="v true", line=dict(color=COLOR_TRUTH, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.t, y=d.omega_wheel, name="ω wheel-derived", line=dict(color=COLOR_ODOM, width=1)), row=2, col=1)
    if d.gyro_yaw_rate.notna().any():
        fig.add_trace(go.Scatter(x=d.t, y=d.gyro_yaw_rate, name="ω gyro", line=dict(color="#7c3aed", width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=d.t, y=d.omega_true, name="ω true", line=dict(color=COLOR_TRUTH, width=2)), row=2, col=1)
    fig.update_yaxes(title_text="m/s", row=1, col=1)
    fig.update_yaxes(title_text="rad/s", row=2, col=1)
    fig.update_xaxes(title_text="Time [s]", row=2, col=1)
    fig.update_layout(title="Wheel odometry and gyro measurements", height=520, **_LAYOUT)
    _legend_top(fig)
    return fig


_STATUS_STYLE = {
    "fix": ("GPS fix (used by EKF)", COLOR_GPS, "circle"),
    GPS_DROPOUT: ("Scheduled fix lost", "#dc2626", "x"),
    "update_failed": ("Update rejected (numerical)", "#000000", "diamond"),
}


def gps_availability_figure(df: pd.DataFrame, step: int | None = None) -> go.Figure:
    """Timeline of GPS events (fixes, dropouts, rejected updates). Idle steps are omitted."""
    d = df if step is None else df.iloc[: step + 1]
    fig = go.Figure()
    if (d.gps_status == GPS_DISABLED).all():
        fig.add_annotation(text="GPS is disabled in this simulation (GPS-denied).", showarrow=False, x=0.5, y=0.5,
                           xref="paper", yref="paper", font=dict(size=14))
    for status, (label, color, symbol) in _STATUS_STYLE.items():
        s = d[d.gps_status == status]
        fig.add_trace(go.Scatter(x=s.t, y=[label] * len(s), mode="markers", name=label,
                                 marker=dict(color=color, symbol=symbol, size=9)))
    fig.update_layout(title="GPS availability", xaxis_title="Time [s]", height=240, **_LAYOUT)
    _legend_top(fig)
    return fig


def innovation_figure(df: pd.DataFrame, step: int | None = None) -> go.Figure:
    """GPS innovations r = z − H x⁻ at the update times."""
    d = df if step is None else df.iloc[: step + 1]
    g = d[d.gps_innov_x.notna()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.t, y=g.gps_innov_x, mode="markers+lines", name="innovation x [m]", line=dict(width=1, color=COLOR_EKF)))
    fig.add_trace(go.Scatter(x=g.t, y=g.gps_innov_y, mode="markers+lines", name="innovation y [m]", line=dict(width=1, color=COLOR_ODOM)))
    fig.update_layout(title="GPS innovation at each update", xaxis_title="Time [s]", yaxis_title="Innovation [m]", height=320, **_LAYOUT)
    _legend_top(fig)
    return fig


def scenario_trajectories_figure(sims: dict[str, "object"], titles: dict[str, str], estimator: str = "ekf") -> go.Figure:
    """Overlay of the true path and the chosen estimator for several scenarios."""
    palette = ["#2563eb", "#ea580c", "#9333ea", "#0891b2", "#be185d", "#65a30d"]
    fig = go.Figure()
    first = next(iter(sims.values())).to_dataframe()
    fig.add_trace(go.Scatter(x=first.x_true, y=first.y_true, name="Ground truth",
                             line=dict(color="black", width=3)))
    for (key, sim), color in zip(sims.items(), palette):
        d = sim.to_dataframe()
        fig.add_trace(go.Scatter(x=d[f"x_{estimator}"], y=d[f"y_{estimator}"], name=titles[key],
                                 line=dict(color=color, width=2)))
    fig.update_layout(
        title=f"Scenario comparison - {'EKF estimate' if estimator == 'ekf' else 'odometry-only estimate'}",
        xaxis=dict(title="X global [m]"), yaxis=dict(title="Y global [m]", scaleanchor="x", scaleratio=1),
        height=560, hovermode="closest", **_LAYOUT,
    )
    _legend_top(fig)
    return fig


def scenario_errors_figure(sims: dict[str, "object"], titles: dict[str, str], estimator: str = "ekf") -> go.Figure:
    palette = ["#2563eb", "#ea580c", "#9333ea", "#0891b2", "#be185d", "#65a30d"]
    fig = go.Figure()
    for (key, sim), color in zip(sims.items(), palette):
        d = sim.to_dataframe()
        fig.add_trace(go.Scatter(x=d.t, y=d[f"err_{estimator}"], name=titles[key], line=dict(color=color, width=2)))
    fig.update_layout(
        title=f"Position error over time - {'EKF estimate' if estimator == 'ekf' else 'odometry-only estimate'}",
        xaxis_title="Time [s]", yaxis_title="Position error [m]", height=400, **_LAYOUT,
    )
    _legend_top(fig)
    return fig
