"""Interactive Plotly trajectory comparison and coordinate-frame figures."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.coordinate_frames import local_to_global, local_vector_to_global, rotation_matrix

# Colour scheme shared by all figures
COLOR_TRUTH = "#16a34a"  # green
COLOR_ODOM = "#ea580c"  # orange
COLOR_EKF = "#2563eb"  # blue
COLOR_GPS = "#9333ea"  # purple
COLOR_BOUNDARY = "#6b7280"  # grey

_ROBOT_BODY_LOCAL = np.array([[1.0, 0.0], [-0.7, 0.55], [-0.4, 0.0], [-0.7, -0.55], [1.0, 0.0]])


def boundary_from_settings(sim_settings) -> dict[str, float] | None:
    """Boundary dict from ``SimulationSettings`` or ``None`` when disabled."""
    if not sim_settings.boundary_enabled:
        return None
    return {
        "xmin": sim_settings.boundary_xmin,
        "xmax": sim_settings.boundary_xmax,
        "ymin": sim_settings.boundary_ymin,
        "ymax": sim_settings.boundary_ymax,
    }


def robot_outline(x: float, y: float, psi: float, size: float) -> np.ndarray:
    """Outline of the robot body (an arrow-head) in global coordinates, via LCSF -> GCSF."""
    return local_to_global(_ROBOT_BODY_LOCAL * size, [x, y], psi)


def _auto_size(df: pd.DataFrame) -> float:
    xs = np.concatenate([df.x_true.to_numpy(), df.x_odo.to_numpy(), df.x_ekf.to_numpy()])
    ys = np.concatenate([df.y_true.to_numpy(), df.y_odo.to_numpy(), df.y_ekf.to_numpy()])
    extent = max(np.ptp(xs), np.ptp(ys), 1.0)
    return 0.04 * extent


def covariance_ellipse(x: float, y: float, P_xy: np.ndarray, n_sigma: float = 2.0) -> np.ndarray:
    """Points of the n-sigma ellipse of a 2x2 position covariance."""
    P = 0.5 * (P_xy + P_xy.T)
    vals, vecs = np.linalg.eigh(P)
    vals = np.clip(vals, 0.0, None)
    th = np.linspace(0, 2 * np.pi, 60)
    circle = np.stack([np.cos(th), np.sin(th)])
    pts = vecs @ (n_sigma * np.sqrt(vals)[:, None] * circle)
    return np.stack([x + pts[0], y + pts[1]], axis=1)


def _add_boundary(fig: go.Figure, boundary: Mapping[str, float] | None) -> None:
    if not boundary:
        return
    b = boundary
    fig.add_trace(
        go.Scatter(
            x=[b["xmin"], b["xmax"], b["xmax"], b["xmin"], b["xmin"]],
            y=[b["ymin"], b["ymin"], b["ymax"], b["ymax"], b["ymin"]],
            mode="lines",
            line=dict(color=COLOR_BOUNDARY, dash="dash", width=1.5),
            name="Environment boundary",
            hoverinfo="skip",
        )
    )


def _base_layout(fig: go.Figure, title: str, height: int = 620) -> None:
    fig.update_layout(
        title=title,
        xaxis=dict(title="X global [m]", zeroline=True),
        yaxis=dict(title="Y global [m]", scaleanchor="x", scaleratio=1, zeroline=True),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=height,
        margin=dict(l=10, r=10, t=70, b=10),
        template="plotly_white",
        uirevision="trajectory",  # keep zoom/pan across Streamlit reruns
    )


def trajectory_figure(
    df: pd.DataFrame,
    step: int | None = None,
    boundary: Mapping[str, float] | None = None,
    show_gps: bool = True,
    show_ekf_ellipse: bool = False,
    covariance_xy: np.ndarray | None = None,
) -> go.Figure:
    """Ground truth vs odometry-only vs EKF, GPS fixes and the current robot pose.

    ``step`` selects the current row (default: last row); only data up to it is drawn.
    """
    k = len(df) - 1 if step is None else int(np.clip(step, 0, len(df) - 1))
    d = df.iloc[: k + 1]
    fig = go.Figure()
    _add_boundary(fig, boundary)

    def line(xc: str, yc: str, name: str, color: str, width: float, dash: str | None = None):
        fig.add_trace(
            go.Scatter(
                x=d[xc], y=d[yc], mode="lines", name=name,
                line=dict(color=color, width=width, dash=dash),
                customdata=np.stack([d.t, d[xc.replace("x_", "psi_")]], axis=1),
                hovertemplate=f"<b>{name}</b><br>t=%{{customdata[0]:.2f}} s<br>"
                "x=%{x:.3f} m<br>y=%{y:.3f} m<br>yaw=%{customdata[1]:.3f} rad<extra></extra>",
            )
        )

    line("x_true", "y_true", "Ground truth", COLOR_TRUTH, 3)
    line("x_odo", "y_odo", "Odometry-only", COLOR_ODOM, 2, "dot")
    line("x_ekf", "y_ekf", "EKF fused", COLOR_EKF, 2.5)

    if show_gps:
        g = d.dropna(subset=["gps_x"])
        fig.add_trace(
            go.Scatter(
                x=g.gps_x, y=g.gps_y, mode="markers", name="GPS measurements",
                marker=dict(color=COLOR_GPS, size=8, symbol="x", line=dict(width=1)),
                customdata=g.t,
                hovertemplate="<b>GPS fix</b><br>t=%{customdata:.2f} s<br>x=%{x:.3f} m<br>y=%{y:.3f} m<extra></extra>",
            )
        )

    size = _auto_size(df)
    row = d.iloc[-1]
    for name, xc, yc, pc, color, dash in (
        ("Robot (truth)", "x_true", "y_true", "psi_true", COLOR_TRUTH, None),
        ("Estimate (EKF)", "x_ekf", "y_ekf", "psi_ekf", COLOR_EKF, "solid"),
    ):
        pts = robot_outline(row[xc], row[yc], row[pc], size * (1.0 if dash is None else 0.7))
        fig.add_trace(
            go.Scatter(
                x=pts[:, 0], y=pts[:, 1], mode="lines", fill="toself", name=name,
                line=dict(color=color, width=2), fillcolor=color, opacity=0.55 if dash is None else 0.35,
                hovertemplate=f"<b>{name}</b><br>t={row.t:.2f} s<br>yaw={row[pc]:.3f} rad<extra></extra>",
            )
        )
    # Heading indicator of the true robot
    tip = local_to_global([[0.0, 0.0], [2.6 * size, 0.0]], [row.x_true, row.y_true], row.psi_true)
    fig.add_trace(
        go.Scatter(
            x=tip[:, 0], y=tip[:, 1], mode="lines", name="Heading",
            line=dict(color="black", width=2), hoverinfo="skip",
        )
    )

    if show_ekf_ellipse and covariance_xy is not None:
        ell = covariance_ellipse(row.x_ekf, row.y_ekf, covariance_xy)
        fig.add_trace(
            go.Scatter(
                x=ell[:, 0], y=ell[:, 1], mode="lines", name="EKF 2σ position ellipse",
                line=dict(color=COLOR_EKF, width=1, dash="dash"), hoverinfo="skip",
            )
        )

    _base_layout(fig, f"Trajectory comparison  (t = {row.t:.2f} s)")
    return fig


def coordinate_frame_figure(
    robot_xy: tuple[float, float], psi: float, local_vector: tuple[float, float]
) -> go.Figure:
    """Global axes, robot-local axes, heading and a local vector mapped into the global frame."""
    rx, ry = robot_xy
    pL = np.asarray(local_vector, dtype=float)
    R = rotation_matrix(psi)
    axis_len = 1.5
    pG = local_to_global(pL, [rx, ry], psi)
    dG = local_vector_to_global(pL, psi)  # displacement in global axes

    fig = go.Figure()

    def arrow(x0, y0, x1, y1, color, label, width=3, dash=None):
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=width, arrowcolor=color, text="",
        )
        fig.add_trace(
            go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines+text", text=["", label], textposition="top right",
                line=dict(color=color, width=width, dash=dash), name=label, hoverinfo="name",
                textfont=dict(color=color, size=14),
            )
        )

    arrow(0, 0, axis_len, 0, "#111827", "X_G")
    arrow(0, 0, 0, axis_len, "#4b5563", "Y_G")
    ex, ey = R[:, 0] * axis_len, R[:, 1] * axis_len
    arrow(rx, ry, rx + ex[0], ry + ex[1], COLOR_TRUTH, "X_L (heading)")
    arrow(rx, ry, rx + ey[0], ry + ey[1], "#15803d", "Y_L")

    pts = robot_outline(rx, ry, psi, 0.25)
    fig.add_trace(
        go.Scatter(
            x=pts[:, 0], y=pts[:, 1], mode="lines", fill="toself", fillcolor=COLOR_TRUTH, opacity=0.4,
            line=dict(color=COLOR_TRUTH), name="Robot", hoverinfo="name",
        )
    )

    # Local vector: drawn as components in the local frame (dashed) and as the global result.
    mid = local_to_global([pL[0], 0.0], [rx, ry], psi)
    fig.add_trace(
        go.Scatter(
            x=[rx, mid[0], pG[0]], y=[ry, mid[1], pG[1]], mode="lines", name="p_L components (along X_L, Y_L)",
            line=dict(color=COLOR_ODOM, dash="dot", width=2), hoverinfo="name",
        )
    )
    arrow(rx, ry, pG[0], pG[1], COLOR_ODOM, "p_G = p_robot_G + R·p_L", width=3)
    fig.add_trace(
        go.Scatter(
            x=[0, rx], y=[0, ry], mode="lines", name="p_robot_G",
            line=dict(color="#6b7280", dash="dash", width=1.5), hoverinfo="name",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[pG[0]], y=[pG[1]], mode="markers", name="Transformed point",
            marker=dict(color=COLOR_ODOM, size=10),
            hovertemplate=f"p_L=({pL[0]:.2f}, {pL[1]:.2f})<br>p_G=({pG[0]:.2f}, {pG[1]:.2f})<extra></extra>",
        )
    )
    fig.add_annotation(
        x=rx + 0.3 * np.cos(psi + 0.4), y=ry + 0.3 * np.sin(psi + 0.4), text=f"ψ = {np.degrees(psi):.0f}°",
        showarrow=False, font=dict(size=13),
    )
    # View window covering the global origin, the robot, its axes and the transformed point.
    xs = [0.0, axis_len, rx - axis_len, rx + axis_len, pG[0]]
    ys = [0.0, axis_len, ry - axis_len, ry + axis_len, pG[1]]
    half = 0.5 * max(max(xs) - min(xs), max(ys) - min(ys)) + 0.6
    cx, cy = 0.5 * (max(xs) + min(xs)), 0.5 * (max(ys) + min(ys))
    fig.update_layout(
        title="Global (X_G, Y_G) and robot-local (X_L, Y_L) coordinate frames",
        xaxis=dict(title="X global [m]", zeroline=False, range=[cx - half, cx + half]),
        yaxis=dict(title="Y global [m]", scaleanchor="x", scaleratio=1, zeroline=False, range=[cy - half, cy + half]),
        template="plotly_white", height=560, showlegend=False,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig
