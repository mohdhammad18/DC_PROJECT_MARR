"""Client-side animated replay of a finished (or partial) simulation using Plotly frames."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.coordinate_frames import local_to_global

from .trajectory import (
    COLOR_BOUNDARY,
    COLOR_EKF,
    COLOR_GPS,
    COLOR_ODOM,
    COLOR_TRUTH,
    robot_outline,
)


def animation_figure(
    df: pd.DataFrame,
    boundary: Mapping[str, float] | None = None,
    max_frames: int = 150,
) -> go.Figure:
    """Animated replay with Play / Pause buttons and a time slider.

    Frames are sub-sampled so that at most ``max_frames`` are produced. All data
    comes from the simulation DataFrame - nothing is synthesised here.
    """
    if len(df) < 2:
        raise ValueError("Need at least two simulation rows to animate.")
    stride = max(1, int(np.ceil((len(df) - 1) / max_frames)))
    indices = list(range(0, len(df), stride))
    if indices[-1] != len(df) - 1:
        indices.append(len(df) - 1)

    extent = max(np.ptp(df.x_true), np.ptp(df.y_true), 1.0)
    size = 0.04 * extent

    def frame_traces(k: int) -> list[go.Scatter]:
        d = df.iloc[: k + 1]
        g = d.dropna(subset=["gps_x"])
        row = df.iloc[k]
        body = robot_outline(row.x_true, row.y_true, row.psi_true, size)
        tip = local_to_global([[0.0, 0.0], [2.6 * size, 0.0]], [row.x_true, row.y_true], row.psi_true)
        return [
            go.Scatter(x=d.x_true, y=d.y_true, mode="lines", line=dict(color=COLOR_TRUTH, width=3)),
            go.Scatter(x=d.x_odo, y=d.y_odo, mode="lines", line=dict(color=COLOR_ODOM, width=2, dash="dot")),
            go.Scatter(x=d.x_ekf, y=d.y_ekf, mode="lines", line=dict(color=COLOR_EKF, width=2.5)),
            go.Scatter(x=g.gps_x, y=g.gps_y, mode="markers", marker=dict(color=COLOR_GPS, size=7, symbol="x")),
            go.Scatter(
                x=body[:, 0], y=body[:, 1], mode="lines", fill="toself", fillcolor=COLOR_TRUTH,
                line=dict(color=COLOR_TRUTH), opacity=0.55,
            ),
            go.Scatter(x=tip[:, 0], y=tip[:, 1], mode="lines", line=dict(color="black", width=2)),
        ]

    names = ["Ground truth", "Odometry-only", "EKF fused", "GPS measurements", "Robot (truth)", "Heading"]
    first = frame_traces(indices[0])
    for tr, name in zip(first, names):
        tr.name = name
    data: list[go.Scatter] = list(first)
    if boundary:
        b = boundary
        data.append(
            go.Scatter(
                x=[b["xmin"], b["xmax"], b["xmax"], b["xmin"], b["xmin"]],
                y=[b["ymin"], b["ymin"], b["ymax"], b["ymax"], b["ymin"]],
                mode="lines", line=dict(color=COLOR_BOUNDARY, dash="dash"), name="Environment boundary",
            )
        )

    frames = [
        go.Frame(data=frame_traces(k), traces=list(range(6)), name=f"{df.t.iloc[k]:.2f}")
        for k in indices
    ]
    fig = go.Figure(data=data, frames=frames)

    # Fix axis ranges to the full data extent so the view does not jump while animating.
    xs = np.concatenate([df.x_true, df.x_odo, df.x_ekf])
    ys = np.concatenate([df.y_true, df.y_odo, df.y_ekf])
    if boundary:
        xs = np.concatenate([xs, [boundary["xmin"], boundary["xmax"]]])
        ys = np.concatenate([ys, [boundary["ymin"], boundary["ymax"]]])
    pad = 0.05 * max(np.ptp(xs), np.ptp(ys), 1.0)

    fig.update_layout(
        title="Animated replay (client-side; use Play / Pause or drag the slider)",
        xaxis=dict(title="X global [m]", range=[xs.min() - pad, xs.max() + pad]),
        yaxis=dict(title="Y global [m]", range=[ys.min() - pad, ys.max() + pad], scaleanchor="x", scaleratio=1),
        template="plotly_white", height=640, margin=dict(l=10, r=10, t=70, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        updatemenus=[
            dict(
                type="buttons", direction="left", x=0.0, y=-0.08, xanchor="left", yanchor="top",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))]),
                ],
            )
        ],
        sliders=[
            dict(
                x=0.12, y=-0.05, len=0.85, xanchor="left", yanchor="top",
                currentvalue=dict(prefix="t = ", suffix=" s"),
                steps=[
                    dict(method="animate", label=f.name,
                         args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
                    for f in frames
                ],
            )
        ],
    )
    return fig
