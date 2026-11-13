"""Motion-command generators for the predefined trajectories.

A trajectory is a sequence of constant ``(v, omega)`` commands, one per
simulation step. The engine converts them to wheel speeds (inverse kinematics)
and drives the ground-truth robot with the resulting wheel speeds.

Durations are quantised to whole time steps and the commanded values are
adjusted so geometric shapes close exactly (e.g. every square corner turns by
exactly 90 degrees, every figure-eight loop by exactly 360 degrees).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import TRAJECTORY_NAMES, TrajectoryConfig

Segment = tuple[int, float, float]  # (number of steps, v, omega)


def _expand(segments: list[Segment], n_steps: int, cycle: bool) -> tuple[NDArray, NDArray]:
    """Expand segments to per-step arrays; repeat if ``cycle``, otherwise stop (v = omega = 0)."""
    v = np.zeros(n_steps)
    w = np.zeros(n_steps)
    k = 0
    while k < n_steps:
        for n, vs, ws in segments:
            m = min(n, n_steps - k)
            v[k : k + m] = vs
            w[k : k + m] = ws
            k += m
            if k >= n_steps:
                break
        if not cycle:
            break
    return v, w


def _square_segments(cfg: TrajectoryConfig, dt: float) -> list[Segment]:
    n_side = max(1, int(round(cfg.side_length / cfg.speed / dt)))
    v_side = cfg.side_length / (n_side * dt)
    n_turn = max(1, int(round((np.pi / 2) / cfg.turn_rate / dt)))
    w_turn = (np.pi / 2) / (n_turn * dt)
    return [(n_side, v_side, 0.0), (n_turn, 0.0, w_turn)] * 4


def _figure_eight_segments(cfg: TrajectoryConfig, dt: float) -> list[Segment]:
    n_loop = max(1, int(round(2 * np.pi * cfg.radius / cfg.speed / dt)))
    w = 2 * np.pi / (n_loop * dt)
    v = cfg.radius * w
    return [(n_loop, v, +w), (n_loop, v, -w)]


def generate_commands(cfg: TrajectoryConfig, n_steps: int, dt: float) -> tuple[NDArray, NDArray]:
    """Return per-step commanded ``(v, omega)`` arrays of length ``n_steps``."""
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")
    if cfg.name not in TRAJECTORY_NAMES:
        raise ValueError(f"Unknown trajectory {cfg.name!r}; choose one of {TRAJECTORY_NAMES}.")
    if cfg.name == "straight":
        return _expand([(n_steps, cfg.speed, 0.0)], n_steps, cycle=False)
    if cfg.name == "circle":
        return _expand([(n_steps, cfg.speed, cfg.speed / cfg.radius)], n_steps, cycle=False)
    if cfg.name == "square":
        return _expand(_square_segments(cfg, dt), n_steps, cycle=True)
    if cfg.name == "figure_eight":
        return _expand(_figure_eight_segments(cfg, dt), n_steps, cycle=True)
    # custom
    segs = [
        (max(1, int(round(float(s["duration"]) / dt))), float(s["v"]), float(s["omega"]))
        for s in cfg.custom_segments
    ]
    return _expand(segs, n_steps, cycle=False)
