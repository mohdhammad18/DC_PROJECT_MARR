"""Differential-drive robot kinematics and ground-truth motion.

Pose state: ``[x, y, psi]`` in the global frame (planar: z = roll = pitch = 0).

Kinematics (alpha_L / alpha_R are wheel angular velocities in rad/s)::

    v     = (r / 2) * (alpha_R + alpha_L)
    omega = (r / b) * (alpha_R - alpha_L)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .coordinate_frames import wrap_angle


@dataclass(frozen=True)
class RobotParams:
    """Geometric parameters of the differential-drive robot."""

    wheel_radius: float = 0.05  # r [m]
    wheel_separation: float = 0.30  # b [m]

    def __post_init__(self) -> None:
        for name in ("wheel_radius", "wheel_separation"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite number, got {value!r}.")


def wheel_speeds_to_body_velocity(
    alpha_l: ArrayLike, alpha_r: ArrayLike, params: RobotParams
) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
    """Forward kinematics: wheel angular velocities -> (v, omega)."""
    r, b = params.wheel_radius, params.wheel_separation
    al = np.asarray(alpha_l, dtype=float)
    ar = np.asarray(alpha_r, dtype=float)
    v = 0.5 * r * (ar + al)
    omega = (r / b) * (ar - al)
    if v.ndim == 0:
        return float(v), float(omega)
    return v, omega


def body_velocity_to_wheel_speeds(
    v: ArrayLike, omega: ArrayLike, params: RobotParams
) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
    """Inverse kinematics: (v, omega) -> (alpha_L, alpha_R)."""
    r, b = params.wheel_radius, params.wheel_separation
    v_a = np.asarray(v, dtype=float)
    w_a = np.asarray(omega, dtype=float)
    alpha_r = (v_a + 0.5 * b * w_a) / r
    alpha_l = (v_a - 0.5 * b * w_a) / r
    if alpha_r.ndim == 0:
        return float(alpha_l), float(alpha_r)
    return alpha_l, alpha_r


def integrate_pose_exact(
    pose: ArrayLike, v: float, omega: float, dt: float
) -> NDArray[np.float64]:
    """Integrate a constant (v, omega) twist over ``dt`` along its exact circular arc.

    For |omega| > 0 the robot moves on a circle of radius v/omega; for
    omega -> 0 the arc degenerates to a straight line, which is evaluated with
    the mid-point heading to avoid dividing by a vanishing omega.
    """
    x, y, psi = (float(c) for c in np.asarray(pose, dtype=float).reshape(3))
    dpsi = omega * dt
    if abs(dpsi) < 1e-6:
        heading = psi + 0.5 * dpsi
        x += v * dt * np.cos(heading)
        y += v * dt * np.sin(heading)
    else:
        radius = v / omega
        x += radius * (np.sin(psi + dpsi) - np.sin(psi))
        y += radius * (np.cos(psi) - np.cos(psi + dpsi))
    return np.array([x, y, wrap_angle(psi + dpsi)], dtype=float)


class DifferentialDriveRobot:
    """Ground-truth differential-drive robot driven by wheel angular velocities."""

    def __init__(
        self,
        params: RobotParams,
        initial_pose: ArrayLike = (0.0, 0.0, 0.0),
    ) -> None:
        self.params = params
        pose = np.asarray(initial_pose, dtype=float).reshape(-1)
        if pose.size != 3 or not np.all(np.isfinite(pose)):
            raise ValueError("initial_pose must be three finite values [x, y, psi].")
        pose[2] = wrap_angle(pose[2])
        self._pose = pose.copy()
        self.v = 0.0
        self.omega = 0.0

    @property
    def pose(self) -> NDArray[np.float64]:
        """Current pose ``[x, y, psi]`` (a copy)."""
        return self._pose.copy()

    def step(self, alpha_l: float, alpha_r: float, dt: float) -> tuple[float, float]:
        """Advance the robot by ``dt`` with constant wheel speeds. Returns (v, omega)."""
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be positive and finite, got {dt!r}.")
        if not (np.isfinite(alpha_l) and np.isfinite(alpha_r)):
            raise ValueError("Wheel angular velocities must be finite.")
        v, omega = wheel_speeds_to_body_velocity(alpha_l, alpha_r, self.params)
        self._pose = integrate_pose_exact(self._pose, v, omega, dt)
        self.v, self.omega = v, omega
        return v, omega
