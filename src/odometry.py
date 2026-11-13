"""Dead reckoning: integrate measured (v, omega) without any absolute correction.

The same discrete motion model is used by the odometry-only estimator and by
the EKF prediction step, so the two estimators differ *only* by the GPS /
heading corrections.

Motion models (``dt`` = time step)::

    euler     x+ = x + v cos(psi) dt            (the model given in the brief)
    midpoint  x+ = x + v cos(psi + omega dt / 2) dt   (second-order accurate)
    psi+ = psi + omega dt  (wrapped to [-pi, pi))
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .coordinate_frames import wrap_angle

INTEGRATION_METHODS: tuple[str, ...] = ("euler", "midpoint")


def validate_integration_method(method: str) -> str:
    if method not in INTEGRATION_METHODS:
        raise ValueError(
            f"Unknown integration method {method!r}; choose one of {INTEGRATION_METHODS}."
        )
    return method


def propagate_pose(
    pose: ArrayLike, v: float, omega: float, dt: float, method: str = "euler"
) -> NDArray[np.float64]:
    """Propagate ``[x, y, psi]`` one step with the given body velocities."""
    validate_integration_method(method)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive and finite, got {dt!r}.")
    if not (np.isfinite(v) and np.isfinite(omega)):
        raise ValueError("Velocities passed to the motion model must be finite.")
    x, y, psi = (float(c) for c in np.asarray(pose, dtype=float).reshape(3))
    heading = psi if method == "euler" else psi + 0.5 * omega * dt
    return np.array(
        [
            x + v * np.cos(heading) * dt,
            y + v * np.sin(heading) * dt,
            wrap_angle(psi + omega * dt),
        ],
        dtype=float,
    )


class DeadReckoning:
    """Odometry-only pose estimator (no GPS, no heading corrections)."""

    def __init__(self, initial_pose: ArrayLike, method: str = "euler") -> None:
        pose = np.asarray(initial_pose, dtype=float).reshape(-1)
        if pose.size != 3 or not np.all(np.isfinite(pose)):
            raise ValueError("initial_pose must be three finite values [x, y, psi].")
        pose[2] = wrap_angle(pose[2])
        self._pose = pose.copy()
        self.method = validate_integration_method(method)

    @property
    def pose(self) -> NDArray[np.float64]:
        return self._pose.copy()

    def update(self, v: float, omega: float, dt: float) -> NDArray[np.float64]:
        """Integrate one step of measured velocities and return the new pose."""
        self._pose = propagate_pose(self._pose, v, omega, dt, self.method)
        return self.pose
