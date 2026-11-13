"""Local (LCSF) and global (GCSF) coordinate-frame utilities for a planar robot.

Conventions
-----------
* The *global* frame G is fixed to the world. Its axes are X_G, Y_G.
* The *local* frame L is attached to the robot. X_L points forward along the
  heading psi, Y_L points to the robot's left. Its origin is the robot position.
* ``psi`` is measured counter-clockwise from X_G to X_L.

Points and free vectors transform differently:

* A **point** ``p_L`` (a location expressed in the robot frame) maps to
  ``p_G = p_robot_G + R(psi) @ p_L``.
* A **free vector** (velocity, displacement) has no origin, so only the
  rotation applies: ``v_G = R(psi) @ v_L``.

Both variants are provided, together with their inverses, so that local and
global quantities are never mixed up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

TWO_PI: float = 2.0 * np.pi


def wrap_angle(angle: ArrayLike) -> float | NDArray[np.float64]:
    """Wrap an angle (or array of angles) in radians to the interval [-pi, pi).

    NaN values are propagated unchanged. Scalars are returned as ``float``.
    """
    wrapped = np.mod(np.asarray(angle, dtype=float) + np.pi, TWO_PI) - np.pi
    if wrapped.ndim == 0:
        return float(wrapped)
    return wrapped


def rotation_matrix(psi: float) -> NDArray[np.float64]:
    """Return the planar rotation matrix R(psi) mapping local axes to global axes."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s], [s, c]], dtype=float)


@dataclass(frozen=True)
class Pose2D:
    """Planar robot pose expressed in the global frame."""

    x: float
    y: float
    psi: float

    @property
    def position(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y], dtype=float)

    def as_array(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y, self.psi], dtype=float)

    @classmethod
    def from_array(cls, arr: ArrayLike) -> "Pose2D":
        a = np.asarray(arr, dtype=float).reshape(-1)
        if a.size != 3:
            raise ValueError(f"A pose needs exactly 3 values [x, y, psi], got {a.size}.")
        return cls(float(a[0]), float(a[1]), float(a[2]))


def _as_points(p: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(p, dtype=float)
    if arr.ndim not in (1, 2) or arr.shape[-1] != 2:
        raise ValueError(f"{name} must have shape (2,) or (N, 2); got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def local_to_global(
    p_local: ArrayLike, robot_position_g: ArrayLike, psi: float
) -> NDArray[np.float64]:
    """Transform point(s) from the robot-local frame to the global frame.

    ``p_G = p_robot_G + R(psi) @ p_L``
    """
    p_l = _as_points(p_local, "p_local")
    origin = _as_points(robot_position_g, "robot_position_g")
    if origin.ndim != 1:
        raise ValueError("robot_position_g must be a single position of shape (2,).")
    # Row-vector form of R @ p is p @ R.T
    return origin + p_l @ rotation_matrix(psi).T


def global_to_local(
    p_global: ArrayLike, robot_position_g: ArrayLike, psi: float
) -> NDArray[np.float64]:
    """Transform point(s) from the global frame to the robot-local frame.

    ``p_L = R(psi).T @ (p_G - p_robot_G)``
    """
    p_g = _as_points(p_global, "p_global")
    origin = _as_points(robot_position_g, "robot_position_g")
    if origin.ndim != 1:
        raise ValueError("robot_position_g must be a single position of shape (2,).")
    # Row-vector form of R.T @ d is d @ R
    return (p_g - origin) @ rotation_matrix(psi)


def local_vector_to_global(v_local: ArrayLike, psi: float) -> NDArray[np.float64]:
    """Rotate free vector(s) (velocity, displacement) from local to global axes."""
    v = _as_points(v_local, "v_local")
    return v @ rotation_matrix(psi).T


def global_vector_to_local(v_global: ArrayLike, psi: float) -> NDArray[np.float64]:
    """Rotate free vector(s) from global to local axes."""
    v = _as_points(v_global, "v_global")
    return v @ rotation_matrix(psi)


def body_velocity_to_global(v: float, psi: float) -> NDArray[np.float64]:
    """Global velocity [x_dot, y_dot] of a robot driving forward at speed ``v``.

    The body-frame velocity is [v, 0] (forward only, no lateral slip).
    """
    return local_vector_to_global([v, 0.0], psi)
