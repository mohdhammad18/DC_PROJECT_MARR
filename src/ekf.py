"""Extended Kalman Filter for planar pose ``X = [x, y, psi]^T``, implemented with NumPy only.

Prediction (inputs v, omega come from wheel odometry and/or the gyro)::

    x+   = x + v cos(psi) dt
    y+   = y + v sin(psi) dt
    psi+ = psi + omega dt
    F    = df/dX = [[1, 0, -v sin(psi) dt],
                    [0, 1,  v cos(psi) dt],
                    [0, 0,  1]]
    P-   = F P F^T + Q

GPS correction (global x, y)::

    z = [x_gps, y_gps]^T,  H = [[1, 0, 0], [0, 1, 0]]
    r = z - H x-                       (innovation)
    S = H P- H^T + R
    K = P- H^T S^-1
    x = x- + K r
    P = (I - K H) P- (I - K H)^T + K R K^T     (Joseph form; equals (I-KH)P- when K is optimal)

Optional heading correction uses H = [0, 0, 1] and a *wrapped* yaw innovation.
All intermediate quantities are returned so they can be inspected and plotted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .coordinate_frames import wrap_angle
from .odometry import propagate_pose, validate_integration_method

H_GPS: NDArray[np.float64] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
H_HEADING: NDArray[np.float64] = np.array([[0.0, 0.0, 1.0]])

_SYMMETRY_TOL = 1e-8
_PSD_TOL = 1e-9
_MAX_COND = 1e12


class NumericalError(RuntimeError):
    """Raised when an EKF update cannot be computed reliably (never ignored silently)."""


def symmetrize(M: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * (M + M.T)


def validate_covariance(
    M: ArrayLike, name: str, size: int, positive_definite: bool = False
) -> NDArray[np.float64]:
    """Check shape, finiteness, symmetry and (semi-)definiteness; return a float copy."""
    A = np.array(M, dtype=float)
    if A.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size}); got {A.shape}.")
    if not np.all(np.isfinite(A)):
        raise ValueError(f"{name} contains non-finite values.")
    scale = max(1.0, float(np.max(np.abs(A))))
    if not np.allclose(A, A.T, atol=_SYMMETRY_TOL * scale):
        raise ValueError(f"{name} must be symmetric.")
    A = symmetrize(A)
    min_eig = float(np.linalg.eigvalsh(A).min())
    if positive_definite and min_eig <= 0.0:
        raise ValueError(f"{name} must be positive definite (min eigenvalue {min_eig:.3e}).")
    if min_eig < -_PSD_TOL * scale:
        raise ValueError(f"{name} must be positive semidefinite (min eigenvalue {min_eig:.3e}).")
    return A


def motion_jacobian(
    state: ArrayLike, v: float, omega: float, dt: float, method: str = "euler"
) -> NDArray[np.float64]:
    """State-transition Jacobian F = d f / d X of the motion model."""
    validate_integration_method(method)
    psi = float(np.asarray(state, dtype=float).reshape(3)[2])
    heading = psi if method == "euler" else psi + 0.5 * omega * dt
    return np.array(
        [
            [1.0, 0.0, -v * np.sin(heading) * dt],
            [0.0, 1.0, v * np.cos(heading) * dt],
            [0.0, 0.0, 1.0],
        ]
    )


@dataclass(frozen=True)
class PredictionResult:
    """Quantities of one EKF prediction step."""

    x_prior: NDArray[np.float64]  # state before prediction
    P_prior: NDArray[np.float64]
    x_pred: NDArray[np.float64]  # predicted state x-
    P_pred: NDArray[np.float64]  # predicted covariance P-
    F: NDArray[np.float64]
    Q: NDArray[np.float64]
    v: float
    omega: float
    dt: float


@dataclass(frozen=True)
class UpdateResult:
    """Quantities of one EKF measurement update (GPS or heading)."""

    sensor: str  # "gps" | "heading"
    z: NDArray[np.float64]  # measurement
    z_pred: NDArray[np.float64]  # h(x-)
    innovation: NDArray[np.float64]  # r = z - h(x-)  (yaw component wrapped)
    H: NDArray[np.float64]
    R: NDArray[np.float64]
    S: NDArray[np.float64]  # innovation covariance
    K: NDArray[np.float64]  # Kalman gain
    x_pred: NDArray[np.float64]  # x-
    P_pred: NDArray[np.float64]  # P-
    x_corrected: NDArray[np.float64]  # x_hat
    P_updated: NDArray[np.float64]
    nis: float  # normalised innovation squared r^T S^-1 r


class ExtendedKalmanFilter:
    """EKF over ``[x, y, psi]`` with GPS and optional heading measurement updates."""

    def __init__(
        self,
        initial_state: ArrayLike,
        initial_covariance: ArrayLike,
        integration: str = "euler",
    ) -> None:
        x0 = np.array(initial_state, dtype=float).reshape(-1)
        if x0.size != 3 or not np.all(np.isfinite(x0)):
            raise ValueError("initial_state must be three finite values [x, y, psi].")
        x0[2] = wrap_angle(x0[2])
        self._x = x0
        self._P = validate_covariance(initial_covariance, "initial_covariance", 3)
        self.integration = validate_integration_method(integration)

    # ------------------------------------------------------------------ access
    @property
    def state(self) -> NDArray[np.float64]:
        return self._x.copy()

    @property
    def covariance(self) -> NDArray[np.float64]:
        return self._P.copy()

    # --------------------------------------------------------------- prediction
    def predict(self, v: float, omega: float, dt: float, Q: ArrayLike) -> PredictionResult:
        """Time update: propagate state and covariance with the motion model."""
        Qm = validate_covariance(Q, "Q", 3)
        x_prior, P_prior = self._x.copy(), self._P.copy()
        F = motion_jacobian(x_prior, v, omega, dt, self.integration)
        x_pred = propagate_pose(x_prior, v, omega, dt, self.integration)
        P_pred = symmetrize(F @ P_prior @ F.T + Qm)
        self._x, self._P = x_pred, P_pred
        return PredictionResult(
            x_prior=x_prior,
            P_prior=P_prior,
            x_pred=x_pred.copy(),
            P_pred=P_pred.copy(),
            F=F,
            Q=Qm,
            v=float(v),
            omega=float(omega),
            dt=float(dt),
        )

    # ------------------------------------------------------------------ updates
    def update_gps(self, z: ArrayLike, R: ArrayLike) -> UpdateResult:
        """Correct with a GPS position fix ``z = [x_gps, y_gps]`` (global frame)."""
        z_arr = np.array(z, dtype=float).reshape(-1)
        if z_arr.size != 2 or not np.all(np.isfinite(z_arr)):
            raise ValueError("GPS measurement must be two finite values [x, y].")
        return self._update("gps", z_arr, H_GPS, R, 2, wrap_rows=())

    def update_heading(self, z: float, R: ArrayLike) -> UpdateResult:
        """Correct with a noisy heading measurement ``z = psi_meas`` (innovation wrapped)."""
        if not np.isfinite(z):
            raise ValueError("Heading measurement must be finite.")
        return self._update("heading", np.array([float(z)]), H_HEADING, R, 1, wrap_rows=(0,))

    def _update(
        self,
        sensor: str,
        z: NDArray[np.float64],
        H: NDArray[np.float64],
        R: ArrayLike,
        m: int,
        wrap_rows: tuple[int, ...],
    ) -> UpdateResult:
        Rm = validate_covariance(R, f"R ({sensor})", m, positive_definite=True)
        x_pred, P_pred = self._x.copy(), self._P.copy()

        z_pred = H @ x_pred
        innovation = z - z_pred
        for row in wrap_rows:  # angular residuals must be wrapped to [-pi, pi)
            innovation[row] = wrap_angle(innovation[row])

        S = symmetrize(H @ P_pred @ H.T + Rm)
        if not np.all(np.isfinite(S)) or np.linalg.cond(S) > _MAX_COND:
            raise NumericalError(
                f"{sensor} update skipped: innovation covariance S is singular or ill-conditioned."
            )
        try:
            # K = P- H^T S^-1  ->  solve S K^T = H P-  (S is symmetric)
            K = np.linalg.solve(S, H @ P_pred).T
            nis = float(innovation @ np.linalg.solve(S, innovation))
        except np.linalg.LinAlgError as exc:  # pragma: no cover - guarded by cond check
            raise NumericalError(f"{sensor} update failed: {exc}") from exc

        x_new = x_pred + K @ innovation
        x_new[2] = wrap_angle(x_new[2])
        I_KH = np.eye(3) - K @ H
        P_new = symmetrize(I_KH @ P_pred @ I_KH.T + K @ Rm @ K.T)  # Joseph form
        if not (np.all(np.isfinite(x_new)) and np.all(np.isfinite(P_new))):
            raise NumericalError(f"{sensor} update produced non-finite values.")

        self._x, self._P = x_new, P_new
        return UpdateResult(
            sensor=sensor,
            z=z.copy(),
            z_pred=z_pred,
            innovation=innovation,
            H=H.copy(),
            R=Rm,
            S=S,
            K=K,
            x_pred=x_pred,
            P_pred=P_pred,
            x_corrected=x_new.copy(),
            P_updated=P_new.copy(),
            nis=nis,
        )
