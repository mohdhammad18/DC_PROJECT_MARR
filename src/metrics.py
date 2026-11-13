"""Error metrics comparing estimated poses with the ground truth."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .coordinate_frames import wrap_angle


def position_error(
    x_true: ArrayLike, y_true: ArrayLike, x_est: ArrayLike, y_est: ArrayLike
) -> NDArray[np.float64]:
    """Euclidean position error ``e_p = sqrt((x_true - x_est)^2 + (y_true - y_est)^2)``."""
    dx = np.asarray(x_true, dtype=float) - np.asarray(x_est, dtype=float)
    dy = np.asarray(y_true, dtype=float) - np.asarray(y_est, dtype=float)
    return np.hypot(dx, dy)


def yaw_error(psi_true: ArrayLike, psi_est: ArrayLike) -> NDArray[np.float64]:
    """Signed yaw error wrapped to [-pi, pi): ``wrap(psi_true - psi_est)``."""
    return np.asarray(wrap_angle(np.asarray(psi_true, dtype=float) - np.asarray(psi_est, dtype=float)))


def summarize_errors(pos_err: ArrayLike, yaw_err: ArrayLike) -> dict[str, float]:
    """Summary statistics of a position-error series and a (signed) yaw-error series."""
    p = np.asarray(pos_err, dtype=float)
    y = np.abs(np.asarray(yaw_err, dtype=float))
    if p.size == 0 or y.size == 0:
        raise ValueError("Cannot summarise empty error series.")
    return {
        "mean_position_error_m": float(np.mean(p)),
        "rmse_position_m": float(np.sqrt(np.mean(p**2))),
        "max_position_error_m": float(np.max(p)),
        "final_position_error_m": float(p[-1]),
        "mean_yaw_error_deg": float(np.degrees(np.mean(y))),
    }


METRIC_LABELS: dict[str, str] = {
    "mean_position_error_m": "Mean position error [m]",
    "rmse_position_m": "RMSE position [m]",
    "max_position_error_m": "Max position error [m]",
    "final_position_error_m": "Final position error [m]",
    "mean_yaw_error_deg": "Mean yaw error [deg]",
}


def compare_estimators(df: pd.DataFrame, upto: int | None = None) -> pd.DataFrame:
    """Metrics table (rows: odometry-only / EKF) for a simulation DataFrame.

    ``upto`` limits the evaluation to rows ``0..upto`` (inclusive).
    """
    d = df if upto is None else df.iloc[: upto + 1]
    rows = {
        "Odometry-only": summarize_errors(d["err_odo"], d["yaw_err_odo"]),
        "EKF": summarize_errors(d["err_ekf"], d["yaw_err_ekf"]),
    }
    table = pd.DataFrame(rows).T.rename(columns=METRIC_LABELS)
    return table
