"""Simulation engine: ground truth, sensors, odometry-only estimate and EKF on one time axis.

Per step ``k -> k+1`` (time ``t = (k+1) dt``):

1. The ground-truth robot moves with the commanded wheel speeds (exact arc).
2. Wheel encoders and the gyro are sampled -> measured ``(v, omega)``.
3. The odometry-only estimator integrates ``(v, omega)`` (never uses GPS).
4. The EKF predicts with the *same* ``(v, omega)``.
5. If a GPS fix is available the EKF applies the GPS correction; an optional
   noisy IMU heading may correct yaw as well.
6. Everything is stored; row ``k`` of the result belongs to time ``k * dt``.

Failed or skipped sensor updates (GPS dropouts, numerical failures) are recorded
in ``gps_status`` / ``warnings`` and logged - never swallowed silently.
This module does not import Streamlit or Plotly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .config import SimConfig
from .ekf import ExtendedKalmanFilter, NumericalError, PredictionResult, UpdateResult
from .metrics import position_error, yaw_error
from .odometry import DeadReckoning
from .robot import (
    DifferentialDriveRobot,
    RobotParams,
    body_velocity_to_wheel_speeds,
    wheel_speeds_to_body_velocity,
)
from .sensors import (
    GPS_FIX,
    GPSSensor,
    IMUSensor,
    WheelOdometrySensor,
    is_scheduled,
    make_rngs,
)
from .trajectories import generate_commands

logger = logging.getLogger(__name__)

GPS_UPDATE_FAILED = "update_failed"
MIN_GPS_R_STD = 0.01  # [m] floor so R stays positive definite for "ideal" sensors
MIN_HEADING_R_STD = 1e-3  # [rad]


class SimulationFinished(RuntimeError):
    """Raised when ``step()`` is called after the last time step."""


@dataclass(frozen=True)
class EKFStepRecord:
    """EKF internals of one time step (for the educational visualisation)."""

    step: int
    time: float
    prediction: PredictionResult
    gps_update: UpdateResult | None = None
    heading_update: UpdateResult | None = None


@dataclass
class SimulationResult:
    """Immutable-ish snapshot of a (possibly partial) run."""

    config: SimConfig
    data: pd.DataFrame
    ekf_records: list[EKFStepRecord | None]
    warnings: list[str] = field(default_factory=list)


class Simulation:
    """Step-able simulation engine. Deterministic for a given configuration and seed."""

    def __init__(self, config: SimConfig) -> None:
        config.validate()
        self.config = config
        self.reset()

    # ----------------------------------------------------------------- setup
    def reset(self) -> None:
        """Rebuild all components and return to t = 0 (same seed -> same run)."""
        cfg = self.config
        sim, rb = cfg.simulation, cfg.robot
        self.dt = sim.dt
        self.n_steps = cfg.n_steps
        self.params = RobotParams(rb.wheel_radius, rb.wheel_separation)

        v_cmd, w_cmd = generate_commands(cfg.trajectory, self.n_steps, self.dt)
        self._v_cmd, self._w_cmd = v_cmd, w_cmd
        self._alpha_l, self._alpha_r = body_velocity_to_wheel_speeds(v_cmd, w_cmd, self.params)

        rngs = make_rngs(sim.seed)
        self.wheel_sensor = WheelOdometrySensor(
            cfg.odometry.wheel_noise_std, cfg.odometry.scale_error, rngs["wheel"]
        )
        self.imu_sensor = IMUSensor(
            cfg.imu.gyro_noise_std,
            cfg.imu.gyro_bias,
            cfg.imu.bias_random_walk_std,
            cfg.imu.heading_noise_std,
            rngs["imu"],
        )
        self.gps_sensor = GPSSensor(
            cfg.gps.enabled,
            cfg.gps.noise_std,
            cfg.gps.rate_hz,
            cfg.gps.dropout_probability,
            rngs["gps"],
            rngs["gps_dropout"],
        )

        pose0 = np.array([rb.initial_x, rb.initial_y, rb.initial_yaw], dtype=float)
        self.robot = DifferentialDriveRobot(self.params, pose0)
        pose0 = self.robot.pose  # yaw wrapped
        self.dead_reckoning = DeadReckoning(pose0, cfg.ekf.integration)
        P0 = np.diag([cfg.ekf.p0_pos_std**2, cfg.ekf.p0_pos_std**2, cfg.ekf.p0_yaw_std**2])
        self.ekf = ExtendedKalmanFilter(pose0, P0, cfg.ekf.integration)

        # Filter noise matrices. Q is a continuous-time spectral density scaled by dt.
        self.Q = np.diag([cfg.ekf.q_pos**2, cfg.ekf.q_pos**2, cfg.ekf.q_yaw**2]) * self.dt
        gps_std = cfg.gps.noise_std if cfg.ekf.match_sensor_noise else cfg.ekf.gps_r_std
        head_std = cfg.imu.heading_noise_std if cfg.ekf.match_sensor_noise else cfg.ekf.heading_r_std
        self.R_gps = np.eye(2) * max(gps_std, MIN_GPS_R_STD) ** 2
        self.R_heading = np.array([[max(head_std, MIN_HEADING_R_STD) ** 2]])

        self.step_index = 0
        self.warnings: list[str] = []
        self._rows: list[dict] = []
        self._records: list[EKFStepRecord | None] = []
        self._rows.append(self._initial_row(pose0))
        self._records.append(None)
        self._df_cache: pd.DataFrame | None = None
        self._df_cache_len = -1

    def _initial_row(self, pose0: NDArray[np.float64]) -> dict:
        nan = float("nan")
        P = self.ekf.covariance
        return {
            "step": 0, "t": 0.0,
            "x_true": pose0[0], "y_true": pose0[1], "psi_true": pose0[2],
            "v_true": nan, "omega_true": nan,
            "v_cmd": nan, "omega_cmd": nan,
            "alpha_l_true": nan, "alpha_r_true": nan,
            "alpha_l_meas": nan, "alpha_r_meas": nan,
            "v_meas": nan, "omega_wheel": nan, "gyro_yaw_rate": nan, "gyro_bias": nan,
            "omega_used": nan, "imu_heading": nan,
            "gps_status": "initial", "gps_x": nan, "gps_y": nan,
            "gps_innov_x": nan, "gps_innov_y": nan, "gps_nis": nan,
            "heading_update": False,
            "x_odo": pose0[0], "y_odo": pose0[1], "psi_odo": pose0[2],
            "x_ekf": pose0[0], "y_ekf": pose0[1], "psi_ekf": pose0[2],
            "sigma_x": float(np.sqrt(P[0, 0])), "sigma_y": float(np.sqrt(P[1, 1])),
            "sigma_psi": float(np.sqrt(P[2, 2])),
        }

    # --------------------------------------------------------------- stepping
    @property
    def done(self) -> bool:
        return self.step_index >= self.n_steps

    @property
    def time(self) -> float:
        return self.step_index * self.dt

    def step(self) -> dict:
        """Advance one time step and return the stored row (dict)."""
        if self.done:
            raise SimulationFinished("The simulation has already reached its final time step.")
        cfg = self.config
        k, dt = self.step_index, self.dt
        n = k + 1  # 1-based index of the sample being produced
        t = n * dt

        # 1. Ground truth
        alpha_l, alpha_r = float(self._alpha_l[k]), float(self._alpha_r[k])
        v_true, w_true = self.robot.step(alpha_l, alpha_r, dt)
        pose_true = self.robot.pose

        # 2. Sensors
        al_m, ar_m = self.wheel_sensor.measure(alpha_l, alpha_r)
        v_meas, w_wheel = wheel_speeds_to_body_velocity(al_m, ar_m, self.params)
        imu = self.imu_sensor.measure(w_true, pose_true[2], dt)
        omega_used = imu.yaw_rate if cfg.imu.enabled else w_wheel

        # 3. Odometry-only (no GPS, no heading correction)
        pose_odo = self.dead_reckoning.update(v_meas, omega_used, dt)

        # 4. EKF prediction
        prediction = self.ekf.predict(v_meas, omega_used, dt, self.Q)

        # 5. EKF corrections
        gps = self.gps_sensor.measure(pose_true[:2], n, dt)
        gps_status = gps.status
        gps_update: UpdateResult | None = None
        if gps.status == GPS_FIX:
            try:
                gps_update = self.ekf.update_gps(gps.position, self.R_gps)
            except NumericalError as exc:
                gps_status = GPS_UPDATE_FAILED
                msg = f"t={t:.2f}s: GPS update rejected: {exc}"
                self.warnings.append(msg)
                logger.warning(msg)

        heading_update: UpdateResult | None = None
        if (
            cfg.imu.enabled
            and cfg.imu.heading_correction
            and is_scheduled(n, cfg.imu.heading_rate_hz, dt)
        ):
            try:
                heading_update = self.ekf.update_heading(imu.heading, self.R_heading)
            except NumericalError as exc:
                msg = f"t={t:.2f}s: heading update rejected: {exc}"
                self.warnings.append(msg)
                logger.warning(msg)

        # 6. Store
        est = self.ekf.state
        P = self.ekf.covariance
        nan = float("nan")
        row = {
            "step": n, "t": t,
            "x_true": pose_true[0], "y_true": pose_true[1], "psi_true": pose_true[2],
            "v_true": v_true, "omega_true": w_true,
            "v_cmd": float(self._v_cmd[k]), "omega_cmd": float(self._w_cmd[k]),
            "alpha_l_true": alpha_l, "alpha_r_true": alpha_r,
            "alpha_l_meas": al_m, "alpha_r_meas": ar_m,
            "v_meas": v_meas, "omega_wheel": w_wheel,
            "gyro_yaw_rate": imu.yaw_rate if cfg.imu.enabled else nan,
            "gyro_bias": imu.bias if cfg.imu.enabled else nan,
            "omega_used": omega_used,
            "imu_heading": imu.heading if (cfg.imu.enabled and cfg.imu.heading_correction) else nan,
            "gps_status": gps_status,
            "gps_x": gps.position[0] if gps.position is not None else nan,
            "gps_y": gps.position[1] if gps.position is not None else nan,
            "gps_innov_x": gps_update.innovation[0] if gps_update else nan,
            "gps_innov_y": gps_update.innovation[1] if gps_update else nan,
            "gps_nis": gps_update.nis if gps_update else nan,
            "heading_update": heading_update is not None,
            "x_odo": pose_odo[0], "y_odo": pose_odo[1], "psi_odo": pose_odo[2],
            "x_ekf": est[0], "y_ekf": est[1], "psi_ekf": est[2],
            "sigma_x": float(np.sqrt(max(P[0, 0], 0.0))),
            "sigma_y": float(np.sqrt(max(P[1, 1], 0.0))),
            "sigma_psi": float(np.sqrt(max(P[2, 2], 0.0))),
        }
        self._rows.append(row)
        self._records.append(EKFStepRecord(n, t, prediction, gps_update, heading_update))
        self.step_index = n
        return row

    def run(self, n_steps: int | None = None) -> "Simulation":
        """Run ``n_steps`` more steps (default: until the end)."""
        remaining = self.n_steps - self.step_index
        count = remaining if n_steps is None else min(int(n_steps), remaining)
        for _ in range(count):
            self.step()
        return self

    # ---------------------------------------------------------------- results
    def to_dataframe(self) -> pd.DataFrame:
        """All stored rows with derived error columns (cached until the next step)."""
        if self._df_cache is not None and self._df_cache_len == len(self._rows):
            return self._df_cache
        df = pd.DataFrame(self._rows)
        df["err_odo"] = position_error(df.x_true, df.y_true, df.x_odo, df.y_odo)
        df["err_ekf"] = position_error(df.x_true, df.y_true, df.x_ekf, df.y_ekf)
        df["yaw_err_odo"] = yaw_error(df.psi_true, df.psi_odo)
        df["yaw_err_ekf"] = yaw_error(df.psi_true, df.psi_ekf)
        self._df_cache, self._df_cache_len = df, len(self._rows)
        return df

    @property
    def ekf_records(self) -> list[EKFStepRecord | None]:
        return self._records

    def result(self) -> SimulationResult:
        return SimulationResult(self.config, self.to_dataframe(), list(self._records), list(self.warnings))

    def boundary_violations(self) -> int:
        """Number of stored time steps where the ground truth lies outside the boundary."""
        s = self.config.simulation
        if not s.boundary_enabled:
            return 0
        df = self.to_dataframe()
        outside = (
            (df.x_true < s.boundary_xmin) | (df.x_true > s.boundary_xmax)
            | (df.y_true < s.boundary_ymin) | (df.y_true > s.boundary_ymax)
        )
        return int(outside.sum())


def run_simulation(config: SimConfig) -> Simulation:
    """Build and fully run a simulation."""
    return Simulation(config).run()
