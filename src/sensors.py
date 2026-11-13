"""Configurable sensor models: wheel odometry, IMU (gyro + heading) and GPS.

Every sensor owns an independent random stream derived from one master seed
(:func:`make_rngs`), so the same seed always reproduces the same measurements
and changing one sensor's settings never alters another sensor's noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import ArrayLike

from .coordinate_frames import wrap_angle

GPS_FIX: Final = "fix"
GPS_DROPOUT: Final = "dropout"  # a fix was scheduled but lost
GPS_IDLE: Final = "idle"  # no fix scheduled at this step
GPS_DISABLED: Final = "disabled"

_STREAM_NAMES: Final = ("wheel", "imu", "gps", "gps_dropout")


def make_rngs(seed: int) -> dict[str, np.random.Generator]:
    """Create independent, reproducible random generators from one master seed."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}.")
    children = np.random.SeedSequence(int(seed)).spawn(len(_STREAM_NAMES))
    return {name: np.random.default_rng(c) for name, c in zip(_STREAM_NAMES, children)}


def schedule_period_steps(rate_hz: float, dt: float) -> int:
    """Number of simulation steps between two samples of a sensor running at ``rate_hz``."""
    if not np.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError(f"rate_hz must be positive and finite, got {rate_hz!r}.")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive and finite, got {dt!r}.")
    return max(1, int(round(1.0 / (rate_hz * dt))))


def is_scheduled(step_index: int, rate_hz: float, dt: float) -> bool:
    """True if a sensor with the given rate samples at (1-based) ``step_index``."""
    return step_index % schedule_period_steps(rate_hz, dt) == 0


def _check_non_negative(name: str, value: float) -> float:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number, got {value!r}.")
    return float(value)


@dataclass
class WheelOdometrySensor:
    """Wheel encoders: measured angular velocity of each wheel.

    ``alpha_meas = alpha_true * (1 + scale_error_on_right_wheel) + N(0, noise_std^2)``

    The scale error models a calibration error (e.g. wheel-radius mismatch). It
    produces systematic drift, whereas the Gaussian noise produces a random walk.
    """

    noise_std: float
    scale_error: float
    rng: np.random.Generator

    def __post_init__(self) -> None:
        self.noise_std = _check_non_negative("wheel noise_std", self.noise_std)
        if not np.isfinite(self.scale_error) or self.scale_error <= -1.0:
            raise ValueError(f"scale_error must be finite and > -1, got {self.scale_error!r}.")

    def measure(self, alpha_l: float, alpha_r: float) -> tuple[float, float]:
        noise = self.rng.normal(0.0, self.noise_std, size=2)
        return (
            float(alpha_l + noise[0]),
            float(alpha_r * (1.0 + self.scale_error) + noise[1]),
        )


@dataclass(frozen=True)
class IMUMeasurement:
    yaw_rate: float  # gyro reading [rad/s]
    bias: float  # gyro bias that was in effect (kept for analysis only)
    heading: float  # noisy heading [rad], always generated; use only if enabled


@dataclass
class IMUSensor:
    """Gyroscope yaw rate with white noise and a slowly varying bias, plus a noisy heading.

    ``gyro = omega_true + bias_k + N(0, gyro_noise_std^2)``
    ``bias_{k+1} = bias_k + N(0, bias_random_walk_std^2 * dt)``
    ``heading = wrap(psi_true + N(0, heading_noise_std^2))``
    """

    gyro_noise_std: float
    gyro_bias: float
    bias_random_walk_std: float
    heading_noise_std: float
    rng: np.random.Generator
    bias: float = field(init=False)

    def __post_init__(self) -> None:
        self.gyro_noise_std = _check_non_negative("gyro_noise_std", self.gyro_noise_std)
        self.bias_random_walk_std = _check_non_negative(
            "bias_random_walk_std", self.bias_random_walk_std
        )
        self.heading_noise_std = _check_non_negative("heading_noise_std", self.heading_noise_std)
        if not np.isfinite(self.gyro_bias):
            raise ValueError("gyro_bias must be finite.")
        self.bias = float(self.gyro_bias)

    def measure(self, omega_true: float, psi_true: float, dt: float) -> IMUMeasurement:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be positive and finite, got {dt!r}.")
        # Fixed draw order keeps the stream identical regardless of what is used later.
        n_gyro, n_walk, n_head = self.rng.normal(0.0, 1.0, size=3)
        self.bias += self.bias_random_walk_std * np.sqrt(dt) * n_walk
        return IMUMeasurement(
            yaw_rate=float(omega_true + self.bias + self.gyro_noise_std * n_gyro),
            bias=float(self.bias),
            heading=float(wrap_angle(psi_true + self.heading_noise_std * n_head)),
        )


@dataclass(frozen=True)
class GPSMeasurement:
    status: str  # GPS_FIX | GPS_DROPOUT | GPS_IDLE | GPS_DISABLED
    position: np.ndarray | None = None  # [x, y] in the GLOBAL frame when status == GPS_FIX

    @property
    def available(self) -> bool:
        return self.status == GPS_FIX


@dataclass
class GPSSensor:
    """Intermittent noisy GPS returning global ``[x, y]`` positions.

    A fix is *scheduled* every ``1 / rate_hz`` seconds (rounded to whole time
    steps). Each scheduled fix is lost with probability ``dropout_probability``.
    """

    enabled: bool
    noise_std: float
    rate_hz: float
    dropout_probability: float
    rng: np.random.Generator
    dropout_rng: np.random.Generator

    def __post_init__(self) -> None:
        self.noise_std = _check_non_negative("gps noise_std", self.noise_std)
        if not (0.0 <= self.dropout_probability <= 1.0):
            raise ValueError("dropout_probability must be within [0, 1].")
        if self.enabled and (not np.isfinite(self.rate_hz) or self.rate_hz <= 0.0):
            raise ValueError(f"GPS rate_hz must be positive when enabled, got {self.rate_hz!r}.")

    def measure(self, true_position: ArrayLike, step_index: int, dt: float) -> GPSMeasurement:
        """Sample the GPS at (1-based) ``step_index``; ``true_position`` is global [x, y]."""
        if not self.enabled:
            return GPSMeasurement(GPS_DISABLED)
        if not is_scheduled(step_index, self.rate_hz, dt):
            return GPSMeasurement(GPS_IDLE)
        lost = self.dropout_rng.random() < self.dropout_probability
        noise = self.rng.normal(0.0, self.noise_std, size=2)
        if lost:
            return GPSMeasurement(GPS_DROPOUT)
        pos = np.asarray(true_position, dtype=float).reshape(2) + noise
        return GPSMeasurement(GPS_FIX, pos)
