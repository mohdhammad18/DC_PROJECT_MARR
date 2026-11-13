"""Ready-to-run experimental scenarios.

A scenario is a set of configuration overrides applied on top of a *base*
configuration. Trajectory, duration, time step and seed always come from the
base, so scenarios are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import SimConfig, deep_merge
from .simulation import Simulation

_IDEAL_ODOMETRY = {"wheel_noise_std": 0.0, "scale_error": 0.0}
_IDEAL_IMU = {"gyro_noise_std": 0.0, "gyro_bias": 0.0, "bias_random_walk_std": 0.0}
_HARSH_ODOMETRY = {"wheel_noise_std": 0.5, "scale_error": 0.015}


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    description: str
    overrides: dict[str, Any] = field(default_factory=dict)


SCENARIOS: dict[str, Scenario] = {
    s.key: s
    for s in (
        Scenario(
            "ideal",
            "1. Ideal sensors",
            "All noise, bias and scale errors are zero, so both estimators track the truth almost perfectly "
            "(remaining error is only discretisation of the motion model).",
            {
                "odometry": _IDEAL_ODOMETRY,
                "imu": {**_IDEAL_IMU, "enabled": True, "heading_correction": False},
                "gps": {"enabled": True, "noise_std": 0.0, "dropout_probability": 0.0},
                "ekf": {"integration": "midpoint", "q_pos": 0.001, "q_yaw": 0.001},
            },
        ),
        Scenario(
            "noisy_odometry",
            "2. Noisy odometry",
            "Wheel-only dead reckoning with noisy, mis-scaled encoders (no gyro, no GPS): heading and "
            "position errors accumulate without bound.",
            {
                "odometry": _HARSH_ODOMETRY,
                "imu": {"enabled": False, "heading_correction": False},
                "gps": {"enabled": False},
            },
        ),
        Scenario(
            "noisy_gps",
            "3. Noisy GPS",
            "Perfect odometry and gyro, but a poor GPS (1.5 m noise, 30 % lost fixes). The scattered "
            "fixes are filtered by the EKF instead of being followed blindly.",
            {
                "odometry": _IDEAL_ODOMETRY,
                "imu": {**_IDEAL_IMU, "enabled": True, "heading_correction": False},
                "gps": {"enabled": True, "noise_std": 1.5, "dropout_probability": 0.3},
            },
        ),
        Scenario(
            "odometry_imu",
            "4. Odometry + IMU",
            "Same harsh wheel odometry as scenario 2, but the gyro supplies the yaw rate: heading-assisted "
            "dead reckoning. Still no GPS, so position error still grows.",
            {
                "odometry": _HARSH_ODOMETRY,
                "imu": {"enabled": True, "heading_correction": False},
                "gps": {"enabled": False},
            },
        ),
        Scenario(
            "full_fusion",
            "5. Full sensor fusion",
            "Wheel odometry + gyro drive the EKF prediction; intermittent noisy GPS fixes correct the "
            "accumulated drift.",
            {
                "imu": {"enabled": True},
                "gps": {"enabled": True},
            },
        ),
        Scenario(
            "gps_denied",
            "6. GPS-denied",
            "The full-fusion sensor suite with GPS switched off: the EKF can only predict, so it behaves "
            "like dead reckoning and drifts.",
            {
                "imu": {"enabled": True},
                "gps": {"enabled": False},
            },
        ),
    )
}


def scenario_config(key: str, base: SimConfig) -> SimConfig:
    """Return a validated copy of ``base`` with the scenario overrides applied."""
    if key not in SCENARIOS:
        raise KeyError(f"Unknown scenario {key!r}; available: {list(SCENARIOS)}.")
    merged = deep_merge(base.to_dict(), SCENARIOS[key].overrides)
    return SimConfig.from_dict(merged).validate()


def run_scenarios(keys: list[str], base: SimConfig) -> dict[str, Simulation]:
    """Run each scenario to completion with the base trajectory, duration, dt and seed."""
    return {k: Simulation(scenario_config(k, base)).run() for k in keys}
