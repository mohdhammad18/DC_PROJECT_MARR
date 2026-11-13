"""Typed configuration for the simulator, loadable from YAML and validated.

``SimConfig.from_dict`` rejects unknown keys (typos) and coerces values to the
types of the dataclass defaults. ``SimConfig.validate`` collects *all* problems
and raises a single :class:`ConfigError`.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

from .odometry import INTEGRATION_METHODS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default_config.yaml"

TRAJECTORY_NAMES: tuple[str, ...] = ("straight", "square", "circle", "figure_eight", "custom")
MAX_STEPS = 200_000


class ConfigError(ValueError):
    """Raised for malformed or physically invalid configuration."""


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


@dataclass
class RobotConfig:
    wheel_radius: float = 0.05
    wheel_separation: float = 0.30
    initial_x: float = 0.0
    initial_y: float = 0.0
    initial_yaw: float = 0.0


@dataclass
class TrajectoryConfig:
    name: str = "square"
    speed: float = 0.5
    turn_rate: float = 0.5
    side_length: float = 8.0
    radius: float = 4.0
    custom_segments: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"duration": 10.0, "v": 0.5, "omega": 0.0},
            {"duration": 6.0, "v": 0.4, "omega": 0.3},
        ]
    )


@dataclass
class SimulationSettings:
    dt: float = 0.05
    duration: float = 80.0
    seed: int = 42
    boundary_enabled: bool = True
    boundary_xmin: float = -2.0
    boundary_xmax: float = 12.0
    boundary_ymin: float = -2.0
    boundary_ymax: float = 12.0


@dataclass
class OdometryConfig:
    wheel_noise_std: float = 0.30
    scale_error: float = 0.01


@dataclass
class IMUConfig:
    enabled: bool = True
    gyro_noise_std: float = 0.01
    gyro_bias: float = 0.002
    bias_random_walk_std: float = 0.0005
    heading_correction: bool = False
    heading_noise_std: float = 0.05
    heading_rate_hz: float = 2.0


@dataclass
class GPSConfig:
    enabled: bool = True
    noise_std: float = 0.5
    rate_hz: float = 1.0
    dropout_probability: float = 0.1


@dataclass
class EKFConfig:
    integration: str = "euler"
    q_pos: float = 0.05
    q_yaw: float = 0.02
    p0_pos_std: float = 0.1
    p0_yaw_std: float = 0.05
    match_sensor_noise: bool = True
    gps_r_std: float = 0.5
    heading_r_std: float = 0.05


_SECTIONS: dict[str, type] = {
    "robot": RobotConfig,
    "trajectory": TrajectoryConfig,
    "simulation": SimulationSettings,
    "odometry": OdometryConfig,
    "imu": IMUConfig,
    "gps": GPSConfig,
    "ekf": EKFConfig,
}


def _build_section(cls: type, data: Mapping[str, Any] | None, section: str):
    data = dict(data or {})
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"Unknown key(s) in '{section}': {sorted(unknown)}. Valid: {sorted(known)}.")
    instance = cls()
    for f in fields(cls):
        if f.name not in data:
            continue
        default = getattr(instance, f.name)
        value = data[f.name]
        try:
            if isinstance(default, bool):
                if not isinstance(value, bool):
                    raise TypeError("expected true/false")
                coerced = value
            elif isinstance(default, int):
                if isinstance(value, bool) or int(value) != value:
                    raise TypeError("expected an integer")
                coerced = int(value)
            elif isinstance(default, float):
                if isinstance(value, bool):
                    raise TypeError("expected a number")
                coerced = float(value)
            elif isinstance(default, str):
                coerced = str(value)
            elif isinstance(default, list):
                if not isinstance(value, list):
                    raise TypeError("expected a list")
                coerced = copy.deepcopy(value)
            else:  # pragma: no cover
                coerced = value
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid value for '{section}.{f.name}': {value!r} ({exc}).") from exc
        setattr(instance, f.name, coerced)
    return instance


@dataclass
class SimConfig:
    robot: RobotConfig = field(default_factory=RobotConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    odometry: OdometryConfig = field(default_factory=OdometryConfig)
    imu: IMUConfig = field(default_factory=IMUConfig)
    gps: GPSConfig = field(default_factory=GPSConfig)
    ekf: EKFConfig = field(default_factory=EKFConfig)

    # ------------------------------------------------------------ conversion
    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SimConfig":
        data = dict(data or {})
        unknown = set(data) - set(_SECTIONS)
        if unknown:
            raise ConfigError(f"Unknown config section(s): {sorted(unknown)}. Valid: {sorted(_SECTIONS)}.")
        return cls(**{name: _build_section(c, data.get(name), name) for name, c in _SECTIONS.items()})

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(asdict(self))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @property
    def n_steps(self) -> int:
        return int(round(self.simulation.duration / self.simulation.dt))

    # ------------------------------------------------------------ validation
    def validate(self) -> "SimConfig":
        """Raise :class:`ConfigError` listing every invalid setting."""
        errs: list[str] = []

        def positive(path: str, v: float) -> None:
            if not _finite(v) or v <= 0:
                errs.append(f"{path} must be a positive finite number (got {v!r}).")

        def non_negative(path: str, v: float) -> None:
            if not _finite(v) or v < 0:
                errs.append(f"{path} must be a non-negative finite number (got {v!r}).")

        r, t, s, o, i, g, e = (
            self.robot, self.trajectory, self.simulation, self.odometry, self.imu, self.gps, self.ekf,
        )
        positive("robot.wheel_radius", r.wheel_radius)
        positive("robot.wheel_separation", r.wheel_separation)
        for name in ("initial_x", "initial_y", "initial_yaw"):
            if not _finite(getattr(r, name)):
                errs.append(f"robot.{name} must be finite.")

        positive("simulation.dt", s.dt)
        positive("simulation.duration", s.duration)
        if _finite(s.dt) and _finite(s.duration) and s.dt > 0 and s.duration > 0:
            if s.dt > 1.0:
                errs.append("simulation.dt must be <= 1 s for a meaningful integration.")
            if s.duration < s.dt:
                errs.append("simulation.duration must be at least one time step.")
            if s.duration / s.dt > MAX_STEPS:
                errs.append(f"duration/dt exceeds the maximum of {MAX_STEPS} steps.")
        if not isinstance(s.seed, int) or s.seed < 0:
            errs.append("simulation.seed must be a non-negative integer.")
        if s.boundary_enabled and not (
            s.boundary_xmin < s.boundary_xmax and s.boundary_ymin < s.boundary_ymax
        ):
            errs.append("Boundary requires xmin < xmax and ymin < ymax.")

        if t.name not in TRAJECTORY_NAMES:
            errs.append(f"trajectory.name must be one of {TRAJECTORY_NAMES} (got {t.name!r}).")
        for name in ("speed", "turn_rate", "side_length", "radius"):
            positive(f"trajectory.{name}", getattr(t, name))
        if t.name == "custom":
            if not t.custom_segments:
                errs.append("trajectory.custom_segments must contain at least one segment.")
            for n, seg in enumerate(t.custom_segments):
                if not isinstance(seg, dict) or not {"duration", "v", "omega"} <= set(seg):
                    errs.append(f"custom segment {n} needs 'duration', 'v' and 'omega'.")
                    continue
                if not all(_finite(seg[k]) for k in ("duration", "v", "omega")):
                    errs.append(f"custom segment {n} has non-finite values.")
                elif seg["duration"] <= 0:
                    errs.append(f"custom segment {n}: duration must be > 0.")

        non_negative("odometry.wheel_noise_std", o.wheel_noise_std)
        if not _finite(o.scale_error) or o.scale_error <= -1.0:
            errs.append("odometry.scale_error must be finite and > -1.")

        non_negative("imu.gyro_noise_std", i.gyro_noise_std)
        non_negative("imu.bias_random_walk_std", i.bias_random_walk_std)
        non_negative("imu.heading_noise_std", i.heading_noise_std)
        if not _finite(i.gyro_bias):
            errs.append("imu.gyro_bias must be finite.")
        if i.heading_correction:
            positive("imu.heading_rate_hz", i.heading_rate_hz)
            if not i.enabled:
                errs.append("imu.heading_correction requires imu.enabled to be true.")

        non_negative("gps.noise_std", g.noise_std)
        if g.enabled:
            positive("gps.rate_hz", g.rate_hz)
        if not (_finite(g.dropout_probability) and 0.0 <= g.dropout_probability <= 1.0):
            errs.append("gps.dropout_probability must be within [0, 1].")

        if e.integration not in INTEGRATION_METHODS:
            errs.append(f"ekf.integration must be one of {INTEGRATION_METHODS}.")
        non_negative("ekf.q_pos", e.q_pos)
        non_negative("ekf.q_yaw", e.q_yaw)
        positive("ekf.p0_pos_std", e.p0_pos_std)
        positive("ekf.p0_yaw_std", e.p0_yaw_std)
        if not e.match_sensor_noise:
            positive("ekf.gps_r_std", e.gps_r_std)
            positive("ekf.heading_r_std", e.heading_r_std)

        if errs:
            raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errs))
        return self


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a deep copy of ``base``."""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None) -> SimConfig:
    """Load a YAML configuration (default: ``config/default_config.yaml``) and validate it."""
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        with open(p, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {p}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Configuration file {p} is not valid YAML: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Configuration file {p} must contain a mapping at the top level.")
    return SimConfig.from_dict(raw).validate()
