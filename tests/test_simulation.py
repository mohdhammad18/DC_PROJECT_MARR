import numpy as np
import pandas as pd
import pytest

from src.config import ConfigError, SimConfig, deep_merge, load_config
from src.metrics import compare_estimators, position_error, summarize_errors, yaw_error
from src.scenarios import SCENARIOS, run_scenarios, scenario_config
from src.sensors import (
    GPS_DISABLED,
    GPS_DROPOUT,
    GPS_FIX,
    GPS_IDLE,
    GPSSensor,
    IMUSensor,
    WheelOdometrySensor,
    make_rngs,
    schedule_period_steps,
)
from src.simulation import Simulation, SimulationFinished, run_simulation
from src.trajectories import generate_commands
from src.config import TRAJECTORY_NAMES


def make_cfg(**sections) -> SimConfig:
    base = load_config().to_dict()
    base["simulation"].update(duration=20.0, dt=0.05)
    return SimConfig.from_dict(deep_merge(base, sections)).validate()


# --------------------------------------------------------------- sensor models
def test_sensor_noise_is_reproducible_for_the_same_seed():
    def sample(seed):
        rngs = make_rngs(seed)
        wheel = WheelOdometrySensor(0.3, 0.0, rngs["wheel"])
        imu = IMUSensor(0.01, 0.002, 0.001, 0.05, rngs["imu"])
        gps = GPSSensor(True, 0.5, 1.0, 0.0, rngs["gps"], rngs["gps_dropout"])
        w = [wheel.measure(10.0, 10.0) for _ in range(50)]
        i = [imu.measure(0.1, 0.2, 0.05).yaw_rate for _ in range(50)]
        g = [gps.measure([1.0, 2.0], n, 0.05).position for n in range(1, 101)]
        return w, i, g

    a, b, c = sample(7), sample(7), sample(8)
    assert a[0] == b[0] and a[1] == b[1]
    assert all((x is None) == (y is None) for x, y in zip(a[2], b[2]))
    assert all(np.array_equal(x, y) for x, y in zip(a[2], b[2]) if x is not None)
    assert a[0] != c[0] and a[1] != c[1]


def test_sensor_streams_are_independent():
    r1, r2 = make_rngs(5), make_rngs(5)
    # Using the wheel sensor must not change what the IMU stream produces.
    WheelOdometrySensor(0.3, 0.0, r1["wheel"]).measure(1.0, 1.0)
    imu1 = IMUSensor(0.01, 0.0, 0.0, 0.05, r1["imu"]).measure(0.0, 0.0, 0.05)
    imu2 = IMUSensor(0.01, 0.0, 0.0, 0.05, r2["imu"]).measure(0.0, 0.0, 0.05)
    assert imu1 == imu2


def test_wheel_noise_statistics_and_scale_error():
    s = WheelOdometrySensor(0.5, 0.1, make_rngs(1)["wheel"])
    m = np.array([s.measure(10.0, 10.0) for _ in range(20000)])
    assert m[:, 0].mean() == pytest.approx(10.0, abs=0.02)
    assert m[:, 1].mean() == pytest.approx(11.0, abs=0.02)  # +10 % scale error on the right wheel
    assert m[:, 0].std() == pytest.approx(0.5, rel=0.03)


def test_imu_bias_and_noise():
    s = IMUSensor(0.0, 0.05, 0.0, 0.0, make_rngs(1)["imu"])
    assert s.measure(0.3, 0.7, 0.1).yaw_rate == pytest.approx(0.35)  # omega + bias, no noise
    assert s.measure(0.3, 0.7, 0.1).heading == pytest.approx(0.7)
    walk = IMUSensor(0.0, 0.0, 0.1, 0.0, make_rngs(2)["imu"])
    biases = [walk.measure(0.0, 0.0, 0.1).bias for _ in range(200)]
    assert len(set(np.round(biases, 8))) > 100  # bias varies slowly


def test_gps_is_intermittent_global_and_can_be_disabled():
    rngs = make_rngs(3)
    gps = GPSSensor(True, 0.0, 1.0, 0.0, rngs["gps"], rngs["gps_dropout"])
    dt = 0.05
    period = schedule_period_steps(1.0, dt)
    statuses = [gps.measure([3.0, 4.0], n, dt) for n in range(1, 101)]
    fixes = [i + 1 for i, m in enumerate(statuses) if m.status == GPS_FIX]
    assert fixes == list(range(period, 101, period))  # intermittent, at the configured rate
    assert all(m.status == GPS_IDLE for i, m in enumerate(statuses) if (i + 1) % period)
    assert np.allclose(statuses[period - 1].position, [3.0, 4.0])  # zero noise: global truth

    off = GPSSensor(False, 0.5, 1.0, 0.0, rngs["gps"], rngs["gps_dropout"])
    assert all(off.measure([0, 0], n, dt).status == GPS_DISABLED for n in range(1, 50))

    lossy = GPSSensor(True, 0.5, 1.0, 1.0, rngs["gps"], rngs["gps_dropout"])
    assert lossy.measure([0, 0], period, dt).status == GPS_DROPOUT


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_invalid_seed_raises(bad):
    with pytest.raises(ValueError):
        make_rngs(bad)


def test_invalid_sensor_parameters_raise():
    rng = make_rngs(0)
    with pytest.raises(ValueError):
        WheelOdometrySensor(-0.1, 0.0, rng["wheel"])
    with pytest.raises(ValueError):
        IMUSensor(np.nan, 0.0, 0.0, 0.0, rng["imu"])
    with pytest.raises(ValueError):
        GPSSensor(True, 0.5, 0.0, 0.0, rng["gps"], rng["gps_dropout"])
    with pytest.raises(ValueError):
        GPSSensor(True, 0.5, 1.0, 1.5, rng["gps"], rng["gps_dropout"])


# ----------------------------------------------------------------- trajectories
@pytest.mark.parametrize("name", TRAJECTORY_NAMES)
def test_every_trajectory_runs_with_all_three_estimates(name):
    cfg = make_cfg(trajectory={"name": name})
    sim = run_simulation(cfg)
    df = sim.to_dataframe()
    assert len(df) == cfg.n_steps + 1
    assert np.isfinite(df[["x_true", "y_true", "x_odo", "y_odo", "x_ekf", "y_ekf"]].to_numpy()).all()
    assert np.all(np.abs(df.psi_true) <= np.pi) and np.all(np.abs(df.psi_ekf) <= np.pi)


def test_square_trajectory_closes_after_one_lap():
    cfg = make_cfg(trajectory={"name": "square", "side_length": 4.0, "speed": 0.5, "turn_rate": 0.5},
                   simulation={"duration": 60.0})
    df = run_simulation(cfg).to_dataframe()
    n_side = round(4.0 / 0.5 / 0.05)
    n_turn = round((np.pi / 2) / 0.5 / 0.05)
    lap = 4 * (n_side + n_turn)
    assert lap < len(df)
    assert np.allclose(df.loc[lap, ["x_true", "y_true", "psi_true"]].to_numpy(float), [0, 0, 0], atol=1e-9)
    # Corner after the first side: (4, 0) and heading +90 deg after the first turn.
    assert np.allclose(df.loc[n_side, ["x_true", "y_true"]].to_numpy(float), [4.0, 0.0], atol=1e-9)
    assert df.loc[n_side + n_turn, "psi_true"] == pytest.approx(np.pi / 2)


def test_figure_eight_returns_to_start_with_two_loops():
    cfg = make_cfg(trajectory={"name": "figure_eight", "radius": 2.0, "speed": 0.5}, simulation={"duration": 60.0})
    cmds_v, cmds_w = generate_commands(cfg.trajectory, cfg.n_steps, cfg.simulation.dt)
    assert cmds_w.max() > 0 > cmds_w.min()  # turns both ways
    df = run_simulation(cfg).to_dataframe()
    n_loop = round(2 * np.pi * 2.0 / 0.5 / 0.05)
    assert np.allclose(df.loc[n_loop, ["x_true", "y_true"]].to_numpy(float), [0, 0], atol=1e-9)
    assert np.allclose(df.loc[2 * n_loop, ["x_true", "y_true"]].to_numpy(float), [0, 0], atol=1e-9)


def test_custom_sequence_follows_segments_then_stops():
    cfg = make_cfg(trajectory={"name": "custom", "custom_segments": [
        {"duration": 2.0, "v": 1.0, "omega": 0.0}, {"duration": 1.0, "v": 0.0, "omega": 0.5}]},
        simulation={"duration": 5.0})
    v, w = generate_commands(cfg.trajectory, cfg.n_steps, cfg.simulation.dt)
    assert np.allclose(v[:40], 1.0) and np.allclose(w[40:60], 0.5)
    assert np.all(v[60:] == 0) and np.all(w[60:] == 0)


# -------------------------------------------------------------------- engine
def test_complete_simulation_runs_and_stores_everything_on_one_time_axis():
    cfg = make_cfg()
    sim = run_simulation(cfg)
    df = sim.to_dataframe()
    assert sim.done and len(df) == cfg.n_steps + 1
    assert np.allclose(np.diff(df.t), cfg.simulation.dt)
    for col in ("x_true", "x_odo", "x_ekf", "gps_x", "gyro_yaw_rate", "err_odo", "err_ekf", "sigma_x"):
        assert col in df.columns
    assert len(sim.ekf_records) == len(df)
    assert (df.gps_status == GPS_FIX).sum() > 0 and df.gps_x.notna().sum() == (df.gps_status == GPS_FIX).sum()


def test_same_seed_and_config_is_deterministic_and_different_seed_differs():
    a = run_simulation(make_cfg()).to_dataframe()
    b = run_simulation(make_cfg()).to_dataframe()
    c = run_simulation(make_cfg(simulation={"seed": 43})).to_dataframe()
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a.x_odo, c.x_odo)
    assert np.allclose(a.x_true, c.x_true)  # ground truth does not depend on the seed


def test_stepwise_execution_equals_full_run_and_reset_reproduces():
    cfg = make_cfg()
    full = run_simulation(cfg).to_dataframe()
    sim = Simulation(cfg)
    for _ in range(cfg.n_steps):
        sim.step()
    pd.testing.assert_frame_equal(sim.to_dataframe(), full)
    with pytest.raises(SimulationFinished):
        sim.step()
    sim.reset()
    assert sim.step_index == 0 and len(sim.to_dataframe()) == 1
    pd.testing.assert_frame_equal(sim.run().to_dataframe(), full)


def test_partial_run_then_continue_matches_full_run():
    cfg = make_cfg()
    sim = Simulation(cfg).run(100)
    assert sim.step_index == 100 and not sim.done
    pd.testing.assert_frame_equal(sim.run().to_dataframe(), run_simulation(cfg).to_dataframe())


def test_odometry_only_never_uses_gps():
    with_gps = run_simulation(make_cfg()).to_dataframe()
    no_gps = run_simulation(make_cfg(gps={"enabled": False})).to_dataframe()
    assert np.allclose(with_gps[["x_odo", "y_odo", "psi_odo"]], no_gps[["x_odo", "y_odo", "psi_odo"]])
    assert not np.allclose(with_gps.x_ekf, no_gps.x_ekf)


def test_gps_denied_ekf_equals_dead_reckoning():
    df = run_simulation(make_cfg(gps={"enabled": False})).to_dataframe()
    assert (df.gps_status.iloc[1:] == GPS_DISABLED).all() and df.gps_x.isna().all()
    assert np.allclose(df[["x_ekf", "y_ekf", "psi_ekf"]], df[["x_odo", "y_odo", "psi_odo"]])
    assert df.sigma_x.iloc[-1] > df.sigma_x.iloc[0]  # uncertainty only grows


def test_ekf_beats_drifting_odometry_and_covariance_matrices_stay_valid():
    cfg = make_cfg(simulation={"duration": 60.0}, odometry={"wheel_noise_std": 0.5, "scale_error": 0.02})
    sim = run_simulation(cfg)
    df = sim.to_dataframe()
    assert df.err_odo.iloc[-1] > 1.0  # dead reckoning accumulated visible drift
    assert df.err_ekf.mean() < 0.35 * df.err_odo.mean()
    assert df.err_ekf.max() < 2.0
    for rec in sim.ekf_records[1:]:
        for P in (rec.prediction.P_pred, getattr(rec.gps_update, "P_updated", rec.prediction.P_pred)):
            assert np.allclose(P, P.T, atol=1e-12) and np.linalg.eigvalsh(P).min() > -1e-10


def test_heading_correction_runs_and_records_updates():
    cfg = make_cfg(imu={"heading_correction": True, "heading_rate_hz": 2.0})
    sim = run_simulation(cfg)
    df = sim.to_dataframe()
    assert df.heading_update.sum() == cfg.n_steps // round(1 / (2.0 * 0.05))
    assert any(r is not None and r.heading_update is not None for r in sim.ekf_records)


def test_gps_dropouts_are_recorded_not_hidden():
    df = run_simulation(make_cfg(gps={"dropout_probability": 1.0})).to_dataframe()
    assert (df.gps_status == GPS_DROPOUT).sum() > 0 and df.gps_x.isna().all()


def test_ideal_sensors_give_nearly_perfect_localization():
    cfg = scenario_config("ideal", make_cfg(trajectory={"name": "circle"}))
    df = run_simulation(cfg).to_dataframe()
    assert df.err_odo.max() < 0.01 and df.err_ekf.max() < 0.01
    assert np.degrees(np.abs(df.yaw_err_ekf)).max() < 0.1


# --------------------------------------------------------------------- scenarios
def test_all_scenarios_run_and_show_the_expected_behaviour():
    base = make_cfg(simulation={"duration": 60.0})
    sims = run_scenarios(list(SCENARIOS), base)
    d = {k: s.to_dataframe() for k, s in sims.items()}
    assert set(d) == set(SCENARIOS)
    assert d["ideal"].err_ekf.max() < 0.05
    # Noisy odometry drifts; IMU assistance reduces the heading drift.
    assert d["noisy_odometry"].err_odo.iloc[-1] > 2.0
    assert d["odometry_imu"].yaw_err_odo.abs().mean() < d["noisy_odometry"].yaw_err_odo.abs().mean()
    # Fusion beats dead reckoning; GPS denial removes the benefit.
    assert d["full_fusion"].err_ekf.mean() < d["gps_denied"].err_ekf.mean()
    assert d["full_fusion"].err_ekf.mean() < d["full_fusion"].err_odo.mean()
    # Noisy GPS: EKF filters the fixes, so it is better than the raw GPS scatter.
    g = d["noisy_gps"].dropna(subset=["gps_x"])
    raw_gps_err = np.hypot(g.gps_x - g.x_true, g.gps_y - g.y_true).mean()
    assert d["noisy_gps"].err_ekf.mean() < raw_gps_err
    # Scenarios share the base trajectory and time axis.
    assert all(np.allclose(d[k].x_true, d["ideal"].x_true) for k in d)


def test_unknown_scenario_raises():
    with pytest.raises(KeyError):
        scenario_config("nope", make_cfg())


# ---------------------------------------------------------------------- config
def test_default_yaml_loads_and_round_trips():
    cfg = load_config()
    assert SimConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    assert cfg.robot.wheel_radius > 0 and cfg.simulation.dt > 0


@pytest.mark.parametrize(
    "override",
    [
        {"simulation": {"dt": 0.0}},
        {"simulation": {"dt": -0.1}},
        {"simulation": {"duration": 0.0}},
        {"simulation": {"duration": 1e9}},
        {"simulation": {"seed": -1}},
        {"robot": {"wheel_radius": 0.0}},
        {"robot": {"wheel_separation": -1.0}},
        {"odometry": {"wheel_noise_std": -0.1}},
        {"gps": {"rate_hz": 0.0}},
        {"gps": {"dropout_probability": 1.5}},
        {"imu": {"gyro_noise_std": float("nan")}},
        {"imu": {"enabled": False, "heading_correction": True}},
        {"trajectory": {"name": "zigzag"}},
        {"trajectory": {"name": "custom", "custom_segments": []}},
        {"trajectory": {"name": "custom", "custom_segments": [{"duration": -1, "v": 1, "omega": 0}]}},
        {"ekf": {"integration": "rk4"}},
        {"ekf": {"p0_pos_std": 0.0}},
    ],
)
def test_invalid_configurations_are_rejected(override):
    base = load_config().to_dict()
    with pytest.raises(ConfigError):
        SimConfig.from_dict(deep_merge(base, override)).validate()


def test_unknown_or_mistyped_config_keys_are_rejected():
    with pytest.raises(ConfigError):
        SimConfig.from_dict({"gps": {"noise": 1.0}})
    with pytest.raises(ConfigError):
        SimConfig.from_dict({"gpss": {}})
    with pytest.raises(ConfigError):
        SimConfig.from_dict({"gps": {"enabled": "yes"}})
    with pytest.raises(ConfigError):
        load_config("this/file/does/not/exist.yaml")


def test_simulation_rejects_invalid_config():
    cfg = load_config()
    cfg.simulation.dt = 0.0
    with pytest.raises(ConfigError):
        Simulation(cfg)


# ----------------------------------------------------------------------- metrics
def test_metrics_formulas():
    e = position_error([0, 3], [0, 0], [0, 0], [0, 4])
    assert np.allclose(e, [0.0, 5.0])
    assert np.allclose(yaw_error([np.pi - 0.1], [-np.pi + 0.1]), [-0.2])
    s = summarize_errors([1.0, 2.0, 3.0], [0.1, -0.1, 0.1])
    assert s["mean_position_error_m"] == pytest.approx(2.0)
    assert s["rmse_position_m"] == pytest.approx(np.sqrt(14 / 3))
    assert s["max_position_error_m"] == 3.0 and s["final_position_error_m"] == 3.0
    assert s["mean_yaw_error_deg"] == pytest.approx(np.degrees(0.1))
    with pytest.raises(ValueError):
        summarize_errors([], [])


def test_compare_estimators_table_uses_simulation_errors():
    df = run_simulation(make_cfg()).to_dataframe()
    table = compare_estimators(df)
    assert list(table.index) == ["Odometry-only", "EKF"]
    assert table.iloc[1, 0] == pytest.approx(df.err_ekf.mean())
    assert table.iloc[0, 3] == pytest.approx(df.err_odo.iloc[-1])
