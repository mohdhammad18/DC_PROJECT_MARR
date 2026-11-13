import numpy as np
import pytest

from src.robot import (
    DifferentialDriveRobot,
    RobotParams,
    body_velocity_to_wheel_speeds,
    integrate_pose_exact,
    wheel_speeds_to_body_velocity,
)

PARAMS = RobotParams(wheel_radius=0.05, wheel_separation=0.30)


def test_velocity_equations_match_brief():
    aL, aR = 8.0, 12.0
    v, omega = wheel_speeds_to_body_velocity(aL, aR, PARAMS)
    assert v == pytest.approx(0.05 / 2 * (aR + aL))
    assert omega == pytest.approx(0.05 / 0.30 * (aR - aL))


def test_equal_wheel_speeds_drive_straight_and_opposite_speeds_spin_in_place():
    v, omega = wheel_speeds_to_body_velocity(10.0, 10.0, PARAMS)
    assert v == pytest.approx(0.5) and omega == 0.0
    v, omega = wheel_speeds_to_body_velocity(-5.0, 5.0, PARAMS)
    assert v == pytest.approx(0.0) and omega > 0


def test_inverse_kinematics_round_trip_and_vectorised():
    v = np.array([0.0, 0.5, -0.2])
    w = np.array([0.0, 0.3, 1.0])
    aL, aR = body_velocity_to_wheel_speeds(v, w, PARAMS)
    v2, w2 = wheel_speeds_to_body_velocity(aL, aR, PARAMS)
    assert np.allclose(v, v2) and np.allclose(w, w2)


def test_straight_line_motion_with_initial_heading():
    psi0 = np.radians(30)
    robot = DifferentialDriveRobot(PARAMS, [1.0, 2.0, psi0])
    dt, n = 0.01, 500
    for _ in range(n):
        robot.step(10.0, 10.0, dt)
    dist = 0.5 * n * dt
    x, y, psi = robot.pose
    assert x == pytest.approx(1.0 + dist * np.cos(psi0))
    assert y == pytest.approx(2.0 + dist * np.sin(psi0))
    assert psi == pytest.approx(psi0)


def test_turning_in_place_changes_only_heading():
    robot = DifferentialDriveRobot(PARAMS, [0.0, 0.0, 0.0])
    omega = PARAMS.wheel_radius / PARAMS.wheel_separation * 2 * 5.0  # aR=5, aL=-5
    for _ in range(100):
        robot.step(-5.0, 5.0, 0.01)
    x, y, psi = robot.pose
    assert (x, y) == (pytest.approx(0.0), pytest.approx(0.0))
    assert psi == pytest.approx(omega * 1.0)


def test_full_circle_returns_to_start_and_radius_is_v_over_omega():
    v, omega = 0.5, 0.25  # radius 2 m
    aL, aR = body_velocity_to_wheel_speeds(v, omega, PARAMS)
    robot = DifferentialDriveRobot(PARAMS, [0.0, 0.0, 0.0])
    n = 400
    dt = 2 * np.pi / omega / n
    pts = []
    for _ in range(n):
        robot.step(aL, aR, dt)
        pts.append(robot.pose[:2])
    assert np.allclose(robot.pose[:2], [0.0, 0.0], atol=1e-9)
    centre = np.array([0.0, v / omega])
    assert np.allclose(np.linalg.norm(np.array(pts) - centre, axis=1), v / omega, atol=1e-9)


def test_exact_integration_is_step_size_independent():
    one = integrate_pose_exact([0, 0, 0.3], 0.5, 0.4, 2.0)
    pose = np.array([0, 0, 0.3])
    for _ in range(4):
        pose = integrate_pose_exact(pose, 0.5, 0.4, 0.5)
    assert np.allclose(one, pose, atol=1e-12)


def test_heading_is_wrapped_consistently():
    robot = DifferentialDriveRobot(PARAMS, [0.0, 0.0, 3.0])
    for _ in range(100):
        robot.step(-10.0, 10.0, 0.01)  # keeps rotating counter-clockwise
        assert -np.pi <= robot.pose[2] < np.pi


def test_initial_yaw_is_wrapped():
    robot = DifferentialDriveRobot(PARAMS, [0, 0, 2 * np.pi + 0.5])
    assert robot.pose[2] == pytest.approx(0.5)


@pytest.mark.parametrize("r,b", [(0.0, 0.3), (-0.05, 0.3), (0.05, 0.0), (np.nan, 0.3), (0.05, np.inf)])
def test_invalid_robot_parameters_raise(r, b):
    with pytest.raises(ValueError):
        RobotParams(r, b)


@pytest.mark.parametrize("dt", [0.0, -0.1, np.nan, np.inf])
def test_invalid_dt_raises(dt):
    robot = DifferentialDriveRobot(PARAMS)
    with pytest.raises(ValueError):
        robot.step(1.0, 1.0, dt)


def test_non_finite_wheel_speed_raises():
    with pytest.raises(ValueError):
        DifferentialDriveRobot(PARAMS).step(np.nan, 1.0, 0.1)
