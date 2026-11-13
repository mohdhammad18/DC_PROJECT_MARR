import numpy as np
import pytest

from src.ekf import (
    ExtendedKalmanFilter,
    NumericalError,
    motion_jacobian,
    validate_covariance,
)
from src.odometry import propagate_pose

P0 = np.diag([0.5, 0.4, 0.05])
Q = np.diag([0.01, 0.01, 0.002])
R = np.diag([0.25, 0.25])


def _numerical_jacobian(state, v, omega, dt, method, eps=1e-6):
    state = np.asarray(state, float)
    J = np.zeros((3, 3))
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        plus = propagate_pose(state + d, v, omega, dt, method)
        minus = propagate_pose(state - d, v, omega, dt, method)
        diff = plus - minus
        diff[2] = (diff[2] + np.pi) % (2 * np.pi) - np.pi  # ignore the wrap branch
        J[:, i] = diff / (2 * eps)
    return J


@pytest.mark.parametrize("method", ["euler", "midpoint"])
def test_motion_jacobian_matches_numerical_derivative(method):
    state = [1.0, -2.0, 0.8]
    F = motion_jacobian(state, 0.7, 0.3, 0.1, method)
    assert np.allclose(F, _numerical_jacobian(state, 0.7, 0.3, 0.1, method), atol=1e-7)


def test_motion_jacobian_euler_has_documented_structure():
    psi, v, dt = 0.9, 0.6, 0.2
    F = motion_jacobian([0, 0, psi], v, 0.1, dt, "euler")
    expected = [[1, 0, -v * np.sin(psi) * dt], [0, 1, v * np.cos(psi) * dt], [0, 0, 1]]
    assert np.allclose(F, expected)


def test_prediction_follows_motion_model_equations():
    ekf = ExtendedKalmanFilter([1.0, 2.0, 0.5], P0)
    v, omega, dt = 0.8, 0.2, 0.1
    res = ekf.predict(v, omega, dt, Q)
    assert np.allclose(
        res.x_pred, [1 + v * np.cos(0.5) * dt, 2 + v * np.sin(0.5) * dt, 0.5 + omega * dt]
    )
    assert np.allclose(ekf.state, res.x_pred)


def test_prediction_covariance_is_F_P_Ft_plus_Q():
    ekf = ExtendedKalmanFilter([0.0, 0.0, 0.4], P0)
    res = ekf.predict(1.0, 0.1, 0.1, Q)
    F = motion_jacobian([0.0, 0.0, 0.4], 1.0, 0.1, 0.1)
    assert np.allclose(res.P_pred, F @ P0 @ F.T + Q)
    assert np.allclose(ekf.covariance, res.P_pred)
    # Uncertainty grows during pure prediction.
    assert np.trace(res.P_pred) > np.trace(P0)


def test_prediction_correlates_position_and_heading_uncertainty():
    ekf = ExtendedKalmanFilter([0.0, 0.0, 0.0], P0)
    res = ekf.predict(1.0, 0.0, 0.1, Q)
    assert abs(res.P_pred[1, 2]) > 0  # heading error moves y when driving along +x


def test_gps_update_matches_textbook_equations():
    ekf = ExtendedKalmanFilter([0.0, 0.0, 0.3], P0)
    ekf.predict(0.5, 0.0, 0.1, Q)
    x_m, P_m = ekf.state, ekf.covariance
    z = np.array([0.4, -0.2])
    up = ekf.update_gps(z, R)

    H = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    r = z - H @ x_m
    S = H @ P_m @ H.T + R
    K = P_m @ H.T @ np.linalg.inv(S)
    assert np.allclose(up.innovation, r)
    assert np.allclose(up.S, S)
    assert np.allclose(up.K, K)
    assert np.allclose(up.x_corrected, x_m + K @ r)
    assert np.allclose(up.P_updated, (np.eye(3) - K @ H) @ P_m)  # Joseph == simple form for optimal K
    assert np.allclose(ekf.state, up.x_corrected) and np.allclose(ekf.covariance, up.P_updated)
    assert up.nis == pytest.approx(float(r @ np.linalg.inv(S) @ r))


def test_gps_update_moves_estimate_towards_measurement_and_reduces_uncertainty():
    ekf = ExtendedKalmanFilter([0.0, 0.0, 0.0], P0)
    before = ekf.covariance
    up = ekf.update_gps([1.0, 1.0], R)
    assert 0.0 < up.x_corrected[0] < 1.0 and 0.0 < up.x_corrected[1] < 1.0
    assert ekf.covariance[0, 0] < before[0, 0] and ekf.covariance[1, 1] < before[1, 1]


def test_precise_gps_is_trusted_more_than_poor_gps():
    z = [2.0, 0.0]
    good = ExtendedKalmanFilter([0, 0, 0], P0)
    good.update_gps(z, np.eye(2) * 1e-4)
    bad = ExtendedKalmanFilter([0, 0, 0], P0)
    bad.update_gps(z, np.eye(2) * 100.0)
    assert good.state[0] > 1.99 and bad.state[0] < 0.05


def test_covariance_stays_symmetric_and_positive_semidefinite():
    rng = np.random.default_rng(3)
    ekf = ExtendedKalmanFilter([0.0, 0.0, 0.0], P0)
    for k in range(300):
        ekf.predict(0.5, 0.2 * np.sin(0.05 * k), 0.05, Q)
        if k % 10 == 0:
            ekf.update_gps(ekf.state[:2] + rng.normal(0, 0.5, 2), R)
        if k % 25 == 0:
            ekf.update_heading(ekf.state[2] + rng.normal(0, 0.05), np.array([[0.0025]]))
        P = ekf.covariance
        assert np.allclose(P, P.T, atol=1e-12)
        assert np.linalg.eigvalsh(P).min() > -1e-10


def test_heading_innovation_is_wrapped():
    ekf = ExtendedKalmanFilter([0.0, 0.0, 3.1], P0)
    up = ekf.update_heading(-3.1, np.array([[0.01]]))
    assert up.innovation[0] == pytest.approx(2 * np.pi - 6.2)  # ~0.083 rad, not -6.2
    assert abs(up.innovation[0]) < 0.1
    # The corrected yaw stays close to +/- pi instead of swinging through zero.
    assert abs(abs(ekf.state[2]) - np.pi) < 0.2


def test_gps_corrects_accumulated_dead_reckoning_drift():
    truth = np.array([0.0, 0.0, 0.0])
    ekf_gps = ExtendedKalmanFilter(truth, P0)
    ekf_dr = ExtendedKalmanFilter(truth, P0)
    for k in range(1, 401):
        truth = propagate_pose(truth, 0.5, 0.0, 0.05)
        biased_omega = 0.05  # gyro bias
        ekf_gps.predict(0.5, biased_omega, 0.05, Q)
        ekf_dr.predict(0.5, biased_omega, 0.05, Q)
        if k % 20 == 0:
            ekf_gps.update_gps(truth[:2], R)
    err_gps = np.linalg.norm(ekf_gps.state[:2] - truth[:2])
    err_dr = np.linalg.norm(ekf_dr.state[:2] - truth[:2])
    assert err_gps < 0.25 * err_dr


def test_singular_innovation_covariance_is_reported_not_ignored():
    ekf = ExtendedKalmanFilter([0, 0, 0], np.zeros((3, 3)))
    with pytest.raises(NumericalError):
        ekf.update_gps([1.0, 1.0], np.diag([1e-20, 1.0]))
    # The failed update must leave the filter untouched.
    assert np.allclose(ekf.state, 0.0)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        ExtendedKalmanFilter([0, 0], P0)
    with pytest.raises(ValueError):
        ExtendedKalmanFilter([0, 0, 0], -np.eye(3))  # not positive semidefinite
    with pytest.raises(ValueError):
        ExtendedKalmanFilter([0, 0, 0], np.array([[1, 2, 0], [0, 1, 0], [0, 0, 1.0]]))  # not symmetric
    ekf = ExtendedKalmanFilter([0, 0, 0], P0)
    with pytest.raises(ValueError):
        ekf.update_gps([np.nan, 0.0], R)
    with pytest.raises(ValueError):
        ekf.update_gps([0.0, 0.0], np.zeros((2, 2)))  # R must be positive definite
    with pytest.raises(ValueError):
        ekf.predict(np.nan, 0.0, 0.1, Q)
    with pytest.raises(ValueError):
        ekf.predict(0.5, 0.0, -0.1, Q)


def test_validate_covariance_accepts_semidefinite():
    A = validate_covariance(np.diag([1.0, 0.0, 2.0]), "A", 3)
    assert np.allclose(A, np.diag([1.0, 0.0, 2.0]))
