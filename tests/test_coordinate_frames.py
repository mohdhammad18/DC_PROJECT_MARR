import numpy as np
import pytest

from src.coordinate_frames import (
    Pose2D,
    body_velocity_to_global,
    global_to_local,
    global_vector_to_local,
    local_to_global,
    local_vector_to_global,
    rotation_matrix,
    wrap_angle,
)


def test_rotation_matrix_is_orthonormal_with_unit_determinant():
    for psi in np.linspace(-6, 6, 25):
        R = rotation_matrix(psi)
        assert np.allclose(R @ R.T, np.eye(2), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)


def test_rotation_matrix_matches_definition():
    psi = 0.7
    expected = [[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]]
    assert np.allclose(rotation_matrix(psi), expected)


def test_local_to_global_known_case():
    # Robot at (2, 3) heading +90 deg: its forward axis points along +Y_G.
    p_g = local_to_global([1.0, 0.0], [2.0, 3.0], np.pi / 2)
    assert np.allclose(p_g, [2.0, 4.0])
    # Its left axis (Y_L) points along -X_G.
    p_g = local_to_global([0.0, 1.0], [2.0, 3.0], np.pi / 2)
    assert np.allclose(p_g, [1.0, 3.0])


def test_zero_heading_is_pure_translation():
    assert np.allclose(local_to_global([1.0, 2.0], [10.0, -5.0], 0.0), [11.0, -3.0])


def test_inverse_transformation_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        robot = rng.uniform(-10, 10, 2)
        psi = rng.uniform(-np.pi, np.pi)
        p_l = rng.uniform(-5, 5, 2)
        p_g = local_to_global(p_l, robot, psi)
        assert np.allclose(global_to_local(p_g, robot, psi), p_l)
        assert np.allclose(local_to_global(global_to_local(p_g, robot, psi), robot, psi), p_g)


def test_batch_transformation_matches_single():
    pts = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, -1.0]])
    batch = local_to_global(pts, [1.0, 1.0], 0.4)
    single = np.array([local_to_global(p, [1.0, 1.0], 0.4) for p in pts])
    assert batch.shape == (3, 2)
    assert np.allclose(batch, single)


def test_free_vectors_are_rotated_but_not_translated():
    v_g = local_vector_to_global([1.0, 0.0], np.pi / 2)
    assert np.allclose(v_g, [0.0, 1.0])
    assert np.allclose(global_vector_to_local(v_g, np.pi / 2), [1.0, 0.0])
    assert np.allclose(body_velocity_to_global(2.0, np.pi / 4), 2.0 * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)]))


def test_transform_preserves_distances():
    a, b = np.array([1.0, 2.0]), np.array([-3.0, 0.5])
    ga, gb = local_to_global(a, [4.0, 4.0], 1.1), local_to_global(b, [4.0, 4.0], 1.1)
    assert np.isclose(np.linalg.norm(ga - gb), np.linalg.norm(a - b))


@pytest.mark.parametrize("bad", [[1.0, 2.0, 3.0], [np.nan, 0.0], np.zeros((2, 3))])
def test_invalid_shapes_or_values_raise(bad):
    with pytest.raises(ValueError):
        local_to_global(bad, [0.0, 0.0], 0.0)


def test_wrap_angle_range_and_equivalence():
    angles = np.linspace(-20, 20, 801)
    wrapped = wrap_angle(angles)
    assert np.all(wrapped >= -np.pi) and np.all(wrapped < np.pi)
    # Wrapping never changes the angle modulo 2*pi.
    assert np.allclose(np.cos(wrapped), np.cos(angles)) and np.allclose(np.sin(wrapped), np.sin(angles))


def test_wrap_angle_known_values_and_types():
    assert wrap_angle(0.0) == 0.0
    assert np.isclose(wrap_angle(3 * np.pi / 2), -np.pi / 2)
    assert np.isclose(wrap_angle(-3 * np.pi / 2), np.pi / 2)
    assert np.isclose(wrap_angle(2 * np.pi + 0.1), 0.1)
    assert isinstance(wrap_angle(1.0), float)
    assert np.isnan(wrap_angle(np.nan))


def test_wrap_angle_difference_across_the_branch_cut():
    # 179 deg vs -179 deg are only 2 deg apart.
    diff = wrap_angle(np.radians(-179) - np.radians(179))
    assert np.isclose(np.degrees(diff), 2.0)


def test_pose2d_helpers():
    p = Pose2D.from_array([1, 2, 3])
    assert np.allclose(p.as_array(), [1, 2, 3]) and np.allclose(p.position, [1, 2])
    with pytest.raises(ValueError):
        Pose2D.from_array([1, 2])
