import numpy as np


from core.transformations import (
    euler_xyz_to_quaternion,
    quaternion_to_euler_xyz,
    rotate_vector_by_quaternion,
    sb_to_eci,
    eci_to_sb,
    quat_multiply,
    update_quaternion_by_angular_velocity,
    ecef_to_eci,
)


class _TimeStub:
    # gast in hours
    def __init__(self, gast_hours: float) -> None:
        self.gast = gast_hours


def test_euler_quat_roundtrip():
    angles = np.array([30.0, -15.0, 80.0])
    q = euler_xyz_to_quaternion(angles, degrees=True)
    out = quaternion_to_euler_xyz(q, degrees=True)
    assert np.allclose(angles, out, atol=1e-6)


def test_rotate_inverse_identity():
    v = np.array([0.2, -1.0, 3.0])
    q = euler_xyz_to_quaternion([10, 20, 30], degrees=True)
    v_b = rotate_vector_by_quaternion(v, q)

    # Inverse is the quaternion conjugate (for unit quaternions)
    q_inv = np.array([-q[0], -q[1], -q[2], q[3]])
    v_back = rotate_vector_by_quaternion(v_b, q_inv)

    assert np.allclose(v, v_back, atol=1e-6)


def test_quat_multiply_identity():
    q_id = np.array([0.0, 0.0, 0.0, 1.0])
    q = euler_xyz_to_quaternion([5, -7, 12], degrees=True)
    assert np.allclose(quat_multiply(q, q_id), q, atol=1e-12)
    assert np.allclose(quat_multiply(q_id, q), q, atol=1e-12)


def test_quat_deriv_and_update_normalizes():
    q = euler_xyz_to_quaternion([0, 0, 0], degrees=True)
    w = np.deg2rad(np.array([0.1, -0.2, 0.3]))
    q2 = update_quaternion_by_angular_velocity(q, w, dt=0.1)
    assert np.isclose(np.linalg.norm(q2), 1.0, atol=1e-9)


def test_ecef_to_eci_rotation_zero_and_quarter_turn():
    v = np.array([1.0, 0.0, 0.0])
    # gast=0h => no rotation
    v0 = ecef_to_eci(v, _TimeStub(gast_hours=0.0))
    assert np.allclose(v0, v, atol=1e-12)
    # gast=6h => +90° around Z (x->y)
    v90 = ecef_to_eci(v, _TimeStub(gast_hours=6.0))
    assert np.allclose(v90, np.array([0.0, 1.0, 0.0]), atol=1e-12)


def test_eci_sb_roundtrip():
    v = np.array([0.3, 0.4, -0.5])
    q = euler_xyz_to_quaternion([12, 34, 56], degrees=True)
    v_b = eci_to_sb(v, q)
    v_back = sb_to_eci(v_b, q)
    assert np.allclose(v, v_back, atol=1e-9)
