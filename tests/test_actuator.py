import numpy as np
import pytest

from spacecraft.actuator import MagnetorquerImplementation


class FakeSV:
    def register_value(self, *args, **kwargs): pass


class FakeSatellite:
    def __init__(self):
        self.inertia_matrix = np.eye(3)
        self.mass = 1.0
        self._angular_velocity = np.array([0.0, 0.0, 0.0])
        self._state_vector = FakeSV()

    @property
    def angular_velocity(self):
        return self._angular_velocity

    @property
    def magnetic_field(self):
        # tuple (sb, eci) in nT for constructor init
        return np.array([14000.0, -3500.0, -34000.0]) * 1e-9, np.array([14000.0, -3500.0, -34000.0]) * 1e-9


class FakeSetup:
    magnetorquer_params = {
        "Coils": 100,
        "CoilArea": 1.0,        # cm^2
        "MaxCurrent": 0.25,    # A
        "SafetyFactor": 0.8,
        "AlphaCap": 0.0,
    }
    b_dot_parameters = {
        "Gain": 2000.0,
        "AngularVelocityRef": [1.0, 1.0, 1.0],
        "Alpha": 1.0,
        "MagneticFieldRef": 45000.0,  # nT
        "Beta": 1.0,
        "ProportionalGain": 0.0,
    }
    b_cross_parameters = {
        "AlignGain": 1.0,
        "ProportionalGain": 1.0,
    }
    iterations_info = {"Step": 1.0}


@pytest.fixture()
def magnetorquer():
    return MagnetorquerImplementation(FakeSetup(), FakeSatellite())


def test_b_decompose(magnetorquer):
    magnetic_field_body_nt = np.array([14000.0, -3500.0, -34000.0]) * 1e-9
    angular_velocity_body_rad_s = np.array([0.004, 0.006, 0.008])

    (
        b_vector_nt,
        b_magnitude_nt,
        b_unit_vector,
        inv_b_magnitude_sq,
        omega_parallel_rad_s,
        omega_perpendicular_rad_s,
    ) = magnetorquer._b_decompose(magnetic_field_body_nt, angular_velocity_body_rad_s)

    # Expected values
    expected_b_magnitude = np.linalg.norm(magnetic_field_body_nt)
    expected_b_unit = magnetic_field_body_nt / expected_b_magnitude
    expected_inv_b_mag_sq = 1.0 / (expected_b_magnitude ** 2)

    omega_dot_bhat = float(np.dot(angular_velocity_body_rad_s, expected_b_unit))
    expected_omega_parallel = omega_dot_bhat * expected_b_unit
    expected_omega_perp = angular_velocity_body_rad_s - expected_omega_parallel

    # Assertions
    assert np.allclose(b_vector_nt, magnetic_field_body_nt)
    assert np.isclose(b_magnitude_nt, expected_b_magnitude, rtol=0, atol=1e-9)
    assert np.allclose(b_unit_vector, expected_b_unit, atol=1e-12)
    assert np.isclose(inv_b_magnitude_sq, expected_inv_b_mag_sq, rtol=0, atol=1e-18)

    # Decomposition checks
    assert np.allclose(omega_parallel_rad_s, expected_omega_parallel, atol=1e-12)
    assert np.allclose(omega_perpendicular_rad_s, expected_omega_perp, atol=1e-12)
    assert np.allclose(omega_parallel_rad_s + omega_perpendicular_rad_s,
                       angular_velocity_body_rad_s, atol=1e-12)
    assert np.isclose(np.dot(omega_perpendicular_rad_s,
                      expected_b_unit), 0.0, atol=1e-12)


def test_compute_error_vec_and_theta_90deg(magnetorquer):
    align_axis = np.array([1.0, 0.0, 0.0])
    target_axis = np.array([0.0, 1.0, 0.0])
    err_vec, theta = magnetorquer._compute_error_vec_and_theta(align_axis, target_axis)
    assert np.isclose(theta, np.pi / 2, atol=1e-6)
    assert np.isclose(np.linalg.norm(err_vec), np.pi / 2, atol=1e-6)


def test_filtered_derivative_basic(magnetorquer):
    magnetic_field_1 = np.array([14000.0, -3500.0, -34000.0]) * 1e-9
    magnetic_field_2 = np.array([14500.0, -3800.0, -34300.0]) * 1e-9
    dt = 2.0
    alpha = 0.7

    # first call: raw derivative vs the previous field BEFORE the call
    prev_field = magnetorquer.magnetic_field_prev.copy()
    db_dt_1 = magnetorquer.filtered_derivative(magnetic_field_1, dt, alpha=alpha)
    assert np.allclose(db_dt_1, (magnetic_field_1 - prev_field) / dt)

    # second call: filtered average of new derivative and prev
    raw_2 = (magnetic_field_2 - magnetic_field_1) / dt
    db_dt_2 = magnetorquer.filtered_derivative(magnetic_field_2, dt, alpha=alpha)
    db_dt_2_expected = alpha * raw_2 + (1 - alpha) * db_dt_1
    assert np.allclose(db_dt_2, db_dt_2_expected)


def test_apply_torquer_with_saturation_caps_current(magnetorquer):
    # command extremely large dipole to force saturation
    commanded_dipole = np.array([1e6, -1e6, 1e6])
    currents, effective_dipole = magnetorquer.apply_torquer_with_saturation(
        commanded_dipole)
    current_limit = magnetorquer.max_current * magnetorquer.safety_factor
    assert np.all(np.abs(currents) <= current_limit)
    # resulting dipole should be <= max achievable
    max_dipole = magnetorquer.no_of_coils * magnetorquer.coil_area * \
        magnetorquer.max_current * magnetorquer.safety_factor
    assert np.linalg.norm(effective_dipole) <= max_dipole
