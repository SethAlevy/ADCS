import numpy as np
import skyfield
from scipy.spatial.transform import Rotation as R
import skyfield.timelib


def enu_to_ecef(
    enu_vec: np.ndarray, lat_deg: float, lon_deg: float
) -> np.ndarray:
    """
    Convert a vector from ENU to ECEF.

    Args:
        enu_vec (np.ndarray): Vector in ENU frame (shape: (3,))
        lat_deg (float): Latitude in degrees.
        lon_deg (float): Longitude in degrees.
    """
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    rot_z = R.from_euler("z", lon, degrees=False)
    rot_y = R.from_euler("y", -(np.pi / 2 - lat), degrees=False)
    enu_to_ecef_rot = rot_z * rot_y
    return enu_to_ecef_rot.apply(enu_vec)


def ecef_to_eci(
    ecef_vec: np.ndarray, time: skyfield.timelib
) -> np.ndarray:
    """
    Convert a vector from ECEF (Earth-Centered, Earth-Fixed) to ECI
    (Earth-Centered Inertial) frame.

    Args:
        ecef_vec (np.ndarray): Vector in ECEF frame (shape: (3,))
        time (skyfield.timelib): Skyfield Time object (UTC or TT)

    Returns:
        np.ndarray: Vector in ECI frame (shape: (3,))
    """
    # Get GAST (Greenwich Apparent Sidereal Time) in radians
    gast_rad = time.gast * np.pi / 12  # GAST in hours, convert to radians

    # ECEF to ECI: rotate by GAST about Z axis
    rot = R.from_euler("z", gast_rad, degrees=False)
    return rot.apply(ecef_vec)


def euler_xyz_to_quaternion(
    euler_angles: np.ndarray, degrees: bool = True
) -> np.ndarray:
    """
    Convert Euler angles in Z-Y-X convention to a quaternion.

    Args:
        euler_angles (np.ndarray): Euler angles [x1, y1, z1] in degrees or radians.
        degrees (bool): If True, input angles are in degrees. If False, in radians.

    Returns:
        np.ndarray: Quaternion as [x, y, z, w] (scipy format).
    """
    rot = R.from_euler("xyz", euler_angles, degrees=degrees)
    return rot.as_quat()


def quaternion_to_euler_xyz(quat, degrees: bool = True) -> np.ndarray:
    """
    Convert a quaternion to Euler angles in X-Y-Z convention (degrees).

    Args:
        quat (np.ndarray): Quaternion in [x, y, z, w] format (scipy convention).

    Returns:
        np.ndarray: Euler angles [x1, y1, z1] in degrees.
    """
    rot = R.from_quat(quat)
    euler_angles = rot.as_euler("xyz", degrees=degrees)
    return euler_angles


def eci_to_sbf(vec_eci: np.ndarray, quat_sb_from_eci: np.ndarray) -> np.ndarray:
    """
    Transform a vector from the ECI frame to the Satellite Body Frame (SBF)
    using a quaternion.

    Args:
        vec_eci (np.ndarray): Vector in ECI frame (shape: (3,))
        quat_sb_from_eci (np.ndarray): Quaternion representing rotation from ECI to
            satellite body frame, in [x, y, z, w] format (scipy convention).

    Returns:
        np.ndarray: Vector in satellite body frame (shape: (3,))
    """
    # The quaternion should represent the rotation from ECI to SB.
    rot = R.from_quat(quat_sb_from_eci)
    return rot.apply(vec_eci)


def quat_deriv(quaternion: np.ndarray, angular_velocity: np.ndarray) -> np.ndarray:
    """
    Compute the quaternion time derivative given angular velocity.

    Args:
        quaternion (np.ndarray): Quaternion [x, y, z, w], should be normalized.
        angular_velocity (np.ndarray): Angular velocity [wx, wy, wz] in rad/s
            (body frame).

    Returns:
        np.ndarray: Quaternion derivative [x_dot, y_dot, z_dot, w_dot]
    """
    omega_quat = np.array([*angular_velocity, 0.0])  # [wx, wy, wz, 0]
    return 0.5 * quat_multiply(quaternion, omega_quat)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamilton product of two quaternions [x, y, z, w].

    Args:
        q1 (np.ndarray): First quaternion [x1, y1, z1, w1].
        q2 (np.ndarray): Second quaternion [x2, y2, z2, w2].

    Returns:
        np.ndarray: Resulting quaternion [x, y, z, w].
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])


def update_quaternion_by_angular_velocity(
    quaternion: np.ndarray, angular_velocity: np.ndarray, dt: float = 1.0
) -> np.ndarray:
    """
    Update the quaternion based on the angular velocity and time step.

    Args:
        quaternion (np.ndarray): Current quaternion in [x, y, z, w] format.
        angular_velocity (np.ndarray): Angular velocity in radians/s.
        dt (float): Time step in seconds.

    Returns:
        np.ndarray: Updated quaternion in [x, y, z, w] format.
    """
    q_dot = quat_deriv(quaternion, angular_velocity)
    quaternion_new = quaternion + q_dot * dt
    quaternion_new /= np.linalg.norm(quaternion_new)  # Normalize the quaternion
    return quaternion_new


def rotation_matrix_to_quaternion(
    rotation_matrix: np.ndarray
) -> np.ndarray:
    """
    Convert a rotation matrix to a quaternion.

    Args:
        rotation_matrix (np.ndarray): 3x3 rotation matrix.

    Returns:
        np.ndarray: Quaternion in [x, y, z, w] format.
    """
    rot = R.from_matrix(rotation_matrix)
    return rot.as_quat()  # [x, y, z, w]


def rotate_vector_by_quaternion(
    vector: np.ndarray, quaternion: np.ndarray
) -> np.ndarray:
    """
    Rotate a vector by a quaternion.

    Args:
        vector (np.ndarray): Vector to be rotated (shape: (3,)).
        quaternion (np.ndarray): Quaternion in [x, y, z, w] format.

    Returns:
        np.ndarray: Rotated vector (shape: (3,)).
    """
    rot = R.from_quat(quaternion)
    return rot.apply(vector)