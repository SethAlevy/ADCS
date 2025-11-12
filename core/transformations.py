import numpy as np
import skyfield
from scipy.spatial.transform import Rotation as R
import skyfield.timelib


def enu_to_ecef(enu_vec: np.ndarray, lat_deg: float, lon_deg: float) -> np.ndarray:
    """
    Convert a vector from ENU (East-North-Up) to ECEF (Earth-Centered, Earth-Fixed).

    Args:
        enu_vec (np.ndarray): Vector in ENU frame.
        lat_deg (float): Latitude in degrees.
        lon_deg (float): Longitude in degrees.

    Returns:
        np.ndarray: Vector in ECEF frame.
    """
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    sφ, cφ = np.sin(lat), np.cos(lat)
    sλ, cλ = np.sin(lon), np.cos(lon)

    # Standard ENU→ECEF rotation (columns are E,N,U expressed in ECEF)
    rot_enu_to_ecef = R.from_matrix(
        np.array(
            [
                [-sλ, cλ, 0.0],
                [-sφ * cλ, -sφ * sλ, cφ],
                [cφ * cλ, cφ * sλ, sφ],
            ]
        )
    )
    return rot_enu_to_ecef.apply(enu_vec)


def ned_to_ecef(ned_vec: np.ndarray, lat_deg: float, lon_deg: float) -> np.ndarray:
    """
    Convert a vector from NED (North, East, Down) to ECEF using ENU as an intermediate.

    Args:
        ned_vec (np.ndarray): Vector in NED frame.
        lat_deg (float): Latitude in degrees.
        lon_deg (float): Longitude in degrees.

    Returns:
        np.ndarray: Vector in ECEF frame.
    """
    # NED → ENU: [N, E, D] → [E, N, U] = [E=NED[1], N=NED[0], U=-NED[2]]
    enu = np.array([ned_vec[1], ned_vec[0], -ned_vec[2]], dtype=float)
    return enu_to_ecef(enu, lat_deg, lon_deg)


def ecef_to_eci(ecef_vec: np.ndarray, time: skyfield.timelib) -> np.ndarray:
    """
    Convert a vector from ECEF (Earth-Centered, Earth-Fixed) to ECI
    (Earth-Centered Inertial) frame.

    Args:
        ecef_vec (np.ndarray): Vector in ECEF frame.
        time (skyfield.timelib): Skyfield Time object (UTC or TT).

    Returns:
        np.ndarray: Vector in ECI frame.
    """
    # Get GAST (Greenwich Apparent Sidereal Time) in radians
    gast_rad = time.gast * np.pi / 12  # GAST in hours, convert to radians

    # ECEF to ECI: rotate forward by GAST about Z axis
    rot = R.from_euler("z", gast_rad, degrees=False)
    return rot.apply(ecef_vec)


def eci_to_sbf(vec_eci: np.ndarray, quat_sb_from_eci: np.ndarray) -> np.ndarray:
    """
    Transform a vector from the ECI (Earth-Centered Inertial) frame to the SBF
    (Satellite Body Frame) using a quaternion.

    Args:
        vec_eci (np.ndarray): Vector in ECI frame.
        quat_sb_from_eci (np.ndarray): Quaternion representing rotation from ECI to
            satellite body frame, in [x, y, z, w] format.

    Returns:
        np.ndarray: Vector in satellite body frame.
    """
    rot = R.from_quat(quat_sb_from_eci)
    return rot.apply(vec_eci)


def sbf_to_eci(vec_sbf: np.ndarray, quat_sb_from_eci: np.ndarray) -> np.ndarray:
    """
    Transform a vector from the SBF (Satellite Body Frame) to the ECI
    (Earth-Centered Inertial) frame using a quaternion.
    The rotation from SBF to ECI is the inverse of the rotation from ECI to SBF.

    Args:
        vec_sbf (np.ndarray): Vector in SBF frame.
        quat_sb_from_eci (np.ndarray): Quaternion representing rotation from ECI to
            satellite body frame, in [x, y, z, w] format.

    Returns:
        np.ndarray: Vector in ECI frame.
    """
    # The inverse of a rotation is its conjugate.
    rot = R.from_quat(quat_sb_from_eci)
    inv_rot = rot.inv()
    return inv_rot.apply(vec_sbf)


def euler_xyz_to_quaternion(
    euler_angles: np.ndarray, degrees: bool = True
) -> np.ndarray:
    """
    Convert Euler angles in X-Y-Z convention to a quaternion.

    Args:
        euler_angles (np.ndarray): Euler angles [x1, y1, z1] in degrees or radians.
        degrees (bool): If True, input angles are in degrees. If False, in radians.

    Returns:
        np.ndarray: Quaternion as [x, y, z, w].
    """
    rot = R.from_euler("xyz", euler_angles, degrees=degrees)
    return rot.as_quat()


def quaternion_to_euler_xyz(quat, degrees: bool = True) -> np.ndarray:
    """
    Convert a quaternion to Euler angles in X-Y-Z convention (degrees).

    Args:
        quat (np.ndarray): Quaternion in [x, y, z, w] format.

    Returns:
        np.ndarray: Euler angles [x1, y1, z1] in degrees.
    """
    rot = R.from_quat(quat)
    return rot.as_euler("xyz", degrees=degrees)


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
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


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


def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to a quaternion.

    Args:
        rotation_matrix (np.ndarray): 3x3 rotation matrix.

    Returns:
        np.ndarray: Quaternion in [x, y, z, w] format.
    """
    rot = R.from_matrix(rotation_matrix)
    return rot.as_quat()


def rotate_vector_by_quaternion(
    vector: np.ndarray, quaternion: np.ndarray
) -> np.ndarray:
    """
    Rotate a vector by a quaternion.

    Args:
        vector (np.ndarray): Vector to be rotated.
        quaternion (np.ndarray): Quaternion in [x, y, z, w] format.

    Returns:
        np.ndarray: Rotated vector.
    """
    rot = R.from_quat(quaternion)
    return rot.apply(vector)


def earth_direction_body(
    position_eci: np.ndarray, quat_sb_from_eci: np.ndarray
) -> np.ndarray:
    """
    Return unit vector in body frame pointing toward Earth's center (nadir).

    Args:
        position_eci (np.ndarray): Satellite position in ECI frame.
        quat_sb_from_eci (np.ndarray): Quaternion rotating ECI -> body [x, y, z, w].

    Returns:
        np.ndarray: Unit Earth direction in body frame.
    """
    p = np.asarray(position_eci, dtype=float)
    n = np.linalg.norm(p)
    if n == 0.0:
        return np.array([0.0, 0.0, -1.0])  # arbitrary fallback
    to_earth_eci = -p / n
    to_earth_body = rotate_vector_by_quaternion(to_earth_eci, quat_sb_from_eci)
    return to_earth_body / (np.linalg.norm(to_earth_body) + 1e-20)


def vector_angular_noise(vec: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate the unit vector by a random axis with angular noise.

    Args:
        vec (np.ndarray): The input unit vector to be rotated.
        angle_deg (float): The angular noise in degrees.

    Returns:
        np.ndarray: The rotated vector.
    """
    angle_rad = np.deg2rad(angle_deg)
    # Generate a random axis perpendicular to vec
    rand_vec = np.random.randn(3)
    rand_vec = rand_vec / np.linalg.norm(rand_vec)
    axis = np.cross(vec, rand_vec)

    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        # If random vector is parallel, pick a fixed perpendicular axis
        axis = np.cross(vec, [1, 0, 0])
        axis_norm = np.linalg.norm(axis)
    axis = axis / axis_norm
    # Create rotation
    rot = R.from_rotvec(axis * angle_rad)
    return rot.apply(vec)


def sun_direction_body(
    sun_vector_eci: np.ndarray, quat_sb_from_eci: np.ndarray
) -> np.ndarray:
    """
    Return unit vector in body frame pointing toward the Sun.
    sun_vector_eci: Sun direction (or position difference) in ECI frame.
                    (Only direction is used; it will be normalized.)
    quat_sb_from_eci: Quaternion rotating ECI -> body [x, y, z, w].

    Returns:
        np.ndarray: Unit Sun direction in body frame.
    """
    v = np.asarray(sun_vector_eci, dtype=float)
    n = np.linalg.norm(v)
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0])  # fallback
    v_hat = v / n
    sun_body = rotate_vector_by_quaternion(v_hat, quat_sb_from_eci)
    return sun_body / np.linalg.norm(sun_body)
