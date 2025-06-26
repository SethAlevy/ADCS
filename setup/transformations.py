import numpy as np
import skyfield
from scipy.spatial.transform import Rotation as R


def enu_to_ecef(enu_vec: np.ndarray, lat_deg: float, lon_deg: float) -> np.ndarray:
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


def ecef_to_eci(ecef_vec: np.ndarray, time: skyfield.timelib.Time) -> np.ndarray:
    """
    Convert a vector from ECEF (Earth-Centered, Earth-Fixed) to ECI (Earth-Centered Inertial) frame.

    Args:
        ecef_vec (np.ndarray): Vector in ECEF frame (shape: (3,))
        time (skyfield.timelib.Time): Skyfield Time object (UTC or TT)

    Returns:
        np.ndarray: Vector in ECI frame (shape: (3,))
    """
    # Get GAST (Greenwich Apparent Sidereal Time) in radians
    gast_rad = time.gast * np.pi / 12  # GAST in hours, convert to radians

    # ECEF to ECI: rotate by GAST about Z axis
    rot = R.from_euler("z", gast_rad, degrees=False)
    return rot.apply(ecef_vec)


def euler_zxz_to_quaternion(z1: float, x: float, z2: float, degrees: bool = True) -> np.ndarray:
    """
    Convert Euler angles (Z-X-Z convention) to a quaternion.

    Args:
        z1 (float): First rotation about Z axis.
        x (float): Rotation about X axis.
        z2 (float): Second rotation about Z axis.
        degrees (bool): If True, input angles are in degrees. If False, in radians.

    Returns:
        np.ndarray: Quaternion as [x, y, z, w] (scipy format).
    """
    rot = R.from_euler("zxz", [z1, x, z2], degrees=degrees)
    return rot.as_quat()


def eci_to_sb(vec_eci, quat_sb_from_eci):
    """
    Transform a vector from the ECI frame to the satellite body frame using a quaternion.

    Args:
        vec_eci (np.ndarray): Vector in ECI frame (shape: (3,))
        quat_sb_from_eci (np.ndarray): Quaternion representing rotation from ECI to satellite body frame,
                                       in [x, y, z, w] format (scipy convention).

    Returns:
        np.ndarray: Vector in satellite body frame (shape: (3,))
    """
    # The quaternion should represent the rotation from ECI to SB.
    rot = R.from_quat(quat_sb_from_eci)
    return rot.apply(vec_eci)