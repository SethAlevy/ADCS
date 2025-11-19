import numpy as np
import skyfield.api as skyfield
from templates.satellite_template import Satellite
from core.logger import log


def time_julian_date(satellite: Satellite) -> skyfield.Time:
    """
    Convert the current epoch to Julian Date. Current date is the epoch
    of the TLE plus the number of seconds from simulation start as a fraction of a day.
    The Julian Date is a continuous count of days since the beginning of the Julian
    Period on January 1, 4713 BC at noon. It is used in astronomy
    and other fields to provide a uniform time scale for calculations.

    Args:
        satellite (Satellite): The satellite object containing the TLE data.

    Returns:
        skyfield.Time: The Julian Date corresponding to the current epoch.
    """

    # Get the Skyfield Time object for the TLE epoch
    tle_epoch_time = satellite._satellite_model.epoch
    time_scale = skyfield.load.timescale()

    # Skyfield Time objects support adding seconds directly
    new_time = time_scale.tt_jd(tle_epoch_time.tt + satellite.iteration / 86400.0)
    return new_time


def basic_state_vector(satellite: Satellite) -> None:
    """
    Add the most important parameters to the state vector.

    Args:
        satellite (Satellite): The satellite object containing
            the TLE data and current status.
    """
    position = satellite.position
    velocity = satellite.linear_velocity

    mag_field_sb, mag_field_eci = satellite.magnetic_field
    sun_vector_sb, sun_vector_eci = satellite.sun_vector

    euler_angles = satellite.euler_angles
    angular_velocity = satellite.angular_velocity

    satellite._state_vector.register_vector(
        "position", position, labels=["x", "y", "z"]
    )
    satellite._state_vector.register_vector(
        "velocity", velocity, labels=["x", "y", "z"]
    )
    satellite._state_vector.register_value("latitude", satellite.latitude)
    satellite._state_vector.register_value("longitude", satellite.longitude)
    satellite._state_vector.register_value("altitude", satellite.altitude)
    satellite._state_vector.register_vector(
        "euler_angles", euler_angles, labels=["x1", "y1", "z1"]
    )
    satellite._state_vector.register_vector(
        "angular_velocity", angular_velocity, labels=["x", "y", "z"]
    )
    satellite._state_vector.register_vector(
        "magnetic_field_sb", mag_field_sb, labels=["x", "y", "z"]
    )
    satellite._state_vector.register_vector(
        "magnetic_field_eci", mag_field_eci, labels=["x", "y", "z"]
    )
    satellite._state_vector.register_vector(
        "sun_vector_sb", sun_vector_sb, labels=["x", "y", "z"]
    )
    satellite._state_vector.register_vector(
        "sun_vector_eci", sun_vector_eci, labels=["x", "y", "z"]
    )
    satellite._state_vector.register_vector(
        "torque", satellite.torque, labels=["x", "y", "z"]
    )
    satellite._state_vector.register_vector(
        "angular_acceleration", satellite._angular_acceleration, labels=["x", "y", "z"]
    )
    satellite._state_vector.register_value(
        "pointing_error", satellite.pointing_error_angle
    )
    satellite._state_vector.register_vector(
        "quaternion", satellite.quaternion, labels=["x", "y", "z", "w"]
    )


def get_lla(satellite: Satellite) -> tuple:
    """
    Get the latitude, longitude and altitude of the satellite.

    Args:
        satellite (Satellite): The satellite object containing
            the TLE data and current status.

    Returns:
        tuple: latitude, longitude and altitude of the satellite.
    """
    latitude = satellite.latitude(satellite.iteration)
    longitude = satellite.longitude(satellite.iteration)
    altitude = satellite.altitude(satellite.iteration)

    return latitude, longitude, altitude


def rad_to_degrees(values: list | np.ndarray) -> np.ndarray:
    """
    Convert a list or numpy array of floats from radians to degrees.

    Args:
        values (list or np.ndarray): Iterable of floats in radians.

    Returns:
        np.ndarray: Array of floats in degrees.
    """
    return np.degrees(values)


def degrees_to_rad(values: list | np.ndarray) -> np.ndarray:
    """
    Convert a list or numpy array of floats from degrees to radians.

    Args:
        values (list or np.ndarray): Iterable of floats in degrees.

    Returns:
        np.ndarray: Array of floats in radians.
    """
    return np.radians(values)


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    Args:
        v (np.ndarray): Input vector.

    Returns:
        np.ndarray: Normalized vector.
    """
    return v / np.linalg.norm(v)


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Compute the skew-symmetric matrix of a 3D vector.

    Args:
        v (np.ndarray): Input vector.

    Returns:
        np.ndarray: Skew-symmetric matrix.
    """
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def filter_decimal_places(array: np.ndarray, decimal_places: int) -> np.ndarray:
    """
    Round each element to specified number of decimal places

    Args:
        array (np.ndarray): Input numpy array or list
        decimal_places (int): Number of decimal places to keep

    Returns:
        np.ndarray: Filtered array with specified precision
    """
    return np.round(array, decimal_places)


def limit_norm(vector: np.ndarray, cap: float, eps: float = 1e-12) -> np.ndarray:
    """
    Uniformly scale 'vector' so that its L2 norm <= cap.
    Returns zeros if cap <= 0 or cap is not finite.

    Args:
        vector (np.ndarray): Input vector.
        cap (float): Maximum allowed norm.
        eps (float, optional): Small value to prevent division by zero.

    Returns:
        np.ndarray: Vector with limited norm.
    """
    if vector.ndim == 0:
        vector = np.array([float(vector)], dtype=float)

    if not np.isfinite(cap) or cap <= 0.0:
        return np.zeros_like(vector)

    n = float(np.linalg.norm(vector))
    return vector.copy() if not np.isfinite(n) \
        or n <= cap else vector * (cap / max(n, eps))


def calculate_pointing_error(
    target_vector: np.ndarray, current_vector: np.ndarray
) -> float:
    """
    Calculate the pointing error angle between two vectors in degrees.

    Args:
        target_vector (np.ndarray): Target direction vector.
        current_vector (np.ndarray): Current direction vector.

    Returns:
        float: Pointing error angle in degrees.
    """
    target_vector = target_vector / np.linalg.norm(target_vector)
    current_vector = current_vector / np.linalg.norm(current_vector)
    dot_product = np.clip(np.dot(target_vector, current_vector), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)


def log_init_state(setup) -> None:
    """
    Log initial simulation state.

    Args:
        setup: Simulation setup object.
    """
    log("Simulation initialized with the following parameters:")
    log("Number of iterations: %s", setup.iterations_info["Stop"])
    log("Satellite mass: %s kg", setup.satellite_params["Mass"])
    log("Satellite inertia: %s kg*m^2", setup.satellite_params["Inertia"])
    log("Initial angular velocity: %s deg/s", setup.angular_velocity)
    log("Initial attitude (Euler angles): %s deg", setup.euler_angles)
    log("Selected sensor fusion algorithm: %s", setup.sensor_fusion_algorithm)
    log("Magnetometer noise: %s nT", setup.magnetometer["AbsoluteNoise"])
    log("Sunsensor noise: %s deg", setup.sunsensor["AngularNoise"])
    log("Sensor on time: %s seconds, actuator on time: %s seconds \n",
        setup.sensors_on_time, setup.actuators_on_time)
