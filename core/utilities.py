import pandas as pd
import numpy as np
import skyfield.api as skyfield
from templates.satellite_template import Satellite


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


def initialize_state_vector(satellite: Satellite) -> pd.DataFrame:
    """
    Initialize the state vector of the satellite.

    Args:
        satellite (Satellite): The satellite object containing
            the TLE data and current status.

    Returns:
        pd.DataFrame: A DataFrame containing the initial state vector of the
            satellite. The state vector includes position, velocity, latitude,
            longitude and altitude etc.
    """
    # Get the initial position and velocity
    position = satellite.position
    velocity = satellite.linear_velocity

    mag_field_sbf, mag_field_eci = satellite.magnetic_field
    sun_vector_sbf, sun_vector_eci = satellite.sun_vector

    euler_angles = satellite.euler_angles
    angular_velocity = satellite.angular_velocity

    # Create a DataFrame to store the state vector
    state_vector = pd.DataFrame(
        {
            "position_x": position[0],
            "position_y": position[1],
            "position_z": position[2],
            "velocity_x": velocity[0],
            "velocity_y": velocity[1],
            "velocity_z": velocity[2],
            "latitude": satellite.latitude,
            "longitude": satellite.longitude,
            "altitude": satellite.altitude,
            "euler_x1": euler_angles[0],
            "euler_y1": euler_angles[1],
            "euler_z1": euler_angles[2],
            "wx": angular_velocity[0],
            "wy": angular_velocity[1],
            "wz": angular_velocity[2],
            "mag_field_sbf_x": mag_field_sbf[0],
            "mag_field_sbf_y": mag_field_sbf[1],
            "mag_field_sbf_z": mag_field_sbf[2],
            "mag_field_eci_x": mag_field_eci[0],
            "mag_field_eci_y": mag_field_eci[1],
            "mag_field_eci_z": mag_field_eci[2],
            "sun_vector_sbf_x": sun_vector_sbf[0],
            "sun_vector_sbf_y": sun_vector_sbf[1],
            "sun_vector_sbf_z": sun_vector_sbf[2],
            "sun_vector_eci_x": sun_vector_eci[0],
            "sun_vector_eci_y": sun_vector_eci[1],
            "sun_vector_eci_z": sun_vector_eci[2],
            "torque_x": satellite.torque[0],
            "torque_y": satellite.torque[1],
            "torque_z": satellite.torque[2],
            "angular_acceleration_x": satellite._angular_acceleration[0],
            "angular_acceleration_y": satellite._angular_acceleration[1],
            "angular_acceleration_z": satellite._angular_acceleration[2],
            "pointing_error": satellite.pointing_error_angle
        },
        index=[0],
    )

    return state_vector


def update_state_vector(
        satellite: Satellite,
        state_vector: pd.DataFrame
) -> pd.DataFrame:
    """
    Update the state vector of the satellite.

    Args:
        satellite (Satellite): The satellite object containing
            the TLE data and current status.
        state_vector (pd.DataFrame): The DataFrame containing the current state
            vector of the satellite.

    Returns:
        pd.DataFrame: A DataFrame containing the updated state vector of the
            satellite. The state vector includes position, velocity, latitude,
            longitude and altitude etc.
    """
    # Get the new position and velocity
    position = satellite.position
    velocity = satellite.linear_velocity

    mag_field_sbf, mag_field_eci = satellite.magnetic_field
    sun_vector_sbf, sun_vector_eci = satellite.sun_vector

    euler_angles = satellite.euler_angles
    angular_velocity = satellite.angular_velocity

    # Update the DataFrame with the new values
    state_vector.loc[satellite.iteration] = {
        "position_x": position[0],
        "position_y": position[1],
        "position_z": position[2],
        "velocity_x": velocity[0],
        "velocity_y": velocity[1],
        "velocity_z": velocity[2],
        "latitude": satellite.latitude,
        "longitude": satellite.longitude,
        "altitude": satellite.altitude,
        "euler_x1": euler_angles[0],
        "euler_y1": euler_angles[1],
        "euler_z1": euler_angles[2],
        "wx": angular_velocity[0],
        "wy": angular_velocity[1],
        "wz": angular_velocity[2],
        "mag_field_sbf_x": mag_field_sbf[0],
        "mag_field_sbf_y": mag_field_sbf[1],
        "mag_field_sbf_z": mag_field_sbf[2],
        "mag_field_eci_x": mag_field_eci[0],
        "mag_field_eci_y": mag_field_eci[1],
        "mag_field_eci_z": mag_field_eci[2],
        "sun_vector_sbf_x": sun_vector_sbf[0],
        "sun_vector_sbf_y": sun_vector_sbf[1],
        "sun_vector_sbf_z": sun_vector_sbf[2],
        "sun_vector_eci_x": sun_vector_eci[0],
        "sun_vector_eci_y": sun_vector_eci[1],
        "sun_vector_eci_z": sun_vector_eci[2],
        "torque_x": satellite.torque[0],
        "torque_y": satellite.torque[1],
        "torque_z": satellite.torque[2],
        "angular_acceleration_x": satellite._angular_acceleration[0],
        "angular_acceleration_y": satellite._angular_acceleration[1],
        "angular_acceleration_z": satellite._angular_acceleration[2],
        "pointing_error": satellite.pointing_error_angle
    }

    return state_vector


def get_lla(satellite: Satellite) -> tuple:
    """
    Get the latitude, longitude and altitude of the satellite.

    Args:
        satellite (Satellite): The satellite object containing
            the TLE data and current status.

    Returns:
        tuple: latitude, longitude and altitude of the satellite.
    """
    lat = satellite.latitude(satellite.iteration)
    lon = satellite.longitude(satellite.iteration)
    alt = satellite.altitude(satellite.iteration)

    return lat, lon, alt


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
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def filter_significant_digits(array: np.ndarray, sig_digits: int) -> np.ndarray:
    """
    Filter the array to have a specified number of significant digits.

    Args:
        array (np.ndarray): Input array of floats.
        sig_digits (int): Number of significant digits to retain.

    Returns:
        np.ndarray: Array with values formatted to the specified significant digits.
    """
    format_str = f"{{:.{sig_digits-1}e}}"  # Scientific notation
    return np.array([float(format_str.format(x)) for x in array])


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


def limit_norm(vector: np.ndarray, cap: float) -> np.ndarray:
    """
    Limit the Euclidean norm of a vector to 'cap' without changing direction.
    If cap <= 0 or ||v|| <= cap, returns the original vector.

    Args:
        vector (np.ndarray): Input vector.
        cap (float): Maximum allowed norm.

    Returns:
        np.ndarray: Vector with limited norm.
    """
    n = float(np.linalg.norm(vector))
    if cap <= 0.0 or n <= cap:
        return vector
    return vector * (cap / n)


def calculate_pointing_error(
    target_vector: np.ndarray,
    current_vector: np.ndarray
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
    angle_deg = np.degrees(angle_rad)
    return angle_deg
