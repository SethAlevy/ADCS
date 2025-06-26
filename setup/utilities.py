import pandas as pd
import skyfield.api as skyfield
from templates.satellite_template import Satellite


def time_julian_date(satellite: Satellite, iteration: int = 0) -> float:
    # sourcery skip: inline-immediately-returned-variable
    """
    Convert the current epoch to Julian Date. Current date is the epoch
    of the TLE plus the number of seconds from simulation start. The
    Julian Date is a continuous count of days since the beginning of the Julian
    Period on January 1, 4713 BC at noon. It is used in astronomy
    and other fields to provide a uniform time scale for calculations.

    Args:
        satellite (Satelliten): The satellite object containing
        the TLE data.
        iteration (int): The current iteration of the simulation. Is also equal
        to the number of seconds from simulation start. Defaults to 0

    Returns:
        float: The Julian Date corresponding to the current epoch.
    """

    # Get the Skyfield Time object for the TLE epoch
    tle_epoch_time = satellite._satellite_model.epoch
    time_scale = skyfield.load.timescale()

    # Skyfield Time objects support adding seconds directly
    new_time = time_scale.tt_jd(tle_epoch_time.tt + iteration / 86400.0)
    return new_time


def initialize_state_vector(satellite: Satellite) -> pd.DataFrame:
    # sourcery skip: inline-immediately-returned-variable
    """
    Initialize the state vector of the satellite.

    Args:
        satellite (Satellite): The satellite object containing
        the TLE data and current status.

    Returns:
        pd.DataFrame: A DataFrame containing the initial state vector of the
        satellite. The state vector includes position, velocity (both x, y and
        z in GCRS), latitude, longitude and altitude.
    """
    # Get the initial position and velocity
    position = satellite.position(0)
    velocity = satellite.linear_velocity(0)

    # Create a DataFrame to store the state vector
    state_vector = pd.DataFrame(
        {
            "position_x": position[0],
            "position_y": position[1],
            "position_z": position[2],
            "velocity_x": velocity[0],
            "velocity_y": velocity[1],
            "velocity_z": velocity[2],
            "latitude": satellite.latitude(0),
            "longitude": satellite.longitude(0),
            "altitude": satellite.altitude(0),
            "euler_z1": satellite.euler_angles()[0],
            "euler_x1": satellite.euler_angles()[1],
            "euler_z2": satellite.euler_angles()[2],
        },
        index=[0],
    )

    return state_vector


def update_state_vector(
    satellite: Satellite, state_vector: pd.DataFrame, iteration: int
) -> pd.DataFrame:
    """
    Update the state vector of the satellite.

    Args:
        satellite (Satellite): The satellite object containing
        the TLE data and current status.
        state_vector (pd.DataFrame): The DataFrame containing the current state
        vector of the satellite.
        iteration (int): The current iteration of the simulation. Is also equal
        to the number of seconds from simulation start.

    Returns:
        pd.DataFrame: A DataFrame containing the updated state vector of the
        satellite. The state vector includes position, velocity (both x, y and
        z in GCRS), latitude, longitude and altitude.
    """
    # Get the new position and velocity
    position = satellite.position(iteration)
    velocity = satellite.linear_velocity(iteration)

    # Update the DataFrame with the new values
    state_vector.loc[iteration] = {
        "position_x": position[0],
        "position_y": position[1],
        "position_z": position[2],
        "velocity_x": velocity[0],
        "velocity_y": velocity[1],
        "velocity_z": velocity[2],
        "latitude": satellite.latitude(iteration),
        "longitude": satellite.longitude(iteration),
        "altitude": satellite.altitude(iteration),
        "euler_z1": satellite.euler_angles()[0],
        "euler_x1": satellite.euler_angles()[1],
        "euler_z2": satellite.euler_angles()[2],
    }

    return state_vector


def get_lla(satellite: Satellite, iteration: int) -> tuple:
    """
    Get the latitude, longitude and altitude of the satellite.

    Args:
        satellite (Satellite): The satellite object containing
        the TLE data and current status.
        iteration (int): The current iteration of the simulation. Is also equal
        to the number of seconds from simulation start.

    Returns:
        tuple: latitude, longitude and altitude of the satellite.
    """
    lat = satellite.latitude(iteration)
    lon = satellite.longitude(iteration)
    alt = satellite.altitude(iteration)

    return lat, lon, alt
