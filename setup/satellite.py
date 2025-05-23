from abc import ABC
from abc import abstractmethod
import numpy as np
import pandas as pd
from setup.two_line_element import TwoLineElement
from setup.initial_settings import SimulationSetup
from pathlib import Path
import sgp4.api as sgp
import skyfield.api as skyfield
from scipy.spatial.transform import Rotation
import setup.utilities as ut


class Satellite(ABC):
    """
    Abstract class for satellite.
    """

    @property
    @abstractmethod
    def mass(self) -> float:
        """
        Mass of the satellite in kg.
        """
        pass

    @property
    @abstractmethod
    def inertia_matrix(self) -> np.ndarray:
        """
        Inertia matrix of the satellite in kg*m^2.
        """
        pass

    @abstractmethod
    def position(self) -> np.ndarray:
        """
        Position of the satellite in m.
        """
        pass

    @abstractmethod
    def linear_velocity(self) -> np.ndarray:
        """
        Linear velocity of the satellite in m/s.
        """
        pass

    @abstractmethod
    def latitude(self) -> np.ndarray:
        """
        Latitude of the satellite in degrees.
        """
        pass

    @abstractmethod
    def longitude(self) -> np.ndarray:
        """
        Longitude of the satellite in degrees.
        """
        pass

    @abstractmethod
    def altitude(self) -> np.ndarray:
        """
        Altitude of the satellite in m.
        """
        pass

    @property
    @abstractmethod
    def angular_velocity(self) -> np.ndarray:
        """
        Angular velocity of the satellite in rad/s.
        """
        pass
    
    @property
    @abstractmethod
    def reference_frame(self) -> str:
        """
        Reference frame of the satellite.
        """
        pass

    @property
    def two_line_element(self) -> TwoLineElement:
        """
        Two-line element set (TLE) of the satellite. Imported from file as object.
        """
        pass

 
class SatelliteImplementation(Satellite):
    """
    Implementation of the Satellite class.
    """

    def __init__(
            self, 
            setup: SimulationSetup = None, 
            tle: TwoLineElement = None, 
    ):
        """
        Initialize the satellite using json file and TLE file.
        """
        self.setup = setup
        self._angular_velocity = self.setup.angular_velocity

        self._two_line_element = tle

        self._satellite_model = skyfield.EarthSatellite(
            self.two_line_element.line_1, self.two_line_element.line_2
        )

    @property
    def mass(self) -> float:
        return self.setup.satellite_params[0]

    @property
    def inertia_matrix(self) -> np.ndarray:
        return self.setup.satellite_params[1]

    def position(self, iteration) -> np.ndarray:
        """
        Position of the satellite obtained using the skyfield library.
        By default returns GCRS (Geocentric Celestial Reference System)
        which is an ECI (Earth-Centered Inertial) frame (fixed to the stars)
        almost similar to J2000 frame. Distance is given in km and calculated
        for the given simulation time.

        Args:
            iteration (int): The current iteration of the simulation. Equals
            the time in seconds from its start.

        Returns:
            np.ndarray: X, Y and Z position of the satellite in km.
        """
        julian_date = ut.time_to_julian_date(self, iteration)
        return self._satellite_model.at(julian_date).position.km
    
    def linear_velocity(self, iteration) -> np.ndarray:
        """
        Linear velocity of the satellite obtained using the skyfield library.
        By default returns GCRS (Geocentric Celestial Reference System)
        which is an ECI (Earth-Centered Inertial) frame (fixed to the stars)
        almost similar to J2000 frame. Distance is given in km and calculated
        for the given simulation time.

        Args:
            iteration (int): The current iteration of the simulation. Equals
            the time in seconds from its start.

        Returns:
            np.ndarray: X, Y and Z velocity of the satellite in km/s.
        """
        julian_date = ut.time_to_julian_date(self, iteration)
        return self._satellite_model.at(julian_date).velocity.km_per_s
    
    def latitude(self, iteration) -> np.ndarray:
        """
        Latitude of the satellite in degrees calculated for the
        given simulation time and using WGS84 (World Geodetic System 1984).

        Args:
            iteration (int): The current iteration of the simulation. Equals
            the time in seconds from its start.

        Returns:
            np.ndarray: Latitude of the satellite in degrees.
        """
        julian_date = ut.time_to_julian_date(self, iteration)
        return skyfield.wgs84.latlon_of(self._satellite_model.at(julian_date))[0].degrees

    def longitude(self, iteration) -> np.ndarray:
        """
        Longitude of the satellite in degrees calculated for the
        given simulation time and using WGS84 (World Geodetic System 1984).

        Args:
            iteration (int): The current iteration of the simulation. Equals
            the time in seconds from its start.

        Returns:
            np.ndarray: Longitude of the satellite in degrees.
        """
        julian_date = ut.time_to_julian_date(self, iteration)
        return skyfield.wgs84.latlon_of(self._satellite_model.at(julian_date))[1].degrees
    
    def altitude(self, iteration) -> np.ndarray:
        """
        Altitude of the satellite in km calculated for the
        given simulation time and using WGS84 (World Geodetic System 1984).

        Args:
            iteration (int): The current iteration of the simulation. Equals
            the time in seconds from its start.

        Returns:
            np.ndarray: Altitude of the satellite in km.
        """
        julian_date = ut.time_to_julian_date(self, iteration)
        # .subpoint().elevation gives altitude in meters; convert to km
        return self._satellite_model.at(julian_date).subpoint().elevation.m / 1000.0

    @property
    def angular_velocity(self) -> np.ndarray:
        return self._angular_velocity
    
    @property
    def reference_frame(self) -> str:
        return self._reference_frame
    
    @property
    def two_line_element(self) -> TwoLineElement:
        return self._two_line_element
