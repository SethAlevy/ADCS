import numpy as np
from abc import ABC
from abc import abstractmethod
from setup.two_line_element import TwoLineElement


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

    @abstractmethod
    def angular_velocity(self) -> np.ndarray:
        """
        Angular velocity of the satellite in degrees/s.
        """
        pass

    @abstractmethod
    def euler_angles(self) -> np.ndarray:
        """
        Euler angles of the satellite in degrees.
        """
        pass

    @abstractmethod
    def quaternion(self) -> np.ndarray:
        """
        Quaternion of the satellite.
        """
        pass

    @property
    @abstractmethod
    def two_line_element(self) -> TwoLineElement:
        """
        Two-line element set (TLE) of the satellite. Imported from file
        as object.
        """
        pass
