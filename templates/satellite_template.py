import numpy as np
from abc import ABC
from abc import abstractmethod
from setup.two_line_element import TwoLineElement


class Satellite(ABC):
    """
    Abstract class for satellite.
    """

    @abstractmethod
    def update_iteration(self, iteration: int) -> None:
        """
        Update the current iteration of the simulation.

        Args:
            iteration (int): The current iteration of the simulation.
            Equals the time in seconds from its start.
        """
        pass

    @property
    @abstractmethod
    def iteration(self) -> int:
        """
        Current iteration of the simulation. Equals the time in seconds
        from its start.
        """
        pass

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

    @property
    @abstractmethod
    def position(self) -> np.ndarray:
        """
        Position of the satellite in m.
        """
        pass

    @property
    @abstractmethod
    def linear_velocity(self) -> np.ndarray:
        """
        Linear velocity of the satellite in m/s.
        """
        pass

    @property
    @abstractmethod
    def latitude(self) -> np.ndarray:
        """
        Latitude of the satellite in degrees.
        """
        pass

    @property
    @abstractmethod
    def longitude(self) -> np.ndarray:
        """
        Longitude of the satellite in degrees.
        """
        pass

    @property
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
        Angular velocity of the satellite in degrees/s. According to the
        aerospace convention, the angular velocity is given in the order
        wy, wz, wx (yaw, pitch, roll)
        """
        pass

    @property
    @abstractmethod
    def euler_angles(self) -> np.ndarray:
        """
        Euler angles of the satellite in degrees.
        """
        pass

    @property
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

    @property
    @abstractmethod
    def magnetic_field(self) -> np.ndarray:
        """
        Get the magnetic field vector at the satellite's position in the SBF and ECI frames.
        The second is rather for debugging purposes. Both are in nT (nanoTesla).

        Returns:
            np.ndarray: Magnetic field vector in the SBF and ECI frames.
        """
        pass

    @property
    @abstractmethod
    def sun_vector(self) -> np.ndarray:
        """
        Get the Sun vector in the ECI frame at the current simulation time.

        Returns:
            np.ndarray: Sun vector in the SB frame.
        """
        pass

    @abstractmethod
    def apply_rotation(self) -> None:
        """
        Apply the rotation to the satellite's position and orientation.
        This method updates the satellite's position and orientation based on
        the current angular velocity and time step.
        """
        pass
