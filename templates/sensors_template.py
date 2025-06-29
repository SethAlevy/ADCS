import numpy as np
from abc import ABC
from abc import abstractmethod
import skyfield.api as skyfield
from templates.satellite_template import Satellite
import datetime as dt


class Magnetometer(ABC):
    """
    Abstract base class for magnetometer sensors.
    """

    @abstractmethod
    def get_magnetic_field(self, satellite: Satellite, date: dt.datetime) -> np.ndarray:
        """
        Get the magnetic field vector at a given satellite and date.
        This method uses the ppigrf library to compute the magnetic field
        vector in the East-North-Up (ENU) reference frame in nT.

        Args:
            satellite (Satellite): The satellite object containing
        the TLE data and current status.
            date (dt.datetime): time object representing the date and time
            for which the magnetic field vector is to be computed.

        Returns:
            np.ndarray: vector containing the magnetic field components
            in the East-North-Up (ENU) reference frame.
        """
        pass

    @abstractmethod
    def simulate_magnetometer(
        self,
        satellite: object,
        julian_date: skyfield.Time,
    ) -> np.ndarray:
        """
        Simulate the magnetometer readings. This method computes the
        magnetic field vector at a given satellite and date, optionally adds
        noise and transforms it to the Satellite Body Frame (SBF) and Earth
        Centered Inertial Frame (ECI).

        Args:
            satellite (object): The satellite object containing the TLE data and current
                status.
            julian_date (skyfield.TIme): Julian date for which the magnetic field vector
                is to be computed.

        Returns:
        np.ndarray: Simulated magnetic field vectors in the Satellite Body Frame
        (SBF) and Earth-Centered Inertial (ECI) frame.
        The first three elements are in the SBF frame, and the next three are in
        the ECI frame.
        """
        pass


class Sunsensor(ABC):
    """
    Abstract base class for sun sensors.
    """

    @abstractmethod
    def sun_vector_eci(self, julian_date: skyfield.Time) -> np.ndarray:
        """
        Compute Sun's position in ECI (ICRF) using Skyfield.

        Args:
            julian_date (skyfield.Time): Julian date for which to compute the Sun's position.

        Returns:
            numpy.ndarray: [x, y, z] in kilometers in ECI/ICRF frame
        """
        pass

    @abstractmethod
    def simulate_sunsensor(self, satellite: Satellite, julian_date: skyfield.Time) -> np.ndarray:
        """
        Simulate the sun sensor readings. This method computes the Sun vector at a given
        satellite and date, optionally adds noise and transforms it to the Satellite Body Frame (SBF).

        Args:
            satellite (Satellite): The satellite object containing the TLE data and current status.
            julian_date (skyfield.Time): Julian date for which the Sun vector is to be computed.

        Returns:
            np.ndarray: Simulated Sun vector in the Satellite Body Frame (SBF).
        """
        pass
        
        
class SensorFusion(ABC):
    """
    Abstract base class for sensor fusion algorithms.
    This class can be extended to include more sensors and their fusion logic.
    """

    @abstractmethod
    def triad(self, v1_i: np.ndarray, v2_i: np.ndarray, v1_b: np.ndarray, v2_b: np.ndarray) -> np.ndarray:
        """
        TRIAD algorithm for attitude determination of two sensors.
        This method computes the quaternion from inertial to body frame
        using two vectors in both frames. The first vector is typically the more 
        accurate.

        Args:
            v1_i (np.ndarray): First vector in inertial frame.
            v2_i (np.ndarray): Second vector in inertial frame.
            v1_b (np.ndarray): First vector in body frame.
            v2_b (np.ndarray): Second vector in body frame.
        
        Returns:
            np.ndarray: quaternion representing the rotation from inertial to body frame.
        """
        pass

    @abstractmethod
    def quest(self, v_b_list: list[np.ndarray], v_i_list: list[np.ndarray]) -> np.ndarray:
        """
        QUEST algorithm for optimal attitude estimation.
        
        Args:
            v_b_list (list of np.ndarray): Body-frame unit vectors.
            v_i_list (list of np.ndarray): Inertial-frame unit vectors.
        
        Returns:
            np.ndarray: Quaternion [x, y, z, w] estimating attitude (ECI to body).
        """
        pass

    @abstractmethod
    def ekf(self, v_b_list: list[np.ndarray], v_i_list: list[np.ndarray], angular_velocity: np.ndarray, timestep: float, quaternion: np.ndarray) -> np.ndarray:
        """
        Extended Kalman Filter (EKF) for attitude estimation.
        
        Args:
            v_b_list (list of np.ndarray): Body-frame unit vectors.
            v_i_list (list of np.ndarray): Inertial-frame unit vectors.
            dt (float): Time step for the EKF update.
            quaternion (np.ndarray): Current quaternion estimate [x, y, z, w].
        
        Returns:
            np.ndarray: Updated quaternion estimate [x, y, z, w].
        """
        pass