import numpy as np
from abc import ABC
from abc import abstractmethod
import skyfield.api as skyfield
from templates.satellite_template import Satellite
import datetime as dt


class Magnetometer(ABC):
    """
    Abstract base class for magnetometer sensors.

    Initialize the Magnetometer class. It is responsible for calculating the
    Earth's magnetic field vector at a given satellite position and time using the
    International Geomagnetic Reference Field (IGRF) model. The measurement
    is simulated in the X, Y, Z axes by transforming it to the Satellite Body
    Frame (SBF) and optionally adding noise.
    """

    @abstractmethod
    def get_magnetic_field(self, satellite: Satellite, date: dt.datetime) -> np.ndarray:
        """
        Get the magnetic field vector at a given satellite position and date.
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
        Centered Inertial Frame (ECI). Returned in nT (nanoTesla).

        Args:
            satellite (object): The satellite object containing the TLE data and current
                status.
            julian_date (skyfield.Time): Julian date for which the magnetic field vector
                is to be computed.

        Returns:
        np.ndarray: Simulated magnetic field vectors in the Satellite Body Frame
        (SBF) and Earth-Centered Inertial (ECI) frame. Returned in nT (nanoTesla).
        The first three elements are in the SBF frame, and the next three are in
        the ECI frame.
        """
        pass


class Sunsensor(ABC):
    """
    Abstract base class for sun sensors.

    Initialize the Sunsensor class. It is responsible for calculating the
    direction of the Sun as it would be observed from the satellite. Due to the
    large distance the altitude is neglected and the vector is approximated
    as the observation from Earth. Optional noise can be applied. By default the
    sunsensor is assumed a bit less accurate than the magnetometer.
    """

    @abstractmethod
    def sun_vector_eci(self, julian_date: skyfield.Time) -> np.ndarray:
        """
        Compute Sun's position in ECI (ICRF) using Skyfield as seen from Earth.

        Args:
            julian_date (skyfield.Time): Julian date for which to compute the Sun's
            position.

        Returns:
            numpy.ndarray: [x, y, z] in kilometers in ECI (ICRF) frame
        """
        pass

    @abstractmethod
    def simulate_sunsensor(
        self, satellite: object, julian_date: skyfield.Time
    ) -> np.ndarray:
        """
        Simulate the Sun sensor readings. This method computes the Sun vector
        at a given satellite position and date, optionally adds noise and transforms it
        to the Satellite Body Frame (SBF).

        Args:
            satellite (object): The satellite object containing the TLE data and
                current status.
            julian_date (skyfield.Time): Julian date for which the Sun vector is to
                be computed.

        Returns:
            np.ndarray: Simulated Sun vector in the Satellite Body Frame (SBF).
        """
        pass


class SensorFusion(ABC):
    """
    Abstract base class for sensor fusion algorithms.

    Initialize the SensorFusion class. Sensor fusion combines data that comes
    from multiple sensors to estimate a more accurate state of the system. In this
    case of attitude determination it applies to the quaternion. Often the order
    in which the parameters are given is relevant (depending on the expected
    accuracy) also estimations about noise bias can be passed while initializing.
    """

    @abstractmethod
    def triad(
        self, v1_i: np.ndarray, v2_i: np.ndarray, v1_b: np.ndarray, v2_b: np.ndarray
    ) -> np.ndarray:
        """
        TRIAD  (Three-Axis Attitude Determination) algorithm for attitude determination
        of two sensors. It is a basic and simple algorithm used in aerospace. This
        method computes a rotation matrix from inertial to body frame using two
        vectors in both frames (SBF, ECI) and constructing a triad from them. The
        first vector is typically the more accurate measurement.

        Useful links:
        https://www.aero.iitb.ac.in/satelliteWiki/index.php/Triad_Algorithm

        Args:
            v1_i (np.ndarray): First vector in inertial frame.
            v2_i (np.ndarray): Second vector in inertial frame.
            v1_b (np.ndarray): First vector in body frame.
            v2_b (np.ndarray): Second vector in body frame.

        Returns:
            np.ndarray: quaternion representing the rotation from inertial to body
                frame.
        """
        pass

    @abstractmethod
    def quest(
        self, v_b_list: list[np.ndarray], v_i_list: list[np.ndarray]
    ) -> np.ndarray:
        """
        QUEST (QUaternion ESTimator) algorithm for optimal attitude estimation of at
        least two sensors. The algorithm solves the Wahba problem by finding the
        a solution (rotation matrix) that minimizes the error between a set of
        weighted vectors in the body frame and their corresponding vectors in the
        inertial frame. It can take more measurements than TRIAD, and it is more
        robust to noise and outliers giving a more accurate estimate.

        Useful links:
        https://www.aero.iitb.ac.in/satelliteWiki/index.php/QuEST
        https://en.wikipedia.org/wiki/Wahba%27s_problem

        Args:
            v_b_list (list of np.ndarray): Body frame unit vectors.
            v_i_list (list of np.ndarray): Inertial frame unit vectors.

        Returns:
            np.ndarray: Quaternion [x, y, z, w] estimating attitude (ECI to body).
        """
        pass

    @abstractmethod
    def ekf(
        self,
        v_b_list: list[np.ndarray],
        v_i_list: list[np.ndarray],
        angular_velocity: np.ndarray,
        timestep: float,
        quaternion: np.ndarray,
    ) -> np.ndarray:
        """
        Extended Kalman Filter (EKF) for attitude estimation based on
        gyroscope measurements and at least two vector measurements. It is a recursive
        algorithm (updates over time) that combines the gyroscope data (angular
        velocity) with the vector measurements. The algorithm consists of two steps:
        prediction and update. The prediction is based on angular velocity, while the
        update incorporates the vector measurements. Compared to QUEST, EKF
        can handle noisy measurements and biases giving a comprehensive and
        accurate estimate.

        Useful links:
        https://medium.com/@sasha_przybylski/the-math-behind-extended-kalman-filtering-0df981a87453

        Args:
            v_b_list (list of np.ndarray): Body frame unit vectors.
            v_i_list (list of np.ndarray): Inertial frame unit vectors.
            dt (float): Time step for the EKF update.
            quaternion (np.ndarray): Current attitude quaternion [x, y, z, w].

        Returns:
            np.ndarray: Updated quaternion estimate [x, y, z, w].
        """
        pass
