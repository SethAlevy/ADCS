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
        Position of the satellite obtained using the skyfield library.
        By default returns GCRS (Geocentric Celestial Reference System)
        which is an ECI (Earth-Centered Inertial) frame (fixed to the stars)
        almost similar to J2000 frame. Distance is given in km and calculated
        for the given simulation time.

        Returns:
            np.ndarray: X, Y and Z position of the satellite in km for current
                iteration.
        """
        pass

    @property
    @abstractmethod
    def linear_velocity(self) -> np.ndarray:
        """
        Linear velocity of the satellite obtained using the skyfield library.
        By default returns GCRS (Geocentric Celestial Reference System)
        which is an ECI (Earth-Centered Inertial) frame (fixed to the stars)
        almost similar to J2000 frame. Distance is given in km and calculated
        for the given simulation time.

        Returns:
            np.ndarray: X, Y and Z velocity of the satellite in km/s for current
                iteration.
        """
        pass

    @property
    @abstractmethod
    def latitude(self) -> float:
        """
        Latitude of the satellite in degrees calculated for the
        given simulation time and using WGS84 (World Geodetic System 1984).

        Returns:
            float: Latitude of the satellite in degrees.
        """
        pass

    @property
    @abstractmethod
    def longitude(self) -> float:
        """
        Longitude of the satellite in degrees calculated for the
        given simulation time and using WGS84 (World Geodetic System 1984).

        Returns:
            float: Longitude of the satellite in degrees.
        """
        pass

    @property
    @abstractmethod
    def altitude(self) -> float:
        """
        Altitude of the satellite in km calculated for the
        given simulation time and using WGS84 (World Geodetic System 1984).

        Returns:
            float: Altitude of the satellite in km.
        """
        pass

    @property
    @abstractmethod
    def angular_velocity(self) -> np.ndarray:
        """
        Angular velocity of the satellite in degrees/s. According to the
        aerospace convention, the angular velocity is given in the order
        wx, wy, wz (roll, pitch, yaw).

        Returns:
            np.ndarray: Angular velocity of the satellite in degrees/s.
        """
        pass

    @property
    @abstractmethod
    def euler_angles(self) -> np.ndarray:
        """
        Euler angles of the satellite in degrees. The angles are used to
        easily initialize the satellites position and later for debugging and
        visualization purposes. The angles are in the order of roll, pitch,
        and yaw (x, y, z) and are in the range of [-180, 180) degrees.
        They are obtained from the quaternion representation due to its
        advantages over Euler angles, such as avoiding gimbal lock and
        providing a more compact representation of rotations.

        Returns:
            np.ndarray: Updated Euler angles of the satellite in degrees
            (roll, pitch and yaw).
        """
        pass

    @property
    @abstractmethod
    def quaternion(self) -> np.ndarray:
        """
        Quaternion of the satellite. This represents the orientation of the
        satellite in space - the rotation from the reference ECI frame to the
        satellite's body frame. The quaternion is a 4-element array that
        contains the vector part and the scalar part (x, y, z, w).
        The vector part can be interpreted as a rotation axis and the scalar
        part represents the angle of rotation around that axis.

        Returns:
            np.ndarray: a 4-element array in the form of [x, y, z, w].
        """
        pass

    @property
    @abstractmethod
    def two_line_element(self) -> TwoLineElement:
        """
        Two-line element set (TLE) of the satellite. Imported from file
        as object. Allows to access the parameters descrbing the satellite's
        orbital parameters such as inclination, right ascension etc.
        """
        pass

    @property
    @abstractmethod
    def magnetic_field(self) -> np.ndarray:
        """
        Get the magnetic field vector at the satellite's position in the SBF and
        ECI frames. The first simulates the measurement, the second is used
        for debugging, sensor fusion algorithms etc. Both are in nT
        (nanoTesla). Adding bias to the SBF vector can be adjusted in the
        magnetometer object.

        Returns:
            np.ndarray: Magnetic field vector in the SBF and ECI frames.
        """
        pass

    @property
    @abstractmethod
    def sun_vector(self) -> np.ndarray:
        """
        Get the Sun vector as observed from Earth. Due to the large distance
        the altitude is neglected. Only a rotation from ECI to SBF is applied.

        Returns:
            np.ndarray: Sun vector in the SBF and ECI frames.
        """
        pass

    @abstractmethod
    def apply_rotation(self) -> None:
        """
        This method updates the satellite's orientation based on the
        current angular velocity. A new quaternion is assigned. This is
        a rather theoretical rotation.
        """
        pass

    @abstractmethod
    def apply_triad(
            self,
            v1_i: np.ndarray,
            v2_i: np.ndarray,
            v1_b: np.ndarray,
            v2_b: np.ndarray
    ) -> None:
        """
        Apply the TRIAD algorithm for attitude determination of two sensors.
        This method computes the quaternion from inertial to body frame
        using two vectors in both frames. The first vector is typically the more
        accurate. The satellites quaternion is updated based on this
        computations.

        Args:
            v1_i (np.ndarray): First vector in inertial frame.
            v2_i (np.ndarray): Second vector in inertial frame.
            v1_b (np.ndarray): First vector in body frame.
            v2_b (np.ndarray): Second vector in body frame.
        """
        pass

    @abstractmethod
    def apply_quest(
            self,
            v_b_list: list[np.ndarray],
            v_i_list: list[np.ndarray]
    ) -> None:
        """
        Apply the QUEST algorithm for attitude determination of at least two
        sensors. This method computes the quaternion from inertial to
        body frame in a slightly more precise way than TRIAD. Weights can
        be optionally added while initializing the sensor fusion class. The
        satellites quaternion is updated based on this computations.

        Args:
            v_b_list (list[np.ndarray]): list with vectors in body frame.
            v_i_list (list[np.ndarray]): list with vectors in inertial frame.
        """
        pass

    @abstractmethod
    def apply_ekf(
            self,
            v_b_list: list[np.ndarray],
            v_i_list: list[np.ndarray],
    ) -> None:
        """
        Apply the Extended Kalman Filter (EKF) for attitude estimation of at
        least two sensors. The method computes the quaternion from inertial to body
        using the angular velocity, estimated state, specified sensor accuracy and
        model parameters. Algorithm parameters can be passed while initializing
        the sensor fusion class. The satellites quaternion is updated based on this
        computations.

        Args:
            v_b_list (list[np.ndarray]): Body-frame unit vectors.
            v_i_list (list[np.ndarray]): Inertial-frame unit vectors.
        """
        pass
