import ppigrf as igrf
import numpy as np
import datetime as dt
import skyfield.api as skyfield
from scipy.linalg import eig
import setup.transformations as tr
import setup.utilities as ut
from templates.satellite_template import Satellite
from scipy.spatial.transform import Rotation as R


class MagnetometerImplementation:
    def __init__(
        self,
        noise: bool = False,
        noise_min: float = 0.02,
        noise_max: float = 0.02,
    ):
        """
        Initialize the Magnetometer class.

        Args:
            noise (bool, optional): If True, adds noise to the magnetic field vector.
            Defaults to False.
            noise_min (float, optional): Minimum noise factor to apply. Defaults to 0.1.
            noise_max (float, optional): Maximum noise factor to apply. Defaults to 0.1.
        """
        self.noise = noise
        self.noise_min = noise_min
        self.noise_max = noise_max

    def get_magnetic_field(
        self, satellite: Satellite, date: dt.datetime
    ) -> np.ndarray:
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

        be, bn, bu = igrf.igrf(
            satellite.latitude, satellite.longitude, satellite.altitude, date
        )

        return np.array([be[0], bn[0], bu[0]])

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
        date_time = dt.datetime.fromtimestamp(julian_date.tt)
        mag_field_enu = self.get_magnetic_field(satellite, date_time)
        mag_field_ecef = tr.enu_to_ecef(
            mag_field_enu, satellite.latitude, satellite.longitude
        )
        mag_field_eci = tr.ecef_to_eci(mag_field_ecef, julian_date)
        mag_field_sbf = tr.eci_to_sbf(
            mag_field_eci,
            tr.euler_xyz_to_quaternion(satellite.euler_angles),
        )

        if self.noise:
            noise_vector = np.random.uniform(
                1 - self.noise_min, 1 + self.noise_max, 3
            )
            mag_field_eci *= noise_vector
            mag_field_sbf *= noise_vector

        return mag_field_sbf, mag_field_eci


class SunsensorImplementation:
    def __init__(
        self,
        noise: bool = False,
        noise_min: float = 0.02,
        noise_max: float = 0.02,
    ):
        """
        Initialize the Sunsensor class.

        Args:
            noise (bool, optional): If True, adds noise to the Sun vector.
            Defaults to False.
            noise_min (float, optional): Minimum noise factor to apply. Defaults to 0.1.
            noise_max (float, optional): Maximum noise factor to apply. Defaults to 0.1.
        """
        self.noise = noise
        self.noise_min = noise_min
        self.noise_max = noise_max

    def sun_vector_eci(self, julian_date: skyfield.Time) -> np.ndarray:
        """
        Compute Sun's position in ECI (ICRF) using Skyfield.

        Args:
            julian_date (skyfield.Time): Julian date for which to compute the Sun's position.

        Returns:
            numpy.ndarray: [x, y, z] in kilometers in ECI/ICRF frame
        """
        # Load planetary ephemeris
        eph = skyfield.load('de421.bsp')  # or 'de440s.bsp' if you want higher precision
        sun = eph['sun']
        earth = eph['earth']

        # Get Sun position relative to Earth in ICRF (equiv. to ECI for inertial applications)
        sun_position_eci = earth.at(julian_date).observe(sun).position.km  # Returns (x, y, z) in km
        return sun_position_eci

    def simulate_sunsensor(self, satellite: object, julian_date: skyfield.Time) -> np.ndarray:
        """
        Simulate the Sun sensor readings. This method computes the Sun vector
        at a given satellite and date, optionally adds noise and transforms it
        to the Satellite Body Frame (SBF).

        Args:
            satellite (object): The satellite object containing the TLE data and current status.
            julian_date (skyfield.Time): Julian date for which the Sun vector is to be computed.

        Returns:
            np.ndarray: Simulated Sun vector in the Satellite Body Frame (SBF).
        """
        sun_eci = self.sun_vector_eci(julian_date)
        sun_sbf = tr.eci_to_sbf(sun_eci, satellite.quaternion)

        if self.noise:
            noise_vector = np.random.uniform(
                1 - self.noise_min, 1 + self.noise_max, 3
            )
            sun_eci *= noise_vector
            sun_sbf *= noise_vector

        return sun_sbf, sun_eci


class SensorFusionImplementation():
    """
    Class for sensor fusion, combining data from multiple sensors.
    This class can be extended to include more sensors and their fusion logic.
    """

    def __init__(self, algorithm: list[str], init_quaternion: np.ndarray, weights: np.ndarray = np.ones(3)):
        """
        Initialize the SensorFusion class.

        Args:
            algorithm (str): List of algorithms to use initialize for the sensor fusion.
                Supported algorithms: 'triad', 'quest', 'ekf'.
        """

        self._data_dict = {alg: {} for alg in algorithm}
        for alg in algorithm:
            self._data_dict[alg][0] = init_quaternion

        if 'quest' in algorithm:
            self.weights = weights  # Equal weights for three vectors

        if 'ekf' in algorithm:
            self.gyro_bias = np.zeros(3)
            self.covariance = np.eye(6) * 0.001

            self.process_noise = np.diag([1e-8]*3 + [1e-10]*3)  # small process noise
            self.measurement_noise = np.eye(3) * 0.1

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
        # Build TRIAD frame in inertial
        R_i = self.build_triad(v1_i, v2_i)

        # Build TRIAD frame in body
        R_b = self.build_triad(v1_b, v2_b)

        # Rotation matrix from inertial to body
        R = R_b @ R_i.T
        quaternion = tr.rotation_matrix_to_quaternion(R)
        quaternion /= np.linalg.norm(quaternion)
        self.save_to_data_dict("triad", quaternion)
        return quaternion

    def build_triad(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Build a TRIAD frame from two vectors. The first vector is used as the
        primary axis, and the second vector is used to define the secondary axis.

        Args:
            v1 (np.ndarray): First vector.
            v2 (np.ndarray): Second vector.

        Returns:
            np.ndarray: 3x3 rotation matrix representing the TRIAD frame.
        """
        t1 = ut.normalize(v1)
        t2 = ut.normalize(np.cross(v1, v2))
        t3 = np.cross(t1, t2)
        R = np.vstack((t1, t2, t3)).T  # 3x3 matrix
        return R
    
    def quest(self, v_b_list: list[np.ndarray], v_i_list: list[np.ndarray]) -> np.ndarray:
        """
        QUEST algorithm for optimal attitude estimation.
        
        Args:
            v_b_list (list of np.ndarray): Body-frame unit vectors.
            v_i_list (list of np.ndarray): Inertial-frame unit vectors.
        
        Returns:
            np.ndarray: Quaternion [x, y, z, w] estimating attitude (ECI to body).
        """
        # Step 1: B matrix
        B = np.zeros((3, 3))
        for v_b, v_i, a in zip(v_b_list, v_i_list, self.weights):
            B += a * np.outer(v_b, v_i)

        S = B + B.T
        sigma = np.trace(B)
        Z = np.array([
            B[1, 2] - B[2, 1],
            B[2, 0] - B[0, 2],
            B[0, 1] - B[1, 0]
        ])

        # Step 2: K matrix
        K = np.zeros((4, 4))
        K[:3, :3] = S - sigma * np.eye(3)
        K[:3, 3] = Z
        K[3, :3] = Z
        K[3, 3] = sigma

        # Step 3: Eigenvector of max eigenvalue
        eigvals, eigvecs = eig(K)
        max_index = np.argmax(eigvals.real)
        quaternion = eigvecs[:, max_index].real
        quaternion /= np.linalg.norm(quaternion)
        self.save_to_data_dict("quest", quaternion)
        return ut.normalize(quaternion)
    
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
        # Placeholder for EKF implementation
        # This should include prediction and update steps based on the sensor data
        # For now, we will just return the input quaternion
        quaternion = self.prediction_step(angular_velocity, quaternion, timestep)
        for v_b, v_i in zip(v_b_list, v_i_list):
            quaternion = self.update_step(v_b, v_i, quaternion)

        self.save_to_data_dict("ekf", quaternion)
        quaternion /= np.linalg.norm(quaternion)
        return quaternion
    
    def prediction_step(self, angular_velocity, quaternion, timestep):
        """
        Predict step of the EKF using gyroscope data.

        Args:
            angular_velocity: Angular velocity measurement [rad/s]
            timestep: Time increment [s]
        """
        corrected_angular_velocity = angular_velocity - self.gyro_bias

        # Use rotation vector for integration
        delta_theta = corrected_angular_velocity * timestep
        delta_rotation = R.from_rotvec(delta_theta)
        updated_rotation = R.from_quat(quaternion) * delta_rotation
        quaternion = updated_rotation.as_quat()
        quaternion /= np.linalg.norm(quaternion)

        # Build F matrix: Jacobian of the error state

        omega_norm = np.linalg.norm(corrected_angular_velocity)
        F = np.eye(6)
        if omega_norm > 1e-5:
            axis = corrected_angular_velocity / omega_norm
            theta = omega_norm * timestep
            skew_axis = ut.skew_symmetric(axis)
            A = np.eye(3) - (theta / 2) * skew_axis + \
                ((1 - np.cos(theta)) / (theta ** 2)) * (skew_axis @ skew_axis)
            F[0:3, 3:6] = -A * timestep
        else:
            F[0:3, 3:6] = -np.eye(3) * timestep

        # Propagate covariance
        self.covariance = F @ self.covariance @ F.T + self.process_noise
        return quaternion
    
    def update_step(self, measurement_vector: np.ndarray, reference_vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        # Predict measurement by rotating reference vector into body frame
        predicted_vector_body = tr.rotate_vector_by_quaternion(reference_vector, quaternion)

        # Innovation
        innovation = measurement_vector - predicted_vector_body

        # Measurement Jacobian (approximate)
        H = np.zeros((3, 6))
        H[:, 0:3] = -ut.skew_symmetric(predicted_vector_body)

        # Kalman Gain
        S = H @ self.covariance @ H.T + self.measurement_noise
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Update state
        delta_state = K @ innovation
        delta_rotation = R.from_rotvec(delta_state[0:3])
        updated_rotation = R.from_quat(quaternion) * delta_rotation

        quaternion = updated_rotation.as_quat()

        self.gyro_bias += delta_state[3:6]
        self.covariance = (np.eye(6) - K @ H) @ self.covariance
        quaternion /= np.linalg.norm(quaternion)
        return quaternion
    
    def save_to_data_dict(self, algorithm: str, data: np.ndarray) -> None:
        """
        Save the computed quaternion to the data dictionary.

        Args:
            algorithm (str): The algorithm used for sensor fusion.
            data (np.ndarray): The computed quaternion.
        """
        if algorithm not in self._data_dict:
            raise ValueError(f"Algorithm '{algorithm}' not initialized")
        iteration = np.max(list(self._data_dict[algorithm].keys())) + 1
        self._data_dict[algorithm][iteration] = data