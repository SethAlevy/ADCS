import ppigrf as igrf
import numpy as np
import datetime as dt
import skyfield.api as skyfield
from scipy.linalg import eig
import core.transformations as tr
import core.utilities as ut
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
        Initialize the Magnetometer class. It is responsible for calculating the
        Earth's magnetic field vector at a given satellite position and time using the
        International Geomagnetic Reference Field (IGRF) model. The measurement
        is simulated in the X, Y, Z axes by transforming it to the Satellite Body
        Frame (SBF) and optionally adding noise.

        Args:
            noise (bool, optional): If True, adds noise to the magnetic field vector.
                Defaults to False.
            noise_min (float, optional): Minimum noise to apply. Defaults to 0.02.
            noise_max (float, optional): Maximum noise to apply. Defaults to 0.02.
        """
        self.noise = noise
        self.noise_min = noise_min
        self.noise_max = noise_max

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
        Centered Inertial Frame (ECI). Returned in nT (nanoTesla).

        Args:
            satellite (object): The satellite object containing the TLE data and current
                status.
            julian_date (skyfield.Time): Julian date for which the magnetic field vector
                is to be computed.

        Returns:
            np.ndarray: Simulated magnetic field vectors in the Satellite Body Frame
                (SBF) and Earth-Centered Inertial (ECI) frame. Returned in nT
                (nanoTesla). The first three elements are in the SBF frame, and the
                next three are in the ECI frame.
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
            mag_field_sbf *= noise_vector

        return mag_field_sbf, mag_field_eci


class SunsensorImplementation:
    def __init__(
        self,
        noise: bool = False,
        noise_min: float = 0.04,
        noise_max: float = 0.04,
    ):
        """
        Initialize the Sunsensor class. It is responsible for calculating the
        direction of the Sun as it would be observed from the satellite. Due to the
        large distance the altitude is neglected and the vector is approximated
        as the observation from Earth. Optional noise can be applied. By default the
        sunsensor is assumed a bit less accurate than the magnetometer.

        Args:
            noise (bool, optional): If True, adds noise to the Sun vector.
                Defaults to False.
            noise_min (float, optional): Minimum noise to apply. Defaults to 0.04.
            noise_max (float, optional): Maximum noise to apply. Defaults to 0.04.
        """
        self.noise = noise
        self.noise_min = noise_min
        self.noise_max = noise_max

    def sun_vector_eci(self, julian_date: skyfield.Time) -> np.ndarray:
        """
        Compute Sun's position in ECI (ICRF) using Skyfield as seen from Earth.

        Args:
            julian_date (skyfield.Time): Julian date for which to compute the Sun's
                position.

        Returns:
            numpy.ndarray: [x, y, z] in kilometers in ECI (ICRF) frame
        """
        eph = skyfield.load('de421.bsp')  # or 'de440s.bsp' if you want higher precision
        sun = eph['sun']
        earth = eph['earth']

        # Get Sun position relative to Earth in ICRF (equiv. to ECI)
        sun_position_eci = earth.at(julian_date).observe(sun).position.km
        return sun_position_eci

    def simulate_sunsensor(
            self,
            satellite: object,
            julian_date: skyfield.Time
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
        sun_eci = self.sun_vector_eci(julian_date)
        sun_sbf = tr.eci_to_sbf(sun_eci, satellite.quaternion)

        if self.noise:
            noise_vector = np.random.uniform(
                1 - self.noise_min, 1 + self.noise_max, 3
            )
            sun_sbf *= noise_vector

        return sun_sbf, sun_eci


class SensorFusionImplementation():
    def __init__(
            self,
            algorithm: list[str],
            init_quaternion: np.ndarray,
            weights: np.ndarray = np.ones(2),
            gyro_bias: np.ndarray = np.zeros(3),
            gyro_process_noise: np.ndarray = np.array([1e-10, 1e-10, 1e-10]),
            attitude_noise: np.ndarray = np.array([1e-8, 1e-8, 1e-8]),
            covariance: np.ndarray = np.eye(6) * 0.001,
            measurement_noise: np.ndarray = np.eye(3) * 0.1
    ):
        """
        Initialize the SensorFusion class. Sensor fusion combines data that comes
        from multiple sensors to estimate a more accurate state of the system. In this
        case of attitude determination it applies to the quaternion. Often the order
        in which the parameters are given is relevant (depending on the expected
        accuracy) also estimations about noise bias can be passed while initializing.

        Args:
            algorithm (str): List of algorithms to initialize for the sensor fusion.
                Supported algorithms: 'triad', 'quest', 'ekf'.
            init_quaternion (np.ndarray): Initial quaternion [x, y, z, w] for the
                attitude.
            weights (np.ndarray, optional): Weights for the QUEST algorithm.
                Defaults to np.ones(3).
            gyro_bias (np.ndarray, optional): Initial gyroscope bias [x, y, z] for
                the EKF algorithm. Defaults to np.zeros(3) assuming ideal conditions.
            gyro_process_noise (np.ndarray, optional): Process noise for the gyroscope
                represents the uncertainty in how the gyroscope bias evolves over time.
                Defaults to np.array([1e-10, 1e-10, 1e-10]) assuming very small noise.
            attitude_noise (np.ndarray, optional): Noise in the attitude estimation
                includes small random perturbations in rotation for the EKF algorithm.
                Defaults to np.array([1e-8, 1e-8, 1e-8]) assuming uniform, small noise.
            covariance (np.ndarray, optional): Initial covariance matrix for the EKF
                algorithm. Describes the uncertainty and corelation between the
                state variables. Small values indicate high confidence, while larger
                values indicate a higher uncertainty. Defaults to np.eye(6) * 0.001.
            measurement_noise (np.ndarray, optional): Measurement noise for the EKF
                algorithm. Defaults to np.eye(3) * 0.1.
        """
        self._data_dict = {alg: {} for alg in algorithm}
        for alg in algorithm:
            self._data_dict[alg][0] = init_quaternion

        if 'quest' in algorithm:
            self.weights = weights

        if 'ekf' in algorithm:
            self.gyro_bias = gyro_bias
            self.gyro_process_noise = gyro_process_noise
            self.attitude_noise = attitude_noise
            self.covariance = covariance

            self.process_noise = np.diag(
                np.concatenate([attitude_noise, gyro_process_noise])
            )
            self.measurement_noise = measurement_noise

    def triad(
            self,
            v1_i: np.ndarray,
            v2_i: np.ndarray,
            v1_b: np.ndarray,
            v2_b: np.ndarray
    ) -> np.ndarray:
        """
        TRIAD  (Three-Axis Attitude Determination) algorithm for attitude determination
        of two sensors. It is a basic and simple algorithm used in aerospace. This
        method computes a rotation matrix from inertial to body frame using two
        vectors in both frames (SBF, ECI). The resulting rotation is a relative
        transformation between two orthogonal triads (coordinate systems created
        based on the given vectors) that represent different frames. The first vector
        is typically the more accurate measurement.

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
        # Build TRIAD in inertial frame
        R_i = self.build_triad(v1_i, v2_i)

        # Build TRIAD in body frame
        R_b = self.build_triad(v1_b, v2_b)

        # Rotation matrix from inertial to body frame
        R = R_b @ R_i.T
        quaternion = tr.rotation_matrix_to_quaternion(R)
        quaternion /= np.linalg.norm(quaternion)
        self.save_to_data_dict("triad", quaternion)
        return quaternion

    def build_triad(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Build an orthogonal triad frame from two vectors (set of three perpendicular
        vectors). The first vector is the primary axis, the second vector is calculated
        as the cross product of the first vector and the second vector, and the third
        vector is the cross product of the first and second vectors. This creates a
        right-handed coordinate system which can be represented as a 3x3 rotation
        matrix.

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

    def quest(
            self,
            v_b_list: list[np.ndarray],
            v_i_list: list[np.ndarray]
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
        # Attitude Profile Matrix (B): sum of weighted outer product of two vectors in
        # the same frame
        B = np.zeros((3, 3))
        for v_b, v_i, a in zip(v_b_list, v_i_list, self.weights):
            B += a * np.outer(v_b, v_i)

        # create auxiliary matrices
        S = B + B.T
        sigma = np.trace(B)
        Z = np.array([
            B[1, 2] - B[2, 1],
            B[2, 0] - B[0, 2],
            B[0, 1] - B[1, 0]
        ])

        # create the K matrix K = [S - sigma*I      Z  ]
        #                         [Z.T          sigma  ]
        K = np.zeros((4, 4))
        K[:3, :3] = S - sigma * np.eye(3)
        K[:3, 3] = Z
        K[3, :3] = Z
        K[3, 3] = sigma

        # Eigenvector of max eigenvalue which is the optimal quaternion
        eigvals, eigvecs = eig(K)
        max_index = np.argmax(eigvals.real)
        quaternion = eigvecs[:, max_index].real
        quaternion /= np.linalg.norm(quaternion)
        self.save_to_data_dict("quest", quaternion)
        return ut.normalize(quaternion)

    def ekf(
            self,
            v_b_list: list[np.ndarray],
            v_i_list: list[np.ndarray],
            angular_velocity: np.ndarray,
            timestep: float,
            quaternion: np.ndarray
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
        quaternion = self.prediction_step(angular_velocity, quaternion, timestep)

        # iterate through all vector measurements updating the quaternion
        for v_b, v_i in zip(v_b_list, v_i_list):
            quaternion = self.update_step(v_b, v_i, quaternion)

        self.save_to_data_dict("ekf", quaternion)
        quaternion /= np.linalg.norm(quaternion)
        return quaternion

    def prediction_step(
            self,
            angular_velocity: np.ndarray,
            quaternion: np.ndarray,
            timestep: float
    ) -> np.ndarray:
        """
        Predict step of the EKF using gyroscope data. The state (quaternion) is
        just rotated by the angular velocity in a given time step giving the
        predicted attitude. The gyroscope bias is also corrected in this step
        propagating the corrected covariance matrix.

        Args:
            angular_velocity (np.ndarray): Angular velocity measurement [rad/s]
            quaternion (np.ndarray): Current attitude quaternion [x, y, z, w]
            timestep (float): Time increment [s]

        Returns:
            np.ndarray: Updated quaternion estimate [x, y, z, w].
        """
        corrected_angular_velocity = angular_velocity - self.gyro_bias

        # Use rotation vector for integration
        delta_theta = corrected_angular_velocity * timestep
        delta_rotation = R.from_rotvec(delta_theta)
        updated_rotation = R.from_quat(quaternion) * delta_rotation
        quaternion = updated_rotation.as_quat()
        quaternion /= np.linalg.norm(quaternion)

        # Build state transition matrix F
        # F is the Jacobian of the state transition function
        # it is a matrix of partial derivatives representing sensitivity of the
        # output state is to input changes.
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

    def update_step(
            self,
            measurement_vector: np.ndarray,
            reference_vector: np.ndarray,
            quaternion: np.ndarray
    ) -> np.ndarray:
        """
        Update step of the EKF using vector measurements. The measurement is
        compared to the predicted measurement (rotated reference vector), then Kalman
        Gain is calculated based on the measurement Jacobian and the covariance matrix.
        Kalman Gain tells how much to trust the measurement compared to the
        prediction. Gyro bias and covariance are also adjusted. Lastly the quaternion
        is computed.

        Args:
            measurement_vector (np.ndarray): Measurement vector [x, y, z]
            reference_vector (np.ndarray): Reference vector [x, y, z]
            quaternion (np.ndarray): Current attitude quaternion [x, y, z, w]

        Returns:
            np.ndarray: Updated attitude quaternion [x, y, z, w]
        """
        # Predict measurement by rotating reference vector into body frame
        predicted_vector_body = tr.rotate_vector_by_quaternion(
            reference_vector, quaternion)

        # Innovation difference between the measurement and predicted vector
        innovation = measurement_vector - predicted_vector_body

        # Measurement Jacobian (approximate) it is a matrix of partial derivatives
        # representing measurement sensitivity to state changes.
        H = np.zeros((3, 6))
        H[:, 0:3] = -ut.skew_symmetric(predicted_vector_body)

        # Kalman Gain determines how much to trust new measurements versus
        # the current prediction
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
