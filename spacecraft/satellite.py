from templates.satellite_template import Satellite
import numpy as np
from setup.two_line_element import TwoLineElement
from spacecraft.actuator import MagnetorquerImplementation
from templates.initial_settings_template import SimulationSetup
from templates.sensors_template import Magnetometer
from templates.sensors_template import Sunsensor
from templates.sensors_template import SensorFusion
import skyfield.api as skyfield
import core.utilities as ut
import core.transformations as tr


class SatelliteImplementation(Satellite):
    """
    Implementation of the Satellite class.
    """

    def __init__(
        self,
        setup: SimulationSetup,
        tle: TwoLineElement,
        magnetometer: Magnetometer = None,
        sunsensor: Sunsensor = None,
        sensor_fusion: SensorFusion = None,
        round_filter: bool = False,
        detumbling_threshold: float = 1,
        measurements_interval: int = 5,
    ):
        """
        Initialize the satellite object to easily obtain parameters that describe
        an orbital object and its measurements. It takes the simulation setup,
        two line element as input parameters that define the initial conditions.
        Sensor objects are also passed to the satellite for for simulation of
        readings. Different sensors can be used, but at least two are needed
        for sensor fusion. Typically the magnetometer is the main sensor.

        Args:
            setup (SimulationSetup): Simulation setup object that contains the
                initial conditions of the simulation, such as angular velocity,
                Euler angles, satellite parameters, and iterations information.
            tle (TwoLineElement): Two line element object that describes the
                initial conditions of the satellite's orbit.
            magnetometer (Magnetometer, optional): Magnetometer object for
                simulating magnetic field measurements. Readings are returned
                in the SBF (Satellite Body Frame) and ECI (Earth-Centered Inertial)
            sunsensor (Sunsensor, optional): Sunsensor object for simulating
                solar vector measurements. Readings are returned in the
                SBF (Satellite Body Frame) and ECI (Earth-Centered Inertial)
            sensor_fusion (SensorFusion, optional): SensorFusion object for
                performing sensor fusion algorithms such as TRIAD, QUEST, or EKF.
            round_filter (bool): If True, applies a rounding filter to the
                satellite's parameters to reduce numerical noise. Defaults to True.
            detumbling_threshold (float): The angular velocity magnitude above which
                detumbling is applied. Defaults to 3.0 degrees/s.
        """

        # TODO add measurement and actuation periods
        self.setup = setup
        self._angular_velocity = self.setup.angular_velocity
        self._euler_angles = self.setup.euler_angles
        self._iteration = self.setup.iterations_info["start"]
        self.round_filter = round_filter

        # quaternions are a very useful way to represent rotations, use the initial
        # Euler angles to calculate the initial quaternion
        self._quaternion = tr.euler_xyz_to_quaternion(self._euler_angles)
        self._quaternion = self._quaternion / np.linalg.norm(self._quaternion)

        self._two_line_element = tle

        # initialize the satellite model using skyfield library
        self._satellite_model = skyfield.EarthSatellite(
            self.two_line_element.line_1, self.two_line_element.line_2
        )

        if magnetometer is not None:
            self.magnetometer = magnetometer
        if sunsensor is not None:
            self.sunsensor = sunsensor
        self.activation_interval = measurements_interval

        self.sensor_fusion = sensor_fusion
        self.magnetorquer = MagnetorquerImplementation(self.setup, self)

        self.detumbling_threshold = detumbling_threshold
        self.start_detumbling = True
        self.start_pointing = False

        self._torque = np.zeros(3)
        self._angular_acceleration = np.zeros(3)

    def update_iteration(self, iteration: int) -> None:
        """
        Update the current iteration of the simulation.

        Args:
            iteration (int): The current iteration of the simulation.
                Equals the time in seconds from its start.
        """
        self._iteration = iteration

    @property
    def iteration(self) -> int:
        """
        Current iteration of the simulation. Equals the time in seconds
        from its start.
        """
        return self._iteration

    @property
    def mass(self) -> float:
        """
        Mass of the satellite in kg.
        """
        return self.setup.satellite_params[1]

    @property
    def inertia_matrix(self) -> np.ndarray:
        """
        Inertia matrix of the satellite in kg*m^2.
        """
        return self.setup.satellite_params[0]

    @property
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
        julian_date = ut.time_julian_date(self)
        position = self._satellite_model.at(julian_date).position.km
        if self.round_filter:
            position = ut.filter_significant_digits(position, 3)
        position = ut.filter_decimal_places(position, 3)
        return position

    @property
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
        julian_date = ut.time_julian_date(self)
        velocity = self._satellite_model.at(julian_date).velocity.km_per_s
        if self.round_filter:
            velocity = ut.filter_significant_digits(velocity, 3)
        return velocity

    @property
    def latitude(self) -> float:
        """
        Latitude of the satellite in degrees calculated for the
        given simulation time and using WGS84 (World Geodetic System 1984).

        Returns:
            float: Latitude of the satellite in degrees.
        """
        julian_date = ut.time_julian_date(self)
        latlon = skyfield.wgs84.latlon_of(self._satellite_model.at(julian_date))
        latitude = latlon[0].degrees
        if self.round_filter:
            latitude = ut.filter_decimal_places(latitude, 2)
        return latitude

    @property
    def longitude(self) -> float:
        """
        Longitude of the satellite in degrees calculated for the
        given simulation time and using WGS84 (World Geodetic System 1984).

        Returns:
            float: Longitude of the satellite in degrees.
        """
        julian_date = ut.time_julian_date(self)
        latlon = skyfield.wgs84.latlon_of(self._satellite_model.at(julian_date))
        longitude = latlon[1].degrees
        if self.round_filter:
            longitude = ut.filter_decimal_places(longitude, 2)
        return longitude

    @property
    def altitude(self) -> float:
        """
        Altitude of the satellite in km calculated for the
        given simulation time and using WGS84 (World Geodetic System 1984).

        Returns:
            float: Altitude of the satellite in km.
        """
        julian_date = ut.time_julian_date(self)
        geocentric = self._satellite_model.at(julian_date)
        # .subpoint().elevation gives altitude in meters; convert to km
        altitude = geocentric.subpoint().elevation.m / 1000.0
        if self.round_filter:
            altitude = ut.filter_decimal_places(altitude, 3)
        return altitude

    @property
    def angular_velocity(self) -> np.ndarray:
        """
        Angular velocity of the satellite in degrees/s. According to the
        aerospace convention, the angular velocity is given in the order
        wx, wy, wz (roll, pitch, yaw).

        Returns:
            np.ndarray: Angular velocity of the satellite in degrees/s.
        """
        new_velocity = self._angular_velocity
        if self.round_filter:
            new_velocity = ut.filter_decimal_places(new_velocity, 2)
        self._angular_velocity = new_velocity
        return new_velocity

    @property
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
        new_euler = tr.quaternion_to_euler_xyz(self.quaternion)
        if self.round_filter:
            new_euler = ut.filter_decimal_places(new_euler, 2)
        self._euler_angles = new_euler
        return new_euler

    @property
    def quaternion(self) -> np.ndarray:
        """
        Quaternion of the satellite. This represents the orientation of the
        satellite in space - the rotation from the reference ECI frame to the
        satellite's body frame. The quaternion is a 4-element array that
        contains the vector part and the scalar part (x, y, z, w).
        The vector part can be interpreted as a rotation axis and the scalar
        part represents the angle of rotation around that axis.

        Useful link:
        https://probablydance.com/2017/08/05/intuitive-quaternions/
        https://quaternions.online/

        Returns:
            np.ndarray: a 4-element array in the form of [x, y, z, w].
        """
        return self._quaternion / np.linalg.norm(self._quaternion)

    @property
    def two_line_element(self) -> TwoLineElement:
        """
        Two-line element set (TLE) of the satellite. Imported from file
        as object. Allows to access the parameters describing the satellite's
        orbital parameters such as inclination, right ascension etc.

        Returns:
            TwoLineElement: TLE object containing the orbital parameters.
        """
        return self._two_line_element

    @property
    def magnetic_field(self) -> np.ndarray:
        """
        Get the magnetic field vector at the satellite's position in the SBF and
        ECI frames. The first simulates the measurement, the second is used
        for debugging, sensor fusion algorithms etc. Both are in nT
        (nanoTesla). Adding bias to the SBF vector can be adjusted in the
        magnetometer object.

        Returns:
            np.ndarray: Magnetic field vector in the SBF and ECI frames in form of
            [[SBFx, SBFy, SBFz], [ECIx, ECIy, ECIz]].
        """
        # if self.iteration % self.activation_interval == 0 or self.iteration == 1:
        julian_date = ut.time_julian_date(self)
        mag_sbf, mag_eci = self.magnetometer.simulate_magnetometer(self, julian_date)
        if self.round_filter:
            mag_sbf = ut.filter_significant_digits(mag_sbf, 4)
            mag_eci = ut.filter_significant_digits(mag_eci, 4)
        self.magnetometer.last_sbf_measurement = mag_sbf
        self.magnetometer.last_eci_measurement = mag_eci
        # else:
        #     mag_sbf = self.magnetometer.last_sbf_measurement
        #     mag_eci = self.magnetometer.last_eci_measurement
        return mag_sbf, mag_eci

    @property
    def sun_vector(self) -> np.ndarray:
        """
        Get the Sun vector as observed from Earth. Due to the large distance
        the altitude is neglected. Only a rotation from ECI to SBF is applied.

        Returns:
            np.ndarray: Sun vector in the SBF and ECI frames in form of
            [[SBFx, SBFy, SBFz], [ECIx, ECIy, ECIz]].
        """
        # if self.iteration % self.activation_interval == 0 or self.iteration == 1:
        julian_date = ut.time_julian_date(self)
        sun_sbf, sun_eci = self.sunsensor.simulate_sunsensor(self, julian_date)
        if self.round_filter:
            sun_sbf = ut.filter_significant_digits(sun_sbf, 5)
            sun_eci = ut.filter_significant_digits(sun_eci, 5)
        self.sunsensor.last_sbf_measurement = sun_sbf
        self.sunsensor.last_eci_measurement = sun_eci
        # else:
        #     sun_sbf = self.sunsensor.last_sbf_measurement
        #     sun_eci = self.sunsensor.last_eci_measurement
        return sun_sbf, sun_eci

    @property
    def pointing_error_angle(self) -> np.ndarray:
        """
        Get the pointing error angle in degrees. This is the angle between the 
        vector that the satellite is aligned to and the Earth vector in ECI frame.
        Initialized as 0.0 and updated after pointing was launched.

        Returns:
            np.ndarray: Pointing error angle in degrees.
        """
        return self.magnetorquer.pointing_error_angle

    @property
    def torque(self) -> np.ndarray:
        """
        Get the torque applied by the magnetorquers in Nm. Initialized as
        [0.0, 0.0, 0.0] and updated after detumbling or pointing was launched.

        Returns:
            np.ndarray: Torque applied by the magnetorquers in Nm.
        """
        if self.start_detumbling or self.start_pointing:
            return self._torque
        else:
            return np.zeros(3)

    @property
    def angular_acceleration(self) -> np.ndarray:
        """
        Get the angular acceleration of the satellite in rad/s^2. Initialized as
        [0.0, 0.0, 0.0] and updated after detumbling or pointing was launched.

        Returns:
            np.ndarray: Angular acceleration of the satellite in rad/s^2.
        """
        if self.start_detumbling or self.start_pointing:
            return self._angular_acceleration
        else:
            return np.zeros(3)

    def apply_rotation(self) -> None:
        """
        This method updates the satellite's orientation based on the
        current angular velocity. A new quaternion is assigned. This is
        a rather theoretical rotation.
        """
        # Update the quaternion based on the angular velocity
        self._quaternion = tr.update_quaternion_by_angular_velocity(
            self._quaternion,
            ut.degrees_to_rad(self.angular_velocity)
        )

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
        self._quaternion = self.sensor_fusion.triad(v1_i, v2_i, v1_b, v2_b)

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
        self._quaternion = self.sensor_fusion.quest(v_b_list, v_i_list)

    def apply_ekf(
            self,
            v_b_list: list[np.ndarray],
            v_i_list: list[np.ndarray],
            quaternion_prev: np.ndarray,
            timestemp: float = 1.0
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
        angular_velocity_rad = ut.degrees_to_rad(self.angular_velocity)
        self._quaternion = self.sensor_fusion.ekf(
            v_b_list,
            v_i_list,
            angular_velocity_rad,
            timestemp,
            quaternion_prev
        )

    def apply_detumbling(
            self,
            method: str,
            adapt_magnetic: bool = False,
            adapt_velocity: bool = False,
            proportional: bool = False,
            modified: bool = False
    ) -> np.ndarray:
        """
        Perform detumbling using the specified method.
        """
        if method == "b_dot":
            if self.start_detumbling:
                angular_acceleration = self.magnetorquer.b_dot(
                    self.magnetic_field[0],
                    1,
                    adapt_magnetic,
                    adapt_velocity,
                    proportional,
                    modified
                )
                self._angular_velocity = self.angular_velocity - ut.rad_to_degrees(
                    angular_acceleration
                )

                self._torque = self.magnetorquer.torque
                self._angular_acceleration = ut.rad_to_degrees(angular_acceleration)
        else:
            raise ValueError(f"Unknown detumbling method: {method}")

    def apply_pointing(
        self,
        method: str,
        task: str,
        align_axis: np.ndarray | list,
    ):
        """
        Apply pointing control to the satellite. This method uses the B-cross
        algorithm to calculate the required torque to align the satellite's
        body frame with a target vector in the inertial frame.

        Args:
            method (str): The method to use for pointing control. Currently,
                only 'b_cross' is supported.
            task (str): The task which the B-cross should perform. As the algorithm is
                dedicated to pointing, currently 'earth-pointing' and 'sun-pointing'
                are supported.
            align_axis (np.ndarray | list): The axis which should be rotated towards
                the given target.

        """
        if method == "b_cross" and self.start_pointing:
            angular_acceleration = self.magnetorquer.b_cross(
                self.magnetic_field[1],
                self.magnetic_field[0],
                task,
                align_axis
            )
            self._angular_velocity = self.angular_velocity - ut.rad_to_degrees(angular_acceleration)

            self._torque = self.magnetorquer.torque
            self._angular_acceleration = ut.rad_to_degrees(angular_acceleration)

    def detumbling_pointing_switch(self) -> None:
        """
        Validate the angular velocity to determine if detumbling should stop or start
        again. Initially set to True.
        """
        if self.start_detumbling and np.linalg.norm(self.angular_velocity) <= self.detumbling_threshold:
            self.start_detumbling = False
            self.start_pointing = True
            print(
                f"Detumbling stopped, angular velocity is below threshold {self.detumbling_threshold}.")
        elif not self.start_detumbling and np.linalg.norm(self.angular_velocity) >= self.detumbling_threshold * 2:
            self.start_detumbling = True
            self.start_pointing = False
            print(
                f"Detumbling started, angular velocity increased above threshold {self.detumbling_threshold * 2}.")
