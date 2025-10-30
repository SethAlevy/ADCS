import numpy as np
from setup.two_line_element import TwoLineElement
from spacecraft.actuator import MagnetorquerImplementation
from templates.satellite_template import Satellite
from templates.initial_settings_template import SimulationSetup
from templates.sensors_template import Magnetometer
from templates.sensors_template import Sunsensor
from templates.sensors_template import SensorFusion
import skyfield.api as skyfield
import core.utilities as ut
import core.transformations as tr
from core.state import State


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
            detumbling_threshold (float): The angular velocity magnitude above which
                detumbling is applied. Defaults to 3.0 degrees/s.
        """
        # Use the initial parameters from setup to set orientation and rotations
        self.setup = setup
        self._angular_velocity = self.setup.angular_velocity
        self._euler_angles = self.setup.euler_angles
        self._iteration = self.setup.iterations_info["start"]

        # Initialize torque and acceleration as zeros
        self._torque = np.zeros(3)
        self._angular_acceleration = np.zeros(3)

        # quaternions are a very useful way to represent rotations, use the initial
        # Euler angles to calculate the initial quaternion
        self._quaternion = tr.euler_xyz_to_quaternion(self._euler_angles)
        self._quaternion = self._quaternion / np.linalg.norm(self._quaternion)

        # initialize the satellite model using skyfield library with SGP4 and tle
        self._two_line_element = tle
        self._satellite_model = skyfield.EarthSatellite(
            self._two_line_element.line_1, self._two_line_element.line_2
        )

        # Sensor initialization, at this time only magnetometers and sunsensors are
        # available, but if others would be implemented one may choose to select
        # various configurations for sensor fusion
        if magnetometer is not None:
            self.magnetometer = magnetometer
        if sunsensor is not None:
            self.sunsensor = sunsensor
        self.sensor_fusion = sensor_fusion
        self.fusion_methods = setup.sensor_fusion_algorithm.capitalize()

        self.magnetorquer = MagnetorquerImplementation(self.setup, self)

        # in real conditions sensors and actuators should not work at the same time
        # this values tell how long each of them is active
        self.sensors_time = True
        self.actuators_time = False
        self.actuator_on_time = setup.actuators_on_time
        self.sensor_on_time = setup.sensors_on_time

        # Set the detumbling/pointing management parameters and initialize the
        # starting values and mode

        self.detumbling_threshold_on = setup.mode_management['detumbling_on']
        self.detumbling_threshold_off = setup.mode_management['detumbling_off']
        self.pointing_error_ang_on = setup.mode_management['pointing_on']
        self.pointing_error_ang_off = setup.mode_management['pointing_off']
        self.pointing_dwell_time = setup.mode_management['pointing_dwell']

        self.start_detumbling = True
        self.start_pointing = False
        self._pointing_error_angle: float = 0.0
        self._pointing_ok_counter = 0

        # Initialize state vector for data storing
        self._state_vector = State()

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
        return self.setup.satellite_params["mass"]

    @property
    def inertia_matrix(self) -> np.ndarray:
        """
        Inertia matrix of the satellite in kg*m^2.
        """
        return self.setup.satellite_params["inertia"]

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
        julian_date = ut.time_julian_date(self)
        mag_sbf, mag_eci = self.magnetometer.simulate_magnetometer(self, julian_date)
        self.magnetometer.last_sbf_measurement = mag_sbf
        self.magnetometer.last_eci_measurement = mag_eci
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
        julian_date = ut.time_julian_date(self)
        sun_sbf, sun_eci = self.sunsensor.simulate_sunsensor(self, julian_date)
        self.sunsensor.last_sbf_measurement = sun_sbf
        self.sunsensor.last_eci_measurement = sun_eci
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
        return self._pointing_error_angle

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

    @property
    def state_vector(self) -> State:
        """
        Get the state vector of the satellite. This object contains all the
        parameters that are stored during the simulation for later analysis.
        """
        return self._state_vector

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
            v_b_list: list[np.ndarray],
            v_i_list: list[np.ndarray]
    ) -> None:
        """
        Apply the TRIAD algorithm for attitude determination of two sensors.
        This method computes the quaternion from inertial to body frame
        using two vectors in both frames. The first vector is typically the more
        accurate. The satellites quaternion is updated based on this
        computations.

        Args:
            v_b_list (list[np.ndarray]): list with vectors in body frame.
            v_i_list (list[np.ndarray]): list with vectors in inertial frame.
        """
        self._quaternion = self.sensor_fusion.triad(v_i_list, v_b_list)

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
            v_b_list (list[np.ndarray]): list with vectors in body frame.
            v_i_list (list[np.ndarray]): list with vectors in inertial frame.
            quaternion_prev (np.ndarray): Previous quaternion estimate.
            timestemp (float): Time step between the previous and current estimate
                in seconds. Default is 1.0 second.
        """
        angular_velocity_rad = ut.degrees_to_rad(self.angular_velocity)
        self._quaternion = self.sensor_fusion.ekf(
            v_b_list,
            v_i_list,
            angular_velocity_rad,
            timestemp,
            quaternion_prev
        )

    def fuse_sensors(
            self,
            v_b_list: list[np.ndarray],
            v_i_list: list[np.ndarray],
            quaternion_prev: np.ndarray = None,
    ) -> None:
        """
        Perform the sensor fusion by applying the algorithm selected in
        initial settings json file.

        Args:
            v_b_list (list[np.ndarray]): list with vectors in body frame.
            v_i_list (list[np.ndarray]): list with vectors in inertial frame.
            quaternion_prev (np.ndarray): Previous quaternion estimate.
        """
        self.apply_rotation()

        if self.fusion_methods == "TRIAD":
            self.apply_triad(v_b_list, v_i_list)
        elif self.fusion_methods == "QUEST":
            self.apply_quest(v_b_list, v_i_list)
        elif self.fusion_methods == "EKF":
            if quaternion_prev is None:
                raise ValueError("Previous quaternion must be provided for EKF.")
            self.apply_ekf(
                v_b_list,
                v_i_list,
                quaternion_prev,
                self.setup.iterations_info["step"]
            )

    def apply_detumbling(
            self,
    ) -> None:
        """
        Detumbling is the process of reducing the angular velocity of the satellite to
        a safe level after deployment. Here the popular B-dot algorithm is implemented
        using the magnetorquers. Different adaptations of the algorithm can be selected
        inside the initial setting json file to adjust the behavior. If no adaptation
        is selected the basic B-dot is used.
        """
        if self.start_detumbling:
            angular_acceleration = self.magnetorquer.b_dot(
                self.magnetic_field[0],
                self.setup.iterations_info['step'],
                self.setup.b_dot_mode['adapt_magnetic'],
                self.setup.b_dot_mode['adapt_velocity'],
                self.setup.b_dot_mode['proportional'],
                self.setup.b_dot_mode['modified']
            )
            self._angular_velocity = self.angular_velocity - ut.rad_to_degrees(
                angular_acceleration
            )

            self._torque = self.magnetorquer.torque
            self._angular_acceleration = ut.rad_to_degrees(angular_acceleration)

    def apply_pointing(
        self,
    ) -> None:
        """
        Apply pointing control to the satellite. This method uses the B-cross
        algorithm to calculate the required torque to align the satellite's
        body frame with a target vector in the inertial frame. The target vector
        can be either the Earth direction or the Sun direction, depending on
        the selected task in the initial settings.
        """
        task = self.setup.b_cross_mode['task']
        align_axis = self.setup.b_cross_mode['axis']

        if task == "earth_pointing":
            target_dir_body = tr.earth_direction_body(self.position, self.quaternion)
        elif task == "sun_pointing":
            target_dir_body = tr.sun_direction_body(self.sun_vector[1], self.quaternion)
        else:
            raise ValueError(f"Unknown pointing task: {task}. Only 'earth_pointing' "
                             "and 'sun_pointing' are supported.")

        self._pointing_error_angle = ut.calculate_pointing_error(
            target_dir_body,
            align_axis
        )

        if self.start_pointing:
            mag_sbf, _ = self.magnetic_field
            angular_acceleration = self.magnetorquer.b_cross(
                mag_sbf,
                align_axis,
                target_dir_body
            )
            self._angular_velocity = self.angular_velocity + \
                ut.rad_to_degrees(angular_acceleration)
            self._torque = self.magnetorquer.torque
            self._angular_acceleration = ut.rad_to_degrees(angular_acceleration)

    def _update_pointing_error_noact(self) -> None:
        """Update _pointing_error_angle even when pointing is off."""
        if self.setup.b_cross_mode['task'] == "earth_pointing":
            target_dir_body = tr.earth_direction_body(self.position, self.quaternion)
        elif self.setup.b_cross_mode['task'] == "sun_pointing":
            target_dir_body = tr.sun_direction_body(self.sun_vector[1], self.quaternion)
        else:
            return
        self._pointing_error_angle = ut.calculate_pointing_error(
            target_dir_body, self.setup.b_cross_mode['axis']
        )

    def manage_modes(self) -> None:
        """
        Mode manager for detumbling / pointing / idle.
        - Starts pointing after detumbling when rate below detumbling_threshold.
        - Finishes pointing after dwell in low-error, low-rate state (to Idle).
        - Re-enters detumbling only for very high rates.
        - Re-enables pointing if error drifts after completion.
        """
        ang_vel_norm = np.linalg.norm(self.angular_velocity)
        if not self.start_pointing:
            self._update_pointing_error_noact()
        pointing_err = self.pointing_error_angle

        # 1) Detumbling -> Pointing (unchanged)
        if self.start_detumbling and ang_vel_norm <= self.detumbling_threshold_off:
            self.start_detumbling = False
            self.start_pointing = True
            self._pointing_ok_counter = 0
            print(
                f"Detumbling stopped (|ω|={ang_vel_norm:.2f} deg/s). Pointing started."
            )

        # 2) Only revert to detumbling at very high rates (avoid at low rates)
        elif not self.start_detumbling and ang_vel_norm >= self.detumbling_threshold_on:
            self.start_detumbling = True
            self.start_pointing = False
            self._pointing_ok_counter = 0
            print(f"Detumbling restarted (|ω|={ang_vel_norm:.2f} deg/s).")

        # 3) Pointing completion: require low angle AND low rate for a while → Idle
        if self.start_pointing:
            near_angle = pointing_err <= self.pointing_error_ang_off
            if near_angle:
                self._pointing_ok_counter += 1
            else:
                self._pointing_ok_counter = 0
            if self._pointing_ok_counter >= self.pointing_dwell_time:
                self.start_pointing = False
                print(
                    f"Pointing completed → Idle (angle≈{pointing_err:.2f}°).")

        # 4) Re-acquire pointing if it drifted after completion (Idle)
        if (not self.start_pointing and not self.start_detumbling and
                pointing_err >= self.pointing_error_ang_on):
            self.start_pointing = True
            self._pointing_ok_counter = 0
            print(f"Pointing re-enabled (drift angle={pointing_err:.1f}°).")
