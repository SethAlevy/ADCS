import json
import datetime
from pathlib import Path
import numpy as np
from typing import Any
from templates.initial_settings_template import SimulationSetup


class SimulationSetupReader(SimulationSetup):
    _BASE_KEYS = {
        "Simulation": {"PlanetConst", "Iterations", "Date"},
        "Satellite": {"Params", "InitialState"},
        "Sensors": {"OnTime", "Magnetometer", "SunSensor", "Gyroscope", "QUEST", "EKF"},
        "Actuators": {"OnTime", "Magnetorquer"},
        "Controls": {"Bdot", "Bcross", "SensorFusion", "ModeManagement"},
    }

    def __init__(self, setup_file: str) -> None:
        self._setup = self._read_initial_parameters(setup_file)
        self.check_for_unknown_settings()

    def _read_initial_parameters(self, setup_file: str) -> Any:
        """
        Reads the initial parameters from setup/initial_parameters.json file.
        """
        if not Path(setup_file).exists():
            raise RuntimeError(f"Setup file not found at: {setup_file}")
        with open(Path(setup_file), "r") as f:
            return json.load(f)

    @property
    def euler_angles(self) -> np.ndarray:
        """
        Initial Euler angles phi, theta, psi. The standard aerospace convention
        X-Y-Z (known as roll-pitch-yaw) is used, where the first rotation is
        around the X-axis (roll), the second rotation is around the Y-axis
        (pitch), and the third rotation is around the Z-axis (yaw).

        returns:
            float: phi -180 to 180 degrees.
            float: theta -180 to 180 degrees.
            float: psi -180 to 180 degrees.
        """
        euler_angles = self._setup["Satellite"]["InitialState"]["EulerAngles"]
        phi = euler_angles[0]
        theta = euler_angles[1]
        psi = euler_angles[2]
        return np.array([phi, theta, psi])

    @property
    def angular_velocity(self) -> np.ndarray:
        """
        Initial angular velocity (q, r, p) in rad/s.

        returns:
            float: q - velocity around y-axis.
            float: r - velocity around z-axis.
            float: p - velocity around x-axis.
        """
        omega = self._setup["Satellite"]["InitialState"]["AngularVelocity"]
        p = omega[0]
        q = omega[1]
        r = omega[2]
        return np.array([p, q, r])

    @property
    def iterations_info(self) -> tuple[int, int, int, int]:
        """
        Simulation time parameters in seconds.

        returns:
            int: start time
            int: end time
            int: time step
            int: logging interval
        """
        data = dict()
        data["Start"] = int(self._setup["Simulation"]["Iterations"]["Start"])
        data["Stop"] = int(self._setup["Simulation"]["Iterations"]["Stop"])
        data["Step"] = int(self._setup["Simulation"]["Iterations"]["Step"])
        data["LogInterval"] = int(
            self._setup["Simulation"]["Iterations"]["LogInterval"]
        )
        data = self._check_for_additional_keys_in_section(
            self._setup["Simulation"]["Iterations"], data)
        return data

    @property
    def magnetorquer_params(self) -> tuple[int, float, float]:
        """
        Magnetorquer parameters, works for every axis of rotation.

        returns:
            int: n_coils - number of coils.
            float: coil_area - area of each coil in cm^2.
            float: max_current - maximum current in the torquer.
        """
        data = dict()
        data["Coils"] = int(self._setup["Actuators"]["Magnetorquer"]["Coils"])
        data["RodArea"] = float(self._setup["Actuators"]["Magnetorquer"]["RodArea"])
        data["MaxCurrent"] = float(self._setup["Actuators"]
                                   ["Magnetorquer"]["MaxCurrent"])
        data["SafetyFactor"] = float(
            self._setup["Actuators"]["Magnetorquer"]["SafetyFactor"])
        data["AlphaCap"] = float(self._setup["Actuators"]["Magnetorquer"]["AlphaCap"])
        data = self._check_for_additional_keys_in_section(
            self._setup["Actuators"]["Magnetorquer"], data)
        return data

    @property
    def satellite_params(self) -> tuple[float, np.ndarray]:
        """
        Satellite parameters.

        returns:
            float: mass of the satellite in kg.
            np.ndarray: inertia matrix in kg*m^2.
        """
        data = dict()
        data["Mass"] = float(self._setup["Satellite"]["Params"]["Mass"])
        data["Inertia"] = np.array(self._setup["Satellite"]["Params"]["Inertia"])
        data
        return data

    @property
    def planet_data(self) -> dict[float, float, float]:
        """
        Parameters and constants describing the planet (G, M, R).

        returns:
            float: G - gravitational constant in m^3/(kg*s^2).
            float: M - mass of the planet in kg.
            float: R - radius of the planet in m.
        """
        data = dict()
        data["G"] = float(self._setup["Simulation"]["PlanetConst"]["G"])
        data["M"] = float(self._setup["Simulation"]["PlanetConst"]["M"])
        data["R"] = float(self._setup["Simulation"]["PlanetConst"]["R"])
        data = self._check_for_additional_keys_in_section(
            self._setup["Simulation"]["PlanetConst"], data)
        return data

    @property
    def date_time(self) -> datetime.datetime:
        """
        Date and time of the simulation start.

        returns:
            datetime: date_time - date and time of the simulation start.
        """
        return {"OnTime": int(self._setup["Sensors"]["OnTime"])}

    @property
    def sensors_on_time(self) -> int:
        """
        Time interval for which the sensors are active. In real life conditions
        sensors and actuators should not work at the same time.

        returns:
            int: on time in seconds.
        """
        return int(self._setup["Sensors"]["OnTime"])

    @property
    def magnetometer(self) -> tuple[bool, float]:
        """
        Magnetometer settings.

        returns:
            bool: noise flag.
            float: maximum noise amplitude (nT).
        """
        data = dict()
        data["Noise"] = bool(self._setup["Sensors"]["Magnetometer"]["Noise"])
        data["AbsoluteNoise"] = float(
            self._setup["Sensors"]["Magnetometer"]["AbsoluteNoise"])
        data = self._check_for_additional_keys_in_section(
            self._setup["Sensors"]["Magnetometer"], data)
        return data

    @property
    def sunsensor(self) -> tuple[bool, float]:
        """
        Sun sensor settings.

        returns:
            bool: noise flag.
            float: angular noise (deg).
        """
        data = dict()
        data["Noise"] = bool(self._setup["Sensors"]["SunSensor"]["Noise"])
        data["AngularNoise"] = float(
            self._setup["Sensors"]["SunSensor"]["AngularNoise"])
        data = self._check_for_additional_keys_in_section(
            self._setup["Sensors"]["SunSensor"], data)
        return data

    @property
    def gyroscope(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Gyroscope settings.

        returns:
            np.ndarray: bias (deg/s).
            np.ndarray: process noise (deg/s).
        """
        data = dict()
        data["Bias"] = np.array(self._setup["Sensors"]["Gyroscope"]["Bias"])
        data["ProcessNoise"] = np.array(
            self._setup["Sensors"]["Gyroscope"]["ProcessNoise"])
        data = self._check_for_additional_keys_in_section(
            self._setup["Sensors"]["Gyroscope"], data)
        return data

    @property
    def quest(self) -> np.ndarray:
        """
        QUEST parameters.

        returns:
            np.ndarray: weights for measurements.
        """
        data = dict()
        data["Weights"] = np.array(self._setup["Sensors"]["QUEST"]["Weights"])
        data = self._check_for_additional_keys_in_section(
            self._setup["Sensors"]["QUEST"], data)
        return data

    @property
    def ekf(self) -> tuple[np.ndarray, float, float]:
        """
        EKF parameters.

        returns:
            np.ndarray: attitude noise in degrees.
            float: covariance value.
            float: measurement noise value.
        """
        data = dict()
        data["AttitudeNoise"] = np.array(self._setup["Sensors"]["EKF"]["AttitudeNoise"])
        data["Covariance"] = float(self._setup["Sensors"]["EKF"]["Covariance"])
        data = self._check_for_additional_keys_in_section(
            self._setup["Sensors"]["EKF"], data)
        return data

    @property
    def actuators_on_time(self) -> int:
        """
        Time interval for which the actuators are active. In real life conditions
        sensors and actuators should not work at the same time.

        returns:
            int: on time in seconds.
        """
        return self._setup["Actuators"]["OnTime"]

    @property
    def b_dot_mode(self) -> dict[bool]:
        """
        B-dot algorithm mods selection. If all are set to False the original
        variant will be used. Some may be combined (the proportional term may be
        used in the modified b-dot, the angular velocity and magnetic field adaptation
        are able to work together).

        returns:
            dict: set of bool indicating the selected mode:
                'proportional' : Whether to adapt the gain based on the angular
                velocity to include damping,
                'modified' : Whether to use the modified B-dot control law that
                is based directly on the angular velocity (gyroscopes) and
                magnetic field measurements, instead of magnetic field rate of change,
                'adapt_velocity' : Whether to adapt the gain based on the angular
                velocity,
                'adapt_magnetic' : Whether to adapt the gain based on the magnetic
                field strength,
                'bang_bang' : Whether to use bang-bang control that sets control
                output to maximum or minimum.
        """

        data = dict()
        data["Proportional"] = bool(self._setup["Controls"]["Bdot"]["Proportional"])
        data["Modified"] = bool(self._setup["Controls"]["Bdot"]["Modified"])
        data["AdaptVelocity"] = bool(self._setup["Controls"]["Bdot"]["AdaptVelocity"])
        data["AdaptMagnetic"] = bool(self._setup["Controls"]["Bdot"]["AdaptMagnetic"])
        data["BangBang"] = bool(self._setup["Controls"]["Bdot"]["BangBang"])
        data = self._check_for_additional_keys_in_section(
            self._setup["Controls"]["Bdot"], data)
        return data

    @property
    def b_dot_parameters(self) -> tuple[int, float, float, float, int, float]:
        """
        B-dot control parameters.

        returns:
            int: (k) The gain factor for the b-dot control law. Applied in the
                standard B-dot control law and is the base for adaptive versions.
            float: (k_p) Proportional gain for the B-dot control law. Determines
                how much the magnetic dipole moment is adjusted based on the
                angular velocity.
            float: Reference angular velocity for the
                adaptive B-dot control law in deg/s. Is the assumed value when the
                algorithm should switch from fast detumbling to more control.
            float: Exponent for the angular velocity adaptation.
            int: Reference magnetic field for the adaptive B-dot control law
                in nT. Is the assumed somewhere about the average magnetic field on the
                low Earth orbit.
            float: Exponent for the magnetic field adaptation.

        """
        data = dict()
        data["Gain"] = float(self._setup["Controls"]["Bdot"]["Gain"])
        data["ProportionalGain"] = float(
            self._setup["Controls"]["Bdot"]["ProportionalGain"])
        data["AngularVelocityRef"] = float(
            self._setup["Controls"]["Bdot"]["AngularVelocityRef"])
        data["Alpha"] = float(self._setup["Controls"]["Bdot"]["Alpha"])
        data["MagneticFieldRef"] = int(
            self._setup["Controls"]["Bdot"]["MagneticFieldRef"])
        data["Beta"] = float(self._setup["Controls"]["Bdot"]["Beta"])
        data = self._check_for_additional_keys_in_section(
            self._setup["Controls"]["Bdot"], data)
        return data

    @property
    def b_cross_mode(self) -> dict[str, np.ndarray]:
        """
        B-cross mode settings.

        returns:
            str: task mode ("earth_pointing" or "sun_pointing").
            np.ndarray: pointing axis.
        """
        data = dict()
        data["Task"] = str(self._setup["Controls"]["Bcross"]["Task"])
        data["PointingAxis"] = np.array(
            self._setup["Controls"]["Bcross"]["PointingAxis"])
        data = self._check_for_additional_keys_in_section(
            self._setup["Controls"]["Bcross"], data)
        return data

    @property
    def b_cross_parameters(self) -> tuple[float, float]:
        """
        B-cross control parameters.

        returns:
            float: align gain that determines how quickly the system aligns with
                the target.
            float: proportional gain that determines how much the system will damp
            angular velocity.
        """
        data = dict()
        data["AlignGain"] = float(self._setup["Controls"]["Bcross"]["AlignGain"])
        data["ProportionalGain"] = float(
            self._setup["Controls"]["Bcross"]["ProportionalGain"])
        data = self._check_for_additional_keys_in_section(
            self._setup["Controls"]["Bcross"], data)
        return data

    @property
    def sensor_fusion_algorithm(self) -> str:
        """
        Selected sensor fusion algorithm.

        returns:
            str: algorithm name.
        """
        return str(self._setup["Controls"]["SensorFusion"]["Algorithm"])

    @property
    def mode_management(self) -> tuple[float, float, float, float, int]:
        """
        Mode management thresholds.

        returns:
            float: detumbling off threshold (deg/s).
            float: detumbling on threshold (deg/s).
            float: pointing off error angle (deg).
            float: pointing on error angle (deg).
            int: pointing dwell time (s).
        """
        data = dict()
        data["DetumblingOff"] = float(
            self._setup["Controls"]["ModeManagement"]["DetumblingOff"])
        data["DetumblingOn"] = float(
            self._setup["Controls"]["ModeManagement"]["DetumblingOn"])
        data["PointingOff"] = float(self._setup["Controls"]
                                    ["ModeManagement"]["PointingOff"])
        data["PointingOn"] = float(self._setup["Controls"]
                                   ["ModeManagement"]["PointingOn"])
        data["PointingDwellTime"] = int(
            self._setup["Controls"]["ModeManagement"]["PointingDwellTime"])
        data = self._check_for_additional_keys_in_section(
            self._setup["Controls"]["ModeManagement"], data)
        return data

    def check_for_unknown_settings(self) -> dict | None:
        """
        Returns a dict of extra keys beyond the first two levels.
        Only checks: top-level sections and their immediate child keys.
        Deeper nesting is not validated here.
        """
        self.other_parameters = {}
        for top_key, top_val in self._setup.items():
            if top_key not in self._BASE_KEYS:
                self.other_parameters[top_key] = top_val  # whole unexpected section
                continue
            # If expected children are defined, compare immediate keys
            if isinstance(top_val, dict):
                expected_children = self._BASE_KEYS[top_key]
                extra_children = {
                    child_key: child_val
                    for child_key, child_val in top_val.items()
                    if child_key not in expected_children
                }
                if extra_children:
                    self.other_parameters[top_key] = extra_children
        return self.other_parameters if self.other_parameters else None

    def _check_for_additional_keys_in_section(self, setup_dict, data_dict) -> dict:
        """
        Checks for additional keys in a section of the setup dictionary that are not
        defined in the data dictionary.

        Args:
            setup_dict (dict): The raw section dictionary from the setup file.
            data_dict (dict): The current data dictionary.

        Returns:
            dict: Updated data dictionary with any additional keys.
        """
        for key in setup_dict:
            if key not in data_dict:
                data_dict[key] = setup_dict[key]
        return data_dict
