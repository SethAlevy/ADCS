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
        data = {
            "Start": int(self._setup["Simulation"]["Iterations"]["Start"]),
            "Stop": int(self._setup["Simulation"]["Iterations"]["Stop"]),
            "Step": int(self._setup["Simulation"]["Iterations"]["Step"]),
            "LogInterval": int(
                self._setup["Simulation"]["Iterations"]["LogInterval"]
            ),
        }
        data = self._check_for_additional_keys_in_section(
            self._setup["Simulation"]["Iterations"], data)
        return data

    @property
    def magnetorquer_params(self) -> tuple[int, float, float]:
        """
        Magnetorquer parameters, works for every axis of rotation.

        returns (dict):
            Coils (int): number of coils.
            CoilArea (float): area of each coil in cm^2.
            MaxCurrent (float): maximum current in the torquer.
            SafetyFactor (float): current reduction factor.
            AlphaCap (float): angular acceleration cap (deg/s^2).
        """
        data = {
            "Coils": int(self._setup["Actuators"]["Magnetorquer"]["Coils"]),
            "CoilArea": float(
                self._setup["Actuators"]["Magnetorquer"]["CoilArea"]
            ),
            "MaxCurrent": float(
                self._setup["Actuators"]["Magnetorquer"]["MaxCurrent"]
            ),
            "SafetyFactor": float(
                self._setup["Actuators"]["Magnetorquer"]["SafetyFactor"]
            ),
            "AlphaCap": float(
                self._setup["Actuators"]["Magnetorquer"]["AlphaCap"]
            ),
        }
        data = self._check_for_additional_keys_in_section(
            self._setup["Actuators"]["Magnetorquer"], data)
        return data

    @property
    def satellite_params(self) -> tuple[float, np.ndarray]:
        """
        Satellite parameters.

        returns (dict):
            Mass (float): mass of the satellite in kg.
            Inertia (np.ndarray): inertia matrix in kg*m^2.
        """
        data = {
            "Mass": float(self._setup["Satellite"]["Params"]["Mass"]),
            "Inertia": np.array(self._setup["Satellite"]["Params"]["Inertia"]),
        }
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
        data = {
            "G": float(self._setup["Simulation"]["PlanetConst"]["G"]),
            "M": float(self._setup["Simulation"]["PlanetConst"]["M"]),
            "R": float(self._setup["Simulation"]["PlanetConst"]["R"]),
        }
        data = self._check_for_additional_keys_in_section(
            self._setup["Simulation"]["PlanetConst"], data)
        return data

    @property
    def date_time(self) -> datetime.datetime:
        """
        Date and time of the simulation start.

        Returns:
            datetime.datetime: Current time if "Now" is true, otherwise parsed from
                JSON fields.
        """
        date_config = self._setup["Simulation"]["Date"]
        if date_config.get("Now", False):
            return datetime.datetime.now()

        return datetime.datetime(
            year=int(date_config["Year"]),
            month=int(date_config["Month"]),
            day=int(date_config["Day"]),
            hour=int(date_config["Hour"]),
            minute=int(date_config["Minute"]),
            second=int(date_config["Second"])
        )

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

        returns (dict):
            Noise (bool): noise flag.
            AbsoluteNoise (float): maximum noise amplitude (nT).
        """
        data = {
            "Noise": bool(self._setup["Sensors"]["Magnetometer"]["Noise"]),
            "AbsoluteNoise": float(
                self._setup["Sensors"]["Magnetometer"]["AbsoluteNoise"]
            ),
        }
        data = self._check_for_additional_keys_in_section(
            self._setup["Sensors"]["Magnetometer"], data)
        return data

    @property
    def sunsensor(self) -> tuple[bool, float]:
        """
        Sun sensor settings.

        returns (dict):
            Noise (bool): noise flag.
            AngularNoise (float): angular noise (deg).
        """
        data = {
            "Noise": bool(self._setup["Sensors"]["SunSensor"]["Noise"]),
            "AngularNoise": float(
                self._setup["Sensors"]["SunSensor"]["AngularNoise"]
            ),
        }
        data = self._check_for_additional_keys_in_section(
            self._setup["Sensors"]["SunSensor"], data)
        return data

    @property
    def gyroscope(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Gyroscope settings.

        returns (dict):
            Bias (np.ndarray): bias (deg/s).
            ProcessNoise (np.ndarray): process noise (deg/s).
        """
        data = {
            "Bias": np.array(self._setup["Sensors"]["Gyroscope"]["Bias"]),
            "ProcessNoise": np.array(
                self._setup["Sensors"]["Gyroscope"]["ProcessNoise"]
            ),
        }
        data = self._check_for_additional_keys_in_section(
            self._setup["Sensors"]["Gyroscope"], data)
        return data

    @property
    def quest(self) -> np.ndarray:
        """
        QUEST parameters.

        returns (dict):
            Weights (np.ndarray): weights for measurements.
        """
        data = {"Weights": np.array(self._setup["Sensors"]["QUEST"]["Weights"])}
        data = self._check_for_additional_keys_in_section(
            self._setup["Sensors"]["QUEST"], data)
        return data

    @property
    def ekf(self) -> tuple[np.ndarray, float, float]:
        """
        EKF parameters.

        returns (dict):
            AttitudeNoise (np.ndarray): attitude noise in degrees.
            Covariance (float): covariance value.
        """
        data = {
            "AttitudeNoise": np.array(self._setup["Sensors"]["EKF"]["AttitudeNoise"]),
            "Covariance": float(self._setup["Sensors"]["EKF"]["Covariance"])
        }
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

        returns (dict of bool):
            Proportional : proportional damping term enabled.
            Modified : use modified B-dot (ω × B).
            AdaptVelocity : velocity-based adaptive gain.
            AdaptMagnetic : magnetic-field-based adaptive gain.
            BangBang : bang-bang control.
        """

        data = {
            "Proportional": bool(self._setup["Controls"]["Bdot"]["Proportional"]),
            "Modified": bool(self._setup["Controls"]["Bdot"]["Modified"]),
            "AdaptVelocity": bool(
                self._setup["Controls"]["Bdot"]["AdaptVelocity"]
            ),
            "AdaptMagnetic": bool(
                self._setup["Controls"]["Bdot"]["AdaptMagnetic"]
            ),
            "BangBang": bool(self._setup["Controls"]["Bdot"]["BangBang"]),
        }
        data = self._check_for_additional_keys_in_section(
            self._setup["Controls"]["Bdot"], data)
        return data

    @property
    def b_dot_parameters(self) -> tuple[int, float, float, float, int, float]:
        """
        B-dot control parameters.

        returns (dict):
             Gain (float): The gain factor for the b-dot control law. Applied in the
                standard B-dot control law and is the base for adaptive versions.
            ProportionalGain (float): Proportional gain for the B-dot control law.
                Determines how much the magnetic dipole moment is adjusted based on the
                angular velocity.
            AngularVelocityRef (float): Reference angular velocity for the
                adaptive B-dot control law in deg/s. Is the assumed value when the
                algorithm should switch from fast detumbling to more control.
            Alpha (float): Exponent for the angular velocity adaptation.
            MagneticFieldRef (int): Reference magnetic field for the adaptive B-dot
            control law in nT. Is the assumed value somewhere about the average
                magnetic field on the low Earth orbit.
            Beta (float): Exponent for the magnetic field adaptation.

        """
        data = {
            "Gain": float(self._setup["Controls"]["Bdot"]["Gain"]),
            "ProportionalGain": float(
                self._setup["Controls"]["Bdot"]["ProportionalGain"]
            ),
            "AngularVelocityRef": float(
                self._setup["Controls"]["Bdot"]["AngularVelocityRef"]
            ),
            "Alpha": float(self._setup["Controls"]["Bdot"]["Alpha"]),
            "MagneticFieldRef": int(
                self._setup["Controls"]["Bdot"]["MagneticFieldRef"]
            ),
            "Beta": float(self._setup["Controls"]["Bdot"]["Beta"]),
        }
        data = self._check_for_additional_keys_in_section(
            self._setup["Controls"]["Bdot"], data)
        return data

    @property
    def b_cross_mode(self) -> dict[str, np.ndarray]:
        """
        B-cross mode settings.

        returns (dict):
            Task (str): task mode ("earth_pointing" or "sun_pointing").
            PointingAxis (np.ndarray): pointing axis.
        """
        data = {
            "Task": str(self._setup["Controls"]["Bcross"]["Task"]),
            "PointingAxis": np.array(
                self._setup["Controls"]["Bcross"]["PointingAxis"]
            ),
        }
        data = self._check_for_additional_keys_in_section(
            self._setup["Controls"]["Bcross"], data)
        return data

    @property
    def b_cross_parameters(self) -> tuple[float, float]:
        """
        B-cross control parameters.

        returns (dict):
            AlignGain (float): alignment gain.
            ProportionalGain (float): damping gain.
        """
        data = {
            "AlignGain": float(self._setup["Controls"]["Bcross"]["AlignGain"]),
            "ProportionalGain": float(
                self._setup["Controls"]["Bcross"]["ProportionalGain"]
            ),
        }
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

        returns (dict):
            DetumblingOff (float): detumbling off threshold (deg/s).
            DetumblingOn (float): detumbling on threshold (deg/s).
            PointingOff (float): pointing off error angle (deg).
            PointingOn (float): pointing on error angle (deg).
            PointingDwellTime (int): pointing dwell time (s).
        """
        data = {
            "DetumblingOff": float(
                self._setup["Controls"]["ModeManagement"]["DetumblingOff"]
            ),
            "DetumblingOn": float(
                self._setup["Controls"]["ModeManagement"]["DetumblingOn"]
            ),
            "PointingOff": float(
                self._setup["Controls"]["ModeManagement"]["PointingOff"]
            ),
            "PointingOn": float(
                self._setup["Controls"]["ModeManagement"]["PointingOn"]
            ),
            "PointingDwellTime": int(
                self._setup["Controls"]["ModeManagement"]["PointingDwellTime"]
            ),
        }
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
                if extra_children := {
                    child_key: child_val
                    for child_key, child_val in top_val.items()
                    if child_key not in expected_children
                }:
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
