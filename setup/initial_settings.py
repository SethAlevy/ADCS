import json
import datetime
from pathlib import Path
import numpy as np
from typing import Any
from templates.initial_settings_template import SimulationSetup


class SimulationSetupReader(SimulationSetup):
    def __init__(self, setup_file: str) -> None:
        self._setup = self._read_initial_parameters(setup_file)

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
        data['start'] = int(self._setup["Simulation"]["Iterations"]["Start"])
        data['stop'] = int(self._setup["Simulation"]["Iterations"]["Stop"])
        data['step'] = int(self._setup["Simulation"]["Iterations"]["Step"])
        data['log_interval'] = int(
            self._setup["Simulation"]["Iterations"]["LogInterval"]
        )
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
        data['n_coils'] = int(
            self._setup["Actuators"]["Magnetorquer"]["Coils"]
            )
        data['coil_area'] = float(
            self._setup["Actuators"]["Magnetorquer"]["RodArea"]
        )
        data['max_current'] = float(
            self._setup["Actuators"]["Magnetorquer"]["MaxCurrent"]
        )
        data["safety_factor"] = float(
            self._setup["Actuators"]["Magnetorquer"]["SafetyFactor"]
        )
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
        data['mass'] = float(self._setup["Satellite"]["Params"]["Mass"])
        data['inertia'] = np.array(self._setup["Satellite"]["Params"]["Inertia"])
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
        data['G'] = float(self._setup["Simulation"]["PlanetConst"]["G"])
        data['M'] = float(self._setup["Simulation"]["PlanetConst"]["M"])
        data['R'] = float(self._setup["Simulation"]["PlanetConst"]["R"])
        return data

    @property
    def date_time(self) -> datetime.datetime:
        """
        Date and time of the simulation start.

        returns:
            datetime: date_time - date and time of the simulation start.
        """
        return (
            datetime.datetime.now()
            if bool(self._setup["Simulation"]["Date"]["Now"])
            else datetime.datetime(
                self._setup["Simulation"]["Date"]["Year"],
                self._setup["Simulation"]["Date"]["Month"],
                self._setup["Simulation"]["Date"]["Day"],
                self._setup["Simulation"]["Date"]["Hour"],
                self._setup["Simulation"]["Date"]["Minute"],
                self._setup["Simulation"]["Date"]["Second"],
            )
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

        returns:
            bool: noise flag.
            float: maximum noise amplitude (nT).
        """
        data = dict()
        data['noise'] = bool(self._setup["Sensors"]["Magnetometer"]["Noise"])
        data['noise_max'] = float(self._setup["Sensors"]["Magnetometer"]["NoiseMax"])
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
        data['noise'] = bool(
            self._setup["Sensors"]["SunSensor"]["Noise"]
            )
        data['angular_noise'] = float(
            self._setup["Sensors"]["SunSensor"]["AngularNoise"]
            )
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
        data['bias'] = np.array(
            self._setup["Sensors"]["Gyroscope"]["Bias"]
            )
        data['process_noise'] = np.array(
            self._setup["Sensors"]["Gyroscope"]["ProcessNoise"]
            )
        return data

    @property
    def quest(self) -> np.ndarray:
        """
        QUEST parameters.

        returns:
            np.ndarray: weights for measurements.
        """
        data = dict()
        data["weights"] = np.array(
            self._setup["Sensors"]["QUEST"]["Weights"]
        )
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
        data['attitude_noise'] = np.array(
            self._setup["Sensors"]["EKF"]["AttitudeNoise"]
            )
        data['covariance'] = float(
            self._setup["Sensors"]["EKF"]["Covariance"]
            )
        data['measurement_noise'] = float(
            self._setup["Sensors"]["EKF"]["MeasurementNoise"]
            )
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
        data['proportional'] = bool(
            self._setup["Controls"]["Bdot"]["Proportional"]
            )
        data['modified'] = bool(
            self._setup["Controls"]["Bdot"]["Modified"]
            )
        data['adapt_velocity'] = bool(
            self._setup["Controls"]["Bdot"]["AdaptVelocity"]
            )
        data['adapt_magnetic'] = bool(
            self._setup["Controls"]["Bdot"]["AdaptMagnetic"]
            )
        data['bang_bang'] = bool(self._setup["Controls"]["Bdot"]["BangBang"])
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
        data['gain'] = float(
            self._setup["Controls"]["Bdot"]["Gain"]
            )
        data['proportional_gain'] = float(
            self._setup["Controls"]["Bdot"]["ProportionalGain"]
        )
        data['angular_velocity_ref'] = float(
            self._setup["Controls"]["Bdot"]["AngularVelocityRef"]
        )
        data['alpha'] = float(
            self._setup["Controls"]["Bdot"]["Alpha"]
        )
        data['magnetic_field_ref'] = int(
            self._setup["Controls"]["Bdot"]["MagneticFieldRef"]
        )
        data['beta'] = float(
            self._setup["Controls"]["Bdot"]["Beta"]
        )
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
        data['task'] = str(self._setup["Controls"]["Bcross"]["Task"])
        data['axis'] = np.array(self._setup["Controls"]["Bcross"]["PointingAxis"])
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
        data['align_gain'] = float(
            self._setup["Controls"]["Bcross"]["AlignGain"]
            )
        data['proportional_gain'] = float(
            self._setup["Controls"]["Bcross"]["ProportionalGain"]
            )
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
        data['detumbling_on'] = float(
            self._setup["Controls"]["ModeManagement"]["DetumblingOn"]
            )
        data['detumbling_off'] = float(
            self._setup["Controls"]["ModeManagement"]["DetumblingOff"]
            )
        data['pointing_on'] = float(
            self._setup["Controls"]["ModeManagement"]["PointingOn"]
            )
        data['pointing_off'] = float(
            self._setup["Controls"]["ModeManagement"]["PointingOff"]
            )
        data['pointing_dwell'] = int(
            self._setup["Controls"]["ModeManagement"]["PointingDwellTime"]
            )
        return data
