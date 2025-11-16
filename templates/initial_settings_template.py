import numpy as np
from abc import ABC, abstractmethod
import datetime


class SimulationSetup(ABC):
    """
    Abstract class for simulation setup. Defines the required initial
    parameters given in setup/initial_parameters.json file.
    """

    @property
    @abstractmethod
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
        pass

    @property
    @abstractmethod
    def angular_velocity(self) -> np.ndarray:
        """
        Initial angular velocity (q, r, p) in rad/s.

        returns:
            float: q - velocity around y-axis.
            float: r - velocity around z-axis.
            float: p - velocity around x-axis.
        """
        pass

    @property
    @abstractmethod
    def iterations_info(self) -> dict:
        """
        Simulation time parameters in seconds.

        returns:
            int: start time
            int: end time
            int: time step
            int: logging interval
        """
        pass

    @property
    @abstractmethod
    def magnetorquer_params(self) -> dict:
        """
        Magnetorquer parameters, works for every axis of rotation.

        returns (dict):
            Coils (int): number of coils.
            CoilArea (float): area of each coil in cm^2.
            MaxCurrent (float): maximum current in the torquer.
            SafetyFactor (float): current reduction factor.
            AlphaCap (float): angular acceleration cap (deg/s^2).
        """
        pass

    @property
    @abstractmethod
    def satellite_params(self) -> dict:
        """
        Satellite parameters.

        returns (dict):
            Mass (float): mass of the satellite in kg.
            Inertia (np.ndarray): inertia matrix in kg*m^2.
        """
        pass

    @property
    @abstractmethod
    def planet_data(self) -> dict:
        """
        Parameters and constants describing the planet (G, M, R).

        returns:
            float: G - gravitational constant in m^3/(kg*s^2).
            float: M - mass of the planet in kg.
            float: R - radius of the planet in m.
        """
        pass

    @property
    @abstractmethod
    def date_time(self) -> datetime.datetime:
        """
        Date and time of the simulation start.

        Returns:
            datetime.datetime: Current time if "Now" is true, otherwise parsed from
            JSON fields.
        """
        pass

    @property
    @abstractmethod
    def sensors_on_time(self) -> int:
        """
        Time interval for which the sensors are active. In real life conditions
        sensors and actuators should not work at the same time.

        returns:
            int: on time in seconds.
        """
        pass

    @property
    @abstractmethod
    def magnetometer(self) -> dict:
        """
        Magnetometer settings.

        returns (dict):
            Noise (bool): noise flag.
            AbsoluteNoise (float): maximum noise amplitude (nT).
        """
        pass

    @property
    @abstractmethod
    def sunsensor(self) -> dict:
        """
        Sun sensor settings.

        returns (dict):
            Noise (bool): noise flag.
            AngularNoise (float): angular noise (deg).
        """
        pass

    @property
    @abstractmethod
    def gyroscope(self) -> dict:
        """
        Gyroscope settings.

        returns (dict):
            Bias (np.ndarray): bias (deg/s).
            ProcessNoise (np.ndarray): process noise (deg/s).
        """
        pass

    @property
    @abstractmethod
    def quest(self) -> dict:
        """
        QUEST parameters.

        returns (dict):
            Weights (np.ndarray): weights for measurements.
        """
        pass

    @property
    @abstractmethod
    def ekf(self) -> dict:
        """
        EKF parameters.

        returns (dict):
            AttitudeNoise (np.ndarray): attitude noise in degrees.
            Covariance (float): covariance value.
        """
        pass

    @property
    @abstractmethod
    def actuators_on_time(self) -> int:
        """
        Time interval for which the actuators are active. In real life conditions
        sensors and actuators should not work at the same time.

        returns:
            int: on time in seconds.
        """
        pass

    @property
    @abstractmethod
    def b_dot_mode(self) -> dict:
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
        pass

    @property
    @abstractmethod
    def b_dot_parameters(self) -> dict:
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
        pass

    @property
    @abstractmethod
    def b_cross_mode(self) -> dict:
        """
        B-cross mode settings.

        returns (dict):
            Task (str): task mode ("earth_pointing" or "sun_pointing").
            PointingAxis (np.ndarray): pointing axis.
        """
        pass

    @property
    @abstractmethod
    def b_cross_parameters(self) -> dict:
        """
        B-cross control parameters.

        returns (dict):
            AlignGain (float): alignment gain.
            ProportionalGain (float): damping gain.
        """
        pass

    @property
    @abstractmethod
    def sensor_fusion_algorithm(self) -> str:
        """
        Selected sensor fusion algorithm.

        returns:
            str: algorithm name.
        """
        pass

    @property
    @abstractmethod
    def mode_management(self) -> dict:
        """
        Mode management thresholds.

        returns (dict):
            DetumblingOff (float): detumbling off threshold (deg/s).
            DetumblingOn (float): detumbling on threshold (deg/s).
            PointingOff (float): pointing off error angle (deg).
            PointingOn (float): pointing on error angle (deg).
            PointingDwellTime (int): pointing dwell time (s).
        """
        pass
