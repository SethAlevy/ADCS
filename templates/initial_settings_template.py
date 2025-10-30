import numpy as np
from abc import ABC
from abc import abstractmethod
import datetime


class SimulationSetup(ABC):
    """
    Abstract class for simulation setup. Defines the required initial
    parameters given in setup/initial_parameters.json file.
    """

    @property
    @abstractmethod
    def euler_angles(self) -> tuple[float, float, float]:
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
    def angular_velocity(self) -> tuple[float, float, float]:
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
    def iterations_info(self) -> tuple[int, int, int]:
        """
        Simulation time parameters in seconds.

        returns:
            int: start time
            int: end time
            int: time step
        """
        pass

    @property
    @abstractmethod
    def magnetorquer_params(self) -> tuple[int, int, float]:
        """
        Magnetorquer parameters, works for every axis of rotation.

        returns:
            int: n_coils - number of coils.
            int: coil_area - area of each coil in cm^2.
            float: max_current - maximum current in A.
        """
        pass

    @property
    @abstractmethod
    def satellite_params(self) -> tuple[int, np.ndarray]:
        """
        Satellite parameters).

        returns:
            int: mass of the satellite in kg.
            np.ndarray: inertia matrix in kg*m^2.
        """
        pass

    @property
    @abstractmethod
    def planet_data(self) -> dict[float, float, float]:
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

        returns:
            datetime: date_time - date and time of the simulation start.
        """
        pass

    @property
    @abstractmethod
    def sensors_on_time(self) -> int:
        """
        Time interval for which the sensors are active.

        returns:
            int: on time in seconds.
        """
        pass

    @property
    @abstractmethod
    def magnetometer(self) -> tuple[bool, float]:
        """
        Magnetometer settings.

        returns:
            bool: noise flag.
            float: maximum noise amplitude (nT).
        """
        pass

    @property
    @abstractmethod
    def sunsensor(self) -> tuple[bool, float]:
        """
        Sun sensor settings.

        returns:
            bool: noise flag.
            float: angular noise (deg).
        """
        pass

    @property
    @abstractmethod
    def gyroscope(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Gyroscope settings.

        returns:
            np.ndarray: bias (deg/s).
            np.ndarray: process noise (deg/s).
        """
        pass

    @property
    @abstractmethod
    def quest(self) -> np.ndarray:
        """
        QUEST parameters.

        returns:
            np.ndarray: weights for measurements.
        """
        pass

    @property
    @abstractmethod
    def ekf(self) -> tuple[np.ndarray, float, float]:
        """
        EKF parameters.

        returns:
            np.ndarray: attitude noise in degrees.
            float: covariance value.
            float: measurement noise value.
        """
        pass

    @property
    @abstractmethod
    def actuators_on_time(self) -> int:
        """
        Time interval for which the actuators are active.

        returns:
            int: on time in seconds.
        """
        pass

    @property
    @abstractmethod
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
        pass

    @property
    @abstractmethod
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
        pass

    @property
    @abstractmethod
    def b_cross_mode(self) -> tuple[str, np.ndarray]:
        """
        B-cross mode settings.

        returns:
            str: target mode ("earth_pointing" or "sun_pointing").
            np.ndarray: pointing axis.
        """
        pass

    @property
    @abstractmethod
    def b_cross_parameters(self) -> tuple[float, float]:
        """
        B-cross control parameters.

        returns:
            float: align gain that determines how quickly the system aligns with 
                the target.
            float: proportional gain that determines how much the system will damp
            angular velocity.
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
        pass
