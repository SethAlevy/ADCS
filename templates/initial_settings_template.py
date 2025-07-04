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
        Initial angular velocity (wx, wy, wx) in rad/s.

        returns:
            float: wx - velocity around x-axis.
            float: wy - velocity around y-axis.
            float: wz - velocity around z-axis.
        """
        pass

    @property
    @abstractmethod
    def iterations_info(self) -> tuple[int, int, int]:
        """
        Simulation time parameters (t0, t_end, t_step) in seconds.

        returns:
            int: t0 - start time
            int: t_end - end time
            int: t_step - time step
        """
        pass

    @property
    @abstractmethod
    def magnetorquer_params(self) -> tuple[int, int]:
        """
        Magnetorquer parameters (n, A), works for every axis of rotation.

        returns:
            int: n - number of coils.
            int: A - area of each coil in m^2.
        """
        pass

    @property
    @abstractmethod
    def satellite_params(self) -> tuple[int, np.ndarray]:
        """
        Satellite parameters (I, m).

        returns:
            int: m - mass of the satellite in kg.
            np.ndarray: I - inertia matrix in kg*m^2.
        """
        pass

    @property
    @abstractmethod
    def planet_data(self) -> tuple[float, float, float]:
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
