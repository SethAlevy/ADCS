import numpy as np
from abc import ABC
from abc import abstractmethod


class Magnetorquer(ABC):
    """
    Abstract class for magnetorquer parameters.
    """

    @abstractmethod
    def b_dot(
        self,
        magnetic_field: np.ndarray,
        timestep: float,
        k: float = 1e-6,
        safety_factor: float = 0.8,
    ) -> np.ndarray:
        """
        Perform detumbling using the magnetorquer and standard B-dot algorithm.
        This method calculates the required magnetic dipole moment (essentially the
        magnetic field of the magnetorquer) to generate opposing torque and
        reduce the angular velocity of the satellite. The assumption in this code
        are three orthogonal coils, each with the same area and number of turns.

        Useful link:
        https://www.uio.no/studier/emner/matnat/fys/FYS3240/v23/lectures/l11---control-systems-v23.pdf
        https://www.aero.iitb.ac.in/satelliteWiki/index.php/B_Dot_Law

        Args:
            magnetic_field (np.ndarray): The current magnetic field vector.
            angular_velocity (np.ndarray): The current angular velocity vector.
            timestep (float): The time step for the simulation.
            k (float): The gain factor for the control law.
            safety_factor (float): A factor to limit the current to a safe value.

        Returns:
            np.ndarray: The angular acceleration vector to be applied to the satellite.
        """
        pass