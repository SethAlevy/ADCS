import numpy as np
from abc import ABC, abstractmethod


class Magnetorquer(ABC):
    """
    Abstract class for magnetorquer parameters.
    """

    @abstractmethod
    def b_dot(
        self,
        magnetic_field: np.ndarray,
        sensing_time: float,
        adapt_magnetic: bool = False,
        adapt_angular: bool = False,
        proportional: bool = False,
        modified: bool = False,
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
            magnetic_field (np.ndarray): The current magnetic field vector in nT.
            sensing_time (float): Sensor on time used to get the mean derivative
                from the magnetic field measurements.
            adapt_magnetic (bool): Whether to adapt the gain based on the magnetic
                field strength.
            adapt_angular (bool): Whether to adapt the gain based on the angular
                velocity to include damping.
            proportional (bool): Whether to include a proportional term based on
                the angular velocity.
            modified (bool): Whether to use the modified B-dot control law that
                is based directly on the angular velocity (gyroscopes) and
                magnetic field measurements, instead of magnetic field rate of change.

        Returns:
            np.ndarray: The angular acceleration vector to be applied to the satellite.
        """
        pass

    @abstractmethod
    def b_cross(
        self,
        magnetic_field_sbf: np.ndarray,
        align_axis: np.ndarray | list,
        target_dir_body: np.ndarray,
    ) -> np.ndarray:
        """
        B-cross pointing control (Earth or Sun pointing). Generates angular acceleration 
        (rad/s^2) based on the error angle. The method combines alignment and damping
        torques to achieve stable pointing.

        Args:
            magnetic_field_sbf (np.ndarray): The magnetic field vector in the
                spacecraft body frame in nT.
            align_axis (np.ndarray | list): The axis in the body frame to be aligned
                with the target direction. Specified inside the initial settings json
                file.
            target_dir_body (np.ndarray): The target direction vector in the body frame.
                Calculated based on the specified mode ("earth_pointing" or
                "sun_pointing").
        """
        pass
