import numpy as np
from templates.initial_settings_template import SimulationSetup
from templates.satellite_template import Satellite


class MagnetorquerImplemetation:
    """
    Abstract class for magnetorquer parameters.
    """

    def __init__(
            self, 
            setup: SimulationSetup, 
            satellite: Satellite,
            k: float = 10000,
            safety_factor: float = 0.9,
    ):
        """
        Initialize the magnetorquer with the satellite and setup parameters.

        Args:
            setup (SimulationSetup): The simulation setup object.
            satellite (Satellite): The satellite object.
            k (float): The gain factor for the b-dot control law.
            safety_factor (float): A factor to limit the current in the 
                magnetorquer to a safe value.
        """
        self._setup = setup
        self._satellite = satellite

        self.no_of_coils = self._setup.magnetorquer_params[0]
        self.coil_area = self._setup.magnetorquer_params[1]
        self.coil_area = self.coil_area * 1e-4  # Convert cm^2 to m^2
        self.max_current = self._setup.magnetorquer_params[2]
        self.inertia = satellite.inertia_matrix
        self.mass = satellite.mass
        self.safety_factor = safety_factor

        self.magnetic_field_prev = satellite.magnetic_field[0] * 1e-9  # nT to T
        self.db_dt_prev = np.zeros(3)

        # B-dot parameters
        self.k = k

    def b_dot(
        self,
        magnetic_field: np.ndarray,
        timestep: float,

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
            angular_velocity (np.ndarray): The current angular velocity vector.
            timestep (float): The time step for the simulation.

        Returns:
            np.ndarray: The angular acceleration vector to be applied to the satellite.
        """
        magnetic_field = magnetic_field * 1e-9  # nT to T

        db_dt = self.filtered_derivative(
            magnetic_field, 
            timestep
        )
        mag_dipol_mom_required = self.k * db_dt
        # Calculate current per axis and apply saturation
        current_per_axis = mag_dipol_mom_required / (self.no_of_coils * self.coil_area)
        current_per_axis = np.clip(current_per_axis,
                                   -self.max_current * self.safety_factor,
                                   self.max_current * self.safety_factor)
        
        # Calculate actual dipole moment after saturation
        mag_dipol_mom_real = current_per_axis * self.no_of_coils * self.coil_area
        torque = np.cross(mag_dipol_mom_real, magnetic_field)
        angular_acceleration = np.linalg.inv(self.inertia) @ torque
        return angular_acceleration
    
    def filtered_derivative(
        self,
        magnetic_field: np.ndarray,
        timestep: float,
        alpha: float = 0.8,
    ) -> np.ndarray:
        """
        Calculate the filtered derivative of the magnetic field.

        Args:
            magnetic_field (np.ndarray): The current magnetic field vector.
            timestep (float): The time step for the simulation.
            alpha (float): The filter coefficient.

        Returns:
            np.ndarray: The filtered derivative of the magnetic field.
        """
        difference = magnetic_field - self.magnetic_field_prev
        db_dt = difference / timestep

        if np.sum(self.db_dt_prev) != 0:
            filtered_db_dt = alpha * db_dt + (1 - alpha) * self.db_dt_prev
        else:
            filtered_db_dt = db_dt

        self.db_dt_prev = filtered_db_dt
        self.magnetic_field_prev = magnetic_field

        return filtered_db_dt
