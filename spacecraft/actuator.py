import numpy as np
import core.utilities as ut
import core.transformations as tr
from scipy.spatial.transform import Rotation as R
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
            safety_factor: float = 0.95,
            k: float = 8000,
            angular_velocity_ref: float = 4,  # deg/s
            alpha: float = 1.8,
            magnetic_field_ref: float = 45000,  # nT
            beta: float = 0.5,
            k_p: float = 0.3,
            k_c: float = 0.5,
    ):
        # TODO add bang-bang control
        """
        Initialize the magnetorquer with the satellite and setup parameters.

        Args:
            setup (SimulationSetup): The simulation setup object.
            satellite (Satellite): The satellite object.
            safety_factor (float): A factor to limit the current in the 
                magnetorquer to a safe value.
            k (float): The gain factor for the b-dot control law. Is also the base
                value for adaptive variants.
            angular_velocity_ref (float): Reference angular velocity for the
                adaptive B-dot control law in deg/s. Is the assumed value when the
                algorithm should switch from fast detumbling to more control.
            alpha (float): Exponent for the angular velocity adaptation.
            magnetic_field_ref (float): Reference magnetic field for the
                adaptive B-dot control law in nT. Is the assumed somewhere
                about the average magnetic field on the low Earth orbit.
            beta (float): Exponent for the magnetic field adaptation.
            k_p (float): Proportional gain for the B-dot control law. Determines 
                how much the magnetic dipole moment is adjusted based on the
                angular velocity.
            k_c (float): Control gain for the B-cross control law.
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

        self.magnetic_field_prev = satellite.magnetic_field[0].copy() * 1e-9  # nT to T
        self.db_dt_prev = np.zeros(3)

        # B-dot parameters
        self.k = k

        # B-dot adapted to angular velocity parameters
        self.angular_velocity_ref = angular_velocity_ref  # deg/s
        self.alpha = alpha

        # B-dot adapted to magnetic field parameters
        self.adapt_magnetic_ref = magnetic_field_ref  # nT
        self.beta = beta

        # B-dot with proportional term
        self.k_p = k_p

        # B-cross parameters
        self.k_c = k_c
        self.pointing_error_angle = 0.0  # Initialize pointing error angle

    def b_dot(
        self,
        magnetic_field: np.ndarray,
        timestep: float,
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
            angular_velocity (np.ndarray): The current angular velocity vector.
            timestep (float): The time step for the simulation.

        Returns:
            np.ndarray: The angular acceleration vector to be applied to the satellite.
        """
        if adapt_magnetic and adapt_angular:
            k = -self.k * (
                self.adapt_magnetic_ref / np.linalg.norm(magnetic_field)
            ) ** self.beta * (
                np.linalg.norm(self._satellite.angular_velocity) / self.angular_velocity_ref
            ) ** self.alpha
        elif adapt_angular:
            k = -self.k * (
                np.linalg.norm(self._satellite.angular_velocity) / self.angular_velocity_ref
            ) ** self.alpha
        elif adapt_magnetic:
            k = -self.k * (
                self.adapt_magnetic_ref / np.linalg.norm(magnetic_field)
            ) ** self.beta
        else:
            k = -self.k

        # print(f"Gain: {k}")
        magnetic_field = magnetic_field * 1e-9  # nT to T
        # print(f"Reference angular velocity (rad/s): {self.angular_velocity_ref}")

        db_dt = self.filtered_derivative(
            magnetic_field, 
            timestep
        )

        if proportional and modified:
            angular_velocity = ut.degrees_to_rad(self._satellite.angular_velocity)
            mag_dipol_mom_required = k * np.cross(angular_velocity, magnetic_field) - self.k_p * angular_velocity
        elif proportional:
            angular_velocity = ut.degrees_to_rad(self._satellite.angular_velocity)
            mag_dipol_mom_required = k * db_dt - self.k_p * angular_velocity
        elif modified:
            angular_velocity = ut.degrees_to_rad(self._satellite.angular_velocity)
            mag_dipol_mom_required = k * np.cross(angular_velocity, magnetic_field)
        else:
            mag_dipol_mom_required = k * db_dt
        current_per_axis = self.apply_torquer_saturation(mag_dipol_mom_required)
        # Calculate actual dipole moment after saturation
        mag_dipol_mom_real = current_per_axis * self.no_of_coils * self.coil_area
        # print(f"Angular velocity (deg/s): {self._satellite.angular_velocity}")
        # print(f"Magnetic field (T): {magnetic_field}")
        # print(f"Magnetic dipole moment b-dot: {mag_dipol_mom_required}")
        # print(f"Magnetic dipole modified b-dot: {k * np.cross(ut.degrees_to_rad(self._satellite.angular_velocity), magnetic_field)}")
        self.torque = np.cross(mag_dipol_mom_real, magnetic_field)
        # print(f"Angular acceleration (deg/s^2): {ut.rad_to_degrees(np.linalg.inv(self.inertia) @ self.torque)}")

        return np.linalg.inv(self.inertia) @ self.torque
    
    def apply_torquer_saturation(
        self,
        magnetic_dipole_moment: np.ndarray,
    ) -> np.ndarray:
        current_per_axis = magnetic_dipole_moment / (self.no_of_coils * self.coil_area)
        current_per_axis = np.clip(current_per_axis,
                                   -self.max_current * self.safety_factor,
                                   self.max_current * self.safety_factor)

        return current_per_axis
    
    def filtered_derivative(
        self,
        magnetic_field: np.ndarray,
        timestep: float,
        alpha: float = 0.9,
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

        # print("magnetic field (T):", magnetic_field)
        # print(f"previous magnetic field (T): {self.magnetic_field_prev}")
        # print(f"Difference in magnetic field (T): {difference}")
        # print(f"Unfiltered derivative (T/s): {db_dt}")

        if np.sum(self.db_dt_prev) != 0:
            filtered_db_dt = alpha * db_dt + (1 - alpha) * self.db_dt_prev
        else:
            filtered_db_dt = db_dt
        # print(f"Filtered derivative (T/s): {filtered_db_dt}")

        # print(f"Previous magnetic field (T): {self.magnetic_field_prev}")
        # print(f"Current magnetic field (T): {magnetic_field}")
        # print(f"Difference in magnetic field (T): {difference}")
        # print(f"Filtered derivative (T/s): {filtered_db_dt}")

        self.db_dt_prev = filtered_db_dt.copy()
        self.magnetic_field_prev = magnetic_field.copy()

        return filtered_db_dt
    
    def b_cross(
        self,
        magnetic_field_eci: np.ndarray,
        magnetic_field_sbf: np.ndarray,
        task: str,
        align_axis: np.ndarray | list,

    ) -> np.ndarray:
        """
        Calculate the B-cross term for the B-dot control law.

        Args:
            magnetic_field_eci (np.ndarray): The current magnetic field vector in ECIF.
            magnetic_field_sbf (np.ndarray): The current magnetic field vector in SBF.
            task (str): The task which the B-cross should perform. As the algorithm is
                dedicated to pointing, currently 'earth-pointing' and 'sun-pointing' 
                are supported.
            align_axis (np.ndarray | list): The axis which should be rotated towards
                the given target.

        Returns:
            np.ndarray: The B-cross term.
        """
        magnetic_field_eci = magnetic_field_eci * 1e-9  # nT to T
        magnetic_field_sbf = magnetic_field_sbf * 1e-9
        q_eci_to_sbf = self._satellite.quaternion

        # get the quaternion that rotates the desired axis in ECI frame
        q_pointing_eci = tr.to_earth_rotation(
            self._satellite.position,
            align_axis,
            q_eci_to_sbf
        )

        self.pointing_error_angle = self.get_pointing_error_angle(q_pointing_eci)

        q_eci_to_desired_sb = tr.quat_multiply(q_pointing_eci, q_eci_to_sbf)
        target_magnetic_field_sbf = tr.rotate_vector_by_quaternion(
            magnetic_field_eci, q_eci_to_desired_sb
        )

        mag_dipol_mom_required = self.k_c * np.cross(
            magnetic_field_sbf, target_magnetic_field_sbf
        )

        current_per_axis = self.apply_torquer_saturation(mag_dipol_mom_required)

        # Calculate actual dipole moment after saturation
        mag_dipol_mom_real = current_per_axis * self.no_of_coils * self.coil_area
        self.torque = np.cross(mag_dipol_mom_real, magnetic_field_sbf)
        return np.linalg.inv(self.inertia) @ self.torque

    def get_pointing_error_angle(self, q: np.ndarray) -> float:
        correction_rotation_object = R.from_quat(q)
        error_angle_rad = correction_rotation_object.magnitude()
        return ut.rad_to_degrees(error_angle_rad)