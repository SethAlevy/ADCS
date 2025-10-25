import numpy as np
import core.utilities as ut
import core.transformations as tr
from scipy.spatial.transform import Rotation as R
from templates.initial_settings_template import SimulationSetup
from templates.satellite_template import Satellite


class MagnetorquerImplementation:
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
            alpha: float = 1.6,
            magnetic_field_ref: float = 45000,  # nT
            beta: float = 0.5,
            k_p: float = 0.2,
            k_c: float = 0.00038,
            k_cp: float = 0.012,  # a bit more damping gain
    ):
        # TODO add bang-bang control
        """
        Initialize the magnetorquer with the satellite and setup parameters.

        Args:
            setup (SimulationSetup): The simulation setup object.
            satellite (Satellite): The satellite object.
            safety_factor (float): A factor to limit the current in the
                magnetorquer to a safe value.
            k (float): The gain factor for the b-dot control law. Applied in the
                standard B-dot control law and is the base for adaptive versions.
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
        self.inertia_inv = np.linalg.inv(self.inertia)
        self.mass = satellite.mass
        self.safety_factor = safety_factor
        # Max achievable dipole (A·m^2)
        self.m_max = self.no_of_coils * self.coil_area * self.max_current * self.safety_factor

        self.magnetic_field_prev = satellite.magnetic_field[0].copy() * 1e-9  # nT to T
        # Use None for robust first-run handling in the derivative filter
        self.db_dt_prev = None

        # B-dot parameters
        self.k = k

        # B-dot adapted to angular velocity parameters
        self.angular_velocity_ref = ut.degrees_to_rad(angular_velocity_ref)
        self.alpha = alpha

        # B-dot adapted to magnetic field parameters
        self.adapt_magnetic_ref = magnetic_field_ref * 1e-9  # nT to T
        self.beta = beta

        # B-dot with proportional term
        self.k_p = k_p

        # B-cross parameters
        self.k_c = k_c

        # B-cross with proportional damping
        self.k_cp = k_cp

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
            timestep (float): The time step for the simulation.
            adapt_magnetic (bool): Whether to adapt the gain based on the magnetic
                field strength.
            adapt_angular (bool): Whether to adapt the gain based on the angular
                velocity.
            proportional (bool): Whether to include a proportional term based on
                the angular velocity.
            modified (bool): Whether to use the modified B-dot control law that
                is based directly on the angular velocity (gyroscopes) and
                magnetic field measurements, instead of magnetic field rate of change.

        Returns:
            np.ndarray: The angular acceleration vector to be applied to the satellite.
        """
        magnetic_field = magnetic_field * 1e-9  # nT to T
        angular_velocity = ut.degrees_to_rad(self._satellite.angular_velocity)

        # Safe adaptive gain (guard small norms)
        mag_field_norm = max(float(np.linalg.norm(magnetic_field)), 1e-9)
        angular_velocity_norm_deg = max(float(np.linalg.norm(
            angular_velocity)), 1e-9)

        # Compute the gain based on the selected methods
        if adapt_magnetic and adapt_angular:
            k = -self.k * (self.adapt_magnetic_ref / mag_field_norm) ** self.beta * \
                (angular_velocity_norm_deg / self.angular_velocity_ref) ** self.alpha
        elif adapt_angular:
            k = -self.k * (angular_velocity_norm_deg /
                           self.angular_velocity_ref) ** self.alpha
        elif adapt_magnetic:
            k = -self.k * (self.adapt_magnetic_ref / mag_field_norm) ** self.beta
        else:
            k = -self.k

        # Compute dB/dt only when needed (not modified B-dot)
        if not modified:
            db_dt = self.filtered_derivative(magnetic_field, timestep)  # T/s
        else:
            # Keep filter state aligned to avoid future spikes when re-enabled
            self.magnetic_field_prev = magnetic_field.copy()
            db_dt = None

        # Compute the required magnetic dipole moment based on the selected method
        if proportional and modified:
            mag_dipol_mom_required = k * \
                np.cross(angular_velocity, magnetic_field) - self.k_p * angular_velocity
        elif proportional:
            mag_dipol_mom_required = k * db_dt - self.k_p * angular_velocity
        elif modified:
            mag_dipol_mom_required = k * np.cross(angular_velocity, magnetic_field)
        else:
            mag_dipol_mom_required = k * db_dt

        current_per_axis = self.apply_torquer_with_saturation(mag_dipol_mom_required)
        return self.current_to_angular_acceleration(current_per_axis, magnetic_field)

    def current_to_angular_acceleration(
        self,
        current_per_axis: np.ndarray,
        magnetic_field: np.ndarray,
    ) -> np.ndarray:
        """
        Convert a given magnetic dipole moment to torque and then angular acceleration.

        Args:
            current_per_axis (np.ndarray): The saturated current per axis in A.
            magnetic_field (np.ndarray): The magnetic field vector in T.

        Returns:
            np.ndarray: The angular acceleration vector in rad/s².
        """
        mag_dipol_mom_real = current_per_axis * self.no_of_coils * self.coil_area
        self.torque = np.cross(mag_dipol_mom_real, magnetic_field)

        # Angular acceleration: ω̇ = I^{-1} (τ − ω × (I ω))
        angular_velocity = ut.degrees_to_rad(self._satellite.angular_velocity)
        i_omega = np.matmul(self.inertia, angular_velocity)
        coriolis = np.cross(angular_velocity, i_omega)
        return np.matmul(self.inertia_inv, (self.torque - coriolis))

    def apply_torquer_with_saturation(
        self,
        magnetic_dipole_moment: np.ndarray,
    ) -> np.ndarray:
        """
        Uniform-scaling saturation. Converts dipole (A·m^2) to per-axis current (A)
        and scales the whole vector so no axis exceeds its limit. The maximum current,
        coil area, and number of turns are predefined torquer parameters.

        Args:
            magnetic_dipole_moment (np.ndarray): The theoretical magnetic dipole moment
                vector in A·m^2.

        Returns:
            np.ndarray: The saturated current per axis in A.
        """
        # Dipole to current
        n_a = float(self.no_of_coils * self.coil_area)
        current_cmd = np.asarray(magnetic_dipole_moment, dtype=float) / n_a

        # Per-axis limits (support scalar or 3-vector), apply safety factor
        current_max = np.asarray(self.max_current, dtype=float)
        if current_max.size == 1:
            current_max = np.full(3, float(current_max))
        current_max = np.abs(current_max) * float(self.safety_factor)

        # One scale for all axes to preserve direction
        denom = np.maximum(np.abs(current_cmd), 1e-12)
        scale = float(min(1.0, np.min(current_max / denom)))
        return np.clip(current_cmd * scale, -current_max, current_max)

    def filtered_derivative(
        self,
        magnetic_field: np.ndarray,
        timestep: float,
        alpha: float = 0.7,
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

        if self.db_dt_prev is None:
            filtered_db_dt = db_dt
        else:
            filtered_db_dt = alpha * db_dt + (1 - alpha) * self.db_dt_prev

        self.db_dt_prev = filtered_db_dt.copy()
        self.magnetic_field_prev = magnetic_field.copy()
        return filtered_db_dt

    def b_cross(
        self,
        magnetic_field_sbf: np.ndarray,
        align_axis: np.ndarray | list,
        target_dir_body: np.ndarray,
    ) -> np.ndarray:
        """
        B-cross pointing control (nadir). Generates angular acceleration (rad/s^2).
        Single switching threshold at 10 deg.
        """

        magnetic_field_sb = magnetic_field_sbf * 1e-9  # nT to T
        align_axis = np.asarray(align_axis, dtype=float)
        align_axis = align_axis / (np.linalg.norm(align_axis) + 1e-20)
        target_dir_body = ut.normalize(target_dir_body)

        # Geometric error e = a × t
        error_vec = np.cross(align_axis, target_dir_body)

        # Rates and base gains
        angular_velocity_rad_s = ut.degrees_to_rad(self._satellite.angular_velocity)
        gain_align = self.k_c * self.m_max if self.k_c <= 1.0 else self.k_c
        gain_damp = self.k_cp * self.m_max if self.k_cp <= 1.0 else self.k_cp

        B2 = float(np.dot(magnetic_field_sb, magnetic_field_sb))
        if B2 <= 0.0:
            commanded_dipole = np.zeros(3)
        else:

            # Classic B-cross (dipole space)
            m_align_raw = -gain_align * \
                np.cross(magnetic_field_sb, error_vec) / B2
            m_damp_raw = -gain_damp * \
                np.cross(magnetic_field_sb, angular_velocity_rad_s) / B2

            m_align = ut.limit_norm(m_align_raw, self.m_max)
            m_damp = ut.limit_norm(m_damp_raw, self.m_max)
            commanded_dipole = m_align + m_damp

        # Actuation and resulting angular acceleration
        current_per_axis = self.apply_torquer_with_saturation(commanded_dipole)
        angular_acceleration_rad_s2 = self.current_to_angular_acceleration(
            current_per_axis, magnetic_field_sb
        )
        # print(f"angle_deg_all: {angle_deg_all}, m_align: {m_align}, m_damp: {m_damp}")
        # print(f"commanded_dipole: {commanded_dipole}")
        # print(f"m_max: {self.m_max}")
        # print(f"angular_acceleration_rad_s2: {angular_acceleration_rad_s2}, angular_velocity_rad_s: {angular_velocity_rad_s}")
        # print("-----")
        # -------------------------------------------------------------------------
        return angular_acceleration_rad_s2

    def get_pointing_error_angle(self, q: np.ndarray) -> float:
        correction_rotation_object = R.from_quat(q)
        error_angle_rad = correction_rotation_object.magnitude()
        return ut.rad_to_degrees(error_angle_rad)
