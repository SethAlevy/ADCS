import numpy as np
import core.utilities as ut
from scipy.spatial.transform import Rotation as R
from templates.initial_settings_template import SimulationSetup
from templates.satellite_template import Satellite
import numpy as np
import core.utilities as ut


class MagnetorquerImplementation:
    """
    Abstract class for magnetorquer parameters.
    """

    def __init__(self, setup: SimulationSetup, satellite: Satellite):
        # TODO add bang-bang control
        """
        Initialize the magnetorquer with the satellite and setup parameters.

        Args:
            setup (SimulationSetup): The simulation setup object. Contains magnetorquer
            parameters as well as detumbling settings.
            satellite (Satellite): The satellite object.
            safety_factor (float): A factor to limit the current in the
                magnetorquer to a safe value.
        """
        self._setup = setup
        self._satellite = satellite
        self._prev_angle_deg = None
        self._prev_time_s = None
        # tunables (fallback defaults if missing in JSON)
        self._angle_rate_max = getattr(setup, "bcross_angle_rate_max", -0.12)   # deg/s
        self._ramp_seconds = getattr(setup, "bcross_ramp_seconds", 300.0)   # s
        self._alpha_cap_deg = getattr(
            setup, "bcross_alpha_cap_deg_s2", 0.008)  # deg/s^2

        # Command shaping (defaults if not in setup)
        self._m_filt = np.zeros(3)
        self._prev_m = np.zeros(3)
        self._m_tau = getattr(setup, "m_filter_tau", 30.0)   # s, low-pass time constant
        self._m_slew = getattr(setup, "m_slew_limit", 0.02)  # A·m^2 per s (vector slew)

        self.no_of_coils = self._setup.magnetorquer_params["n_coils"]
        self.coil_area = self._setup.magnetorquer_params["coil_area"]
        self.coil_area = self.coil_area * 1e-4  # Convert cm^2 to m^2
        self.max_current = self._setup.magnetorquer_params["max_current"]
        self.safety_factor = setup.magnetorquer_params["safety_factor"]

        self.inertia = satellite.inertia_matrix
        self.inertia_inv = np.linalg.inv(self.inertia)
        self.mass = satellite.mass

        self.magnetic_field_prev = satellite.magnetic_field[0].copy() * 1e-9  # nT to T
        # Use None for robust first-run handling in the derivative filter
        self.db_dt_prev = None

        # B-dot parameters
        self.k = setup.b_dot_parameters["gain"]

        # B-dot adapted to angular velocity parameters
        angular_velocity_ref = setup.b_dot_parameters["angular_velocity_ref"]
        self.angular_velocity_ref = ut.degrees_to_rad(angular_velocity_ref)
        self.alpha = setup.b_dot_parameters["alpha"]

        # B-dot adapted to magnetic field parameters
        magnetic_field_ref = setup.b_dot_parameters["magnetic_field_ref"]
        self.adapt_magnetic_ref = magnetic_field_ref * 1e-9  # nT to T
        self.beta = setup.b_dot_parameters["beta"]

        # B-dot with proportional term
        self.k_p = setup.b_dot_parameters["proportional_gain"]

        # B-cross parameters
        self.k_c = setup.b_cross_parameters["align_gain"]

        # B-cross with proportional damping
        self.k_cp = setup.b_cross_parameters["proportional_gain"]

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
                velocity to include damping.
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
        # vector saturation that preserves direction
        m_cmd = np.asarray(magnetic_dipole_moment, dtype=float)
        m_max = float(self.no_of_coils * self.coil_area *
                      self.max_current * self.safety_factor)
        norm = np.linalg.norm(m_cmd)
        if norm > m_max and norm > 0:
            m_cmd = m_cmd * (m_max / norm)

        # map dipole to coil currents and clamp per-axis as a final safety
        i_cmd = m_cmd / (self.no_of_coils * self.coil_area)
        i_lim = self.max_current * self.safety_factor
        i_cmd = np.clip(i_cmd, -i_lim, i_lim)
        return i_cmd

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
        B-cross pointing control (nadir). Generates angular acceleration (rad/s^2)
        based in the error angle. The method combines alignment and damping
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

        magnetic_field_sb = magnetic_field_sbf * 1e-9  # nT to T
        align_axis = np.asarray(align_axis, dtype=float)
        align_axis = ut.normalize(align_axis)
        target_dir_body = ut.normalize(target_dir_body)

        # Geometric error e = a × t
        error_vec = np.cross(align_axis, target_dir_body)

        # Angle for info
        cos_at = float(np.clip(np.dot(align_axis, target_dir_body), -1.0, 1.0))
        angle_deg_all = ut.rad_to_degrees(float(np.arccos(cos_at)))

        # Rates and base gains
        angular_velocity_rad_s = ut.degrees_to_rad(self._satellite.angular_velocity)

        # time and dt
        t_s = float(self._satellite.iteration)

        # 1) longer align ramp
        ramp = min(1.0, max(0.0, t_s / max(self._ramp_seconds, 1e-3)))
        gain_align = self.k_c * ramp
        gain_damp = self.k_cp
        B2 = float(np.dot(magnetic_field_sb, magnetic_field_sb))

        # compute raw dipoles
        if B2 > 0.0:
            m_align = -gain_align * np.cross(magnetic_field_sb, error_vec) / B2
            m_damp = -gain_damp * \
                    np.cross(magnetic_field_sb, angular_velocity_rad_s) / B2
        else:
            m_align = np.zeros(3)
            m_damp = np.zeros(3)

        # 2) dynamic approach-rate gate on ALIGN only when it would add energy
        dtheta_dt = 0.0 if self._prev_angle_deg is None else (angle_deg_all - self._prev_angle_deg)
        if dtheta_dt < self._angle_rate_max and dtheta_dt < 0.0:
            if angle_deg_all >= 30.0:
                m_align = np.zeros(3)

        if angle_deg_all < 5.0:
            m_damp = m_damp * np.sqrt(5.0 / angle_deg_all)

        commanded_dipole = m_align + m_damp

        # 3) vector angular-acceleration cap (preserves direction)
        commanded_dipole, alpha_scale = self.limit_by_alpha(
            commanded_dipole, magnetic_field_sb, alpha_max_deg=self._alpha_cap_deg
        )

        # actuate
        current_per_axis = self.apply_torquer_with_saturation(commanded_dipole)
        angular_acceleration_rad_s2 = self.current_to_angular_acceleration(
            current_per_axis, magnetic_field_sb)

        # history + brief log
        self._prev_angle_deg = angle_deg_all
        self._prev_time_s = t_s
        if (self._satellite.iteration % 25) == 0:
            print(f" B-cross: angle={angle_deg_all} deg"
                  f" align dipole ={m_align} damp dipole ={m_damp:}")
            print(
                f"   approach_gate: dθ/dt={dtheta_dt:.3f}  ramp={ramp:.2f}")
        return angular_acceleration_rad_s2

    def get_pointing_error_angle(self, q: np.ndarray) -> float:
        correction_rotation_object = R.from_quat(q)
        error_angle_rad = correction_rotation_object.magnitude()
        return ut.rad_to_degrees(error_angle_rad)
