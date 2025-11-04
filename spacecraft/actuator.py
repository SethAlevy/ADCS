import numpy as np
import core.utilities as ut
from scipy.spatial.transform import Rotation as R
from templates.initial_settings_template import SimulationSetup
from templates.satellite_template import Satellite


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

        # Command shaping (dipole)
        self._m_filt: np.ndarray = np.zeros(3, dtype=float)
        self._prev_m: np.ndarray = np.zeros(3, dtype=float)
        self._m_tau: float = 20.0          # seconds (0 → disable LPF)
        self._m_slew: float = 0.02         # A·m²/s (0 → disable slew)

        # Command shaping (torque) – simple and robust
        self._tau_filt: np.ndarray = np.zeros(3, dtype=float)
        self._tau_prev: np.ndarray = np.zeros(3, dtype=float)

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

        # --- b_cross-only damping gate: force damp-only above 0.3 deg/s until < 0.2 deg/s
        self._bcross_damp_only_gate = False

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

    def coils_to_dipole(self, current_per_axis: np.ndarray) -> np.ndarray:
        """
        Convert coil currents [A] to magnetic dipole [A·m²].
        """
        return np.asarray(current_per_axis, dtype=float) * self.no_of_coils * self.coil_area

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

        angular_velocity_rad_s = ut.degrees_to_rad(self._satellite.angular_velocity)
        omega_norm = np.linalg.norm(angular_velocity_rad_s)

        #     pointing_omega_threshold_off_mid)
        pointing_deg_threshold = 50.0

        # Quaternion error (align_axis -> target_dir_body), then rotation vector
        a = align_axis
        tgt = target_dir_body
        dot = float(np.clip(np.dot(a, tgt), -1.0, 1.0))
        v = np.cross(a, tgt)
        if dot >= 1.0 - 1e-12:
            error_vec = np.zeros(3)
            theta = 0.0
        elif dot <= -1.0 + 1e-12:
            # 180° rotation: choose any orthogonal axis
            tmp = np.array([1.0, 0.0, 0.0])
            if abs(a[0]) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0])
            axis = np.cross(a, tmp)
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            error_vec = axis * np.pi
            theta = np.pi
        else:
            q = np.array([v[0], v[1], v[2], 1.0 + dot])  # [x,y,z,w]
            q = q / (np.linalg.norm(q) + 1e-12)
            rot = R.from_quat(q)
            error_vec = rot.as_rotvec()  # [rad]
            theta = float(np.linalg.norm(error_vec))

        if theta < np.deg2rad(0.2):  # ~0.2 deg
            error_vec = np.zeros(3)

        gain_align = self.k_c
        gain_damp = self.k_cp

        # --- B magnitude clamp and ω projection (defines b, inv_b2, omega_perp) ---
        b = magnetic_field_sb
        b_norm = float(np.linalg.norm(b))
        b_safe = max(b_norm, 1e-9)
        inv_b2 = 1.0 / (b_safe * b_safe)
        if b_norm > 0.0:
            b_hat = b / b_norm
            omega_par = b_hat * float(np.dot(angular_velocity_rad_s, b_hat))
        else:
            b_hat = np.zeros(3)
            omega_par = np.zeros(3)
        omega_perp = angular_velocity_rad_s - omega_par

        # --- compute dipoles (exclusive) ---
        # Alignment term (only when not damping)
        m_align = gain_align * np.cross(error_vec, b) * inv_b2  # A·m^2

        # Damping term (only when damping)
        m_damp = gain_damp * np.cross(omega_perp, b) * inv_b2  # A·m^2
        m_damp_base = m_damp.copy()  # keep base (pre-angle-logic) for overrides

        if theta > np.deg2rad(pointing_deg_threshold):
            m_align = np.zeros(3)
        if theta < np.deg2rad(pointing_deg_threshold) and theta > np.deg2rad(15.0):
            m_damp = np.zeros(3)
        if theta < np.deg2rad(15.0):
            m_align = np.zeros(3)
            # m_damp *= 10

        # --- High-rate hysteresis: force DAMP ONLY above 0.3 deg/s until < 0.2 deg/s ---
        omega_on = ut.degrees_to_rad(0.3)  # enter damp-only
        omega_off = ut.degrees_to_rad(0.2)  # exit damp-only
        if self._bcross_damp_only_gate:
            if omega_norm < omega_off:
                self._bcross_damp_only_gate = False
        else:
            if omega_norm > omega_on:
                self._bcross_damp_only_gate = True

        if self._bcross_damp_only_gate:
            # Override angle bands while gated: damping only, using base damping term
            m_align = np.zeros(3)
            m_damp = m_damp_base

        # --- b_cross-only angular-velocity scaling of dipole (τ scales with m) ---
        omega_norm_prev = getattr(self, "_bcross_omega_norm_prev", None)
        deadband = ut.degrees_to_rad(1e-3)   # ignore tiny changes
        scale_inc = 0.4                      # when |ω| increases
        scale_dec = 1.8                     # when |ω| decreases
        scale = 1.0
        if omega_norm_prev is not None and not self._bcross_damp_only_gate:
            d_omega = float(omega_norm - omega_norm_prev)
            if d_omega > deadband:
                scale = scale_inc
            elif d_omega < -deadband:
                scale = scale_dec
        self._bcross_omega_norm_prev = float(omega_norm)

        commanded_dipole_raw = (m_align + m_damp) * scale

        # Unified dipole shaping: LPF + slew (uses self._m_tau, self._m_slew)
        dt = float(self._setup.iterations_info.get("step", 1.0))
        commanded_dipole = self._shape_dipole(commanded_dipole_raw, dt)

        current_per_axis = self.apply_torquer_with_saturation(commanded_dipole)
        m_actual = self.coils_to_dipole(current_per_axis)

        # compute actual angular acceleration from actual m (for physics & logging)
        angular_acceleration_rad_s2 = self.current_to_angular_acceleration(
            current_per_axis, b)

        # diagnostics using actual applied dipole (m_actual)
        tau = np.cross(m_actual, b)
        power = float(np.dot(tau, angular_velocity_rad_s))
        if (self._satellite.iteration % 1) == 0:
            print(f" --- iteration {self._satellite.iteration} diagnostics ---")
            print(f"theta_deg: {np.degrees(theta):.3f}, omega_norm: {omega_norm:.4e}")
            print(f"m_align:{m_align}, ||m_align||:{np.linalg.norm(m_align):.3e}")
            print(f"m_damp:{m_damp}, ||m_damp||:{np.linalg.norm(m_damp):.3e}")
            print(
                f"tau·omega: {power:.3e}, omega: {np.rad2deg(angular_velocity_rad_s)}")

        # store angle in degrees (fix unit bug)
        self._prev_angle_deg = float(np.degrees(theta))
        self.omega_prev = np.rad2deg(angular_velocity_rad_s)
        return angular_acceleration_rad_s2

    def check_for_omega_sign_change(self, angular_velocity_deg_s: np.ndarray) -> list[bool]:
        sign_changes = []
        for current, previous in zip(angular_velocity_deg_s, self.omega_prev):
            sign_changes.append(np.sign(current) != np.sign(previous))
        return sign_changes

    def get_pointing_error_angle(self, q: np.ndarray) -> float:
        correction_rotation_object = R.from_quat(q)
        error_angle_rad = correction_rotation_object.magnitude()
        return ut.rad_to_degrees(error_angle_rad)

    def _recent_ang_vel_from_state(self, n: int) -> np.ndarray | None:
        """
        Fetch last n angular velocity rows [deg/s] from the state vector DataFrame.
        Expected columns: angular_velocity_x, angular_velocity_y, angular_velocity_z.
        Returns None if unavailable.
        """
        try:
            df = self._satellite.state_vector.to_dataframe()
            cols = ["angular_velocity_x", "angular_velocity_y", "angular_velocity_z"]
            if not all(c in df.columns for c in cols):
                return None
            arr = df[cols].tail(int(max(1, n))).to_numpy(dtype=float)
            return arr if arr.size else None
        except Exception:
            return None

    def _shape_dipole(self, m_cmd: np.ndarray, dt: float) -> np.ndarray:
        """
        First-order low-pass toward m_cmd and vector slew limit.
        Set self._m_tau=0 and self._m_slew=0 to bypass.
        """
        m_cmd = np.asarray(m_cmd, dtype=float)

        # LPF
        if self._m_tau > 0:
            alpha = float(np.clip(dt / max(self._m_tau, 1e-6), 0.0, 1.0))
            self._m_filt = self._m_filt + alpha * (m_cmd - self._m_filt)
            m_lp = self._m_filt
        else:
            m_lp = m_cmd

        # Slew limit
        if self._m_slew > 0:
            delta = m_lp - self._prev_m
            max_step = float(self._m_slew) * dt
            n = float(np.linalg.norm(delta))
            if n > max_step and n > 0.0:
                delta *= (max_step / n)
            self._prev_m = self._prev_m + delta
            return self._prev_m
        else:
            self._prev_m = m_lp
            return m_lp
