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
        self._m_tau: float = 10.0  # nominal
        self._m_slew: float = 0.1  # keep

        # Dipole shaper state (required by _shape_dipole)
        self._m_filt: np.ndarray = np.zeros(3, dtype=float)
        self._prev_m: np.ndarray = np.zeros(3, dtype=float)

        # Command shaping (torque) – simple and robust
        self._tau_filt: np.ndarray = np.zeros(3, dtype=float)
        self._tau_prev: np.ndarray = np.zeros(3, dtype=float)

        self.no_of_coils = self._setup.magnetorquer_params["Coils"]
        self.coil_area = self._setup.magnetorquer_params["CoilArea"]
        self.coil_area = self.coil_area * 1e-4  # Convert cm^2 to m^2
        self.max_current = self._setup.magnetorquer_params["MaxCurrent"]
        self.safety_factor = setup.magnetorquer_params["SafetyFactor"]
        self.alpha_cap = setup.magnetorquer_params.get("AlphaCap", 0.0)
        # Convert and store rad/s² or None if disabled.
        if self.alpha_cap and self.alpha_cap > 0.0:
            self._alpha_cap_rad = ut.degrees_to_rad(float(self.alpha_cap))
        else:
            self._alpha_cap_rad = None

        self.inertia = satellite.inertia_matrix
        self.inertia_inv = np.linalg.inv(self.inertia)
        self.mass = satellite.mass

        self.magnetic_field_prev = satellite.magnetic_field[0].copy() * 1e-9  # nT to T
        self.db_dt_prev = None

        # B-dot parameters
        self.k = setup.b_dot_parameters["Gain"]

        # B-dot adapted to angular velocity parameters
        angular_velocity_ref = setup.b_dot_parameters["AngularVelocityRef"]
        self.angular_velocity_ref = ut.degrees_to_rad(angular_velocity_ref)
        self.alpha = setup.b_dot_parameters["Alpha"]

        # B-dot adapted to magnetic field parameters
        magnetic_field_ref = setup.b_dot_parameters["MagneticFieldRef"]
        self.adapt_magnetic_ref = magnetic_field_ref * 1e-9  # nT to T
        self.beta = setup.b_dot_parameters["Beta"]

        # B-dot with proportional term
        self.k_p = setup.b_dot_parameters["ProportionalGain"]

        # B-cross parameters
        self.k_c = setup.b_cross_parameters["AlignGain"]

        # B-cross with proportional damping
        self.k_cp = setup.b_cross_parameters["ProportionalGain"]

        # b_cross-only damping gate: force damp-only above 0.3 deg/s until < 0.2 deg/s
        self._bcross_damp_only_gate = False
        self._poor_geom_gate = False

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
        magnetic_field = magnetic_field * 1e-9  # nT to T
        angular_velocity = ut.degrees_to_rad(self._satellite.angular_velocity)

        # Safe adaptive gain (guard small norms)
        mag_field_norm = max(float(np.linalg.norm(magnetic_field)), 1e-9)
        angular_velocity_norm_deg = max(float(np.linalg.norm(angular_velocity)), 1e-9)

        # Compute the gain based on the selected methods
        if adapt_magnetic and adapt_angular:
            k = (
                -self.k
                * (self.adapt_magnetic_ref / mag_field_norm) ** self.beta
                * (angular_velocity_norm_deg / self.angular_velocity_ref) ** self.alpha
            )
        elif adapt_angular:
            k = (
                -self.k
                * (angular_velocity_norm_deg / self.angular_velocity_ref) ** self.alpha
            )
        elif adapt_magnetic:
            k = -self.k * (self.adapt_magnetic_ref / mag_field_norm) ** self.beta
        else:
            k = -self.k

        # Compute dB/dt only when needed (not modified B-dot)
        if not modified:
            db_dt = self.filtered_derivative(magnetic_field, sensing_time)  # T/s
        else:
            # Keep filter state aligned to avoid future spikes when re-enabled
            self.magnetic_field_prev = magnetic_field.copy()
            db_dt = None

        # Compute the required magnetic dipole moment based on the selected method
        if proportional and modified:
            mag_dipol_mom_required = (
                k * np.cross(angular_velocity, magnetic_field)
                - self.k_p * angular_velocity
            )
        elif proportional:
            mag_dipol_mom_required = k * db_dt - self.k_p * angular_velocity
        elif modified:
            mag_dipol_mom_required = k * np.cross(angular_velocity, magnetic_field)
        else:
            mag_dipol_mom_required = k * db_dt

        current_per_axis, commanded_dipol = self.apply_torquer_with_saturation(
            mag_dipol_mom_required)
        angular_acceleration, current_per_axis = self.current_to_angular_acceleration(
            commanded_dipol,
            magnetic_field,
            current_per_axis,
        )
        return angular_acceleration

    def current_to_angular_acceleration(
        self,
        commanded_dipol: np.ndarray,
        magnetic_field: np.ndarray,
        current_per_axis: np.ndarray,
    ):
        """
        Convert a given magnetic dipole moment to torque and then angular acceleration.
        If alpha_cap is set, scale torque/current so ||alpha|| <= cap.

        Args:
            commanded_dipol (np.ndarray): The commanded magnetic dipole moment in
                A·m².
            magnetic_field (np.ndarray): The magnetic field vector in T.
            current_per_axis (np.ndarray): The saturated current per axis in A.
        Returns:
            (np.ndarray, np.ndarray):
                angular acceleration [rad/s²], current_per_axis [A].
        """

        tau_raw = np.cross(commanded_dipol, magnetic_field)

        # Dynamics terms
        omega_rad = ut.degrees_to_rad(self._satellite.angular_velocity)
        i_omega = np.matmul(self.inertia, omega_rad)
        coriolis = np.cross(omega_rad, i_omega)

        # Apply alpha cap exactly once (may scale torque and currents)
        tau_eff, alpha, s = self._apply_alpha_cap(tau_raw, coriolis)
        if s < 1.0:
            current_per_axis = current_per_axis * s

        # Store effective torque
        self.torque = tau_eff

        return alpha, current_per_axis

    def _apply_alpha_cap(
        self,
        tau_raw: np.ndarray,
        coriolis: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Enforce optional alpha cap by scaling magnetic torque.
        alpha = I^{-1}(s*tau_raw - coriolis) = s*a + b, s in [0,1].
        Returns (tau_limited, alpha_limited, s).
        """
        a = np.matmul(self.inertia_inv, tau_raw)
        b = -np.matmul(self.inertia_inv, coriolis)
        alpha = a + b
        s = 1.0

        if self._alpha_cap_rad is None:
            return tau_raw, alpha, s

        cap = float(self._alpha_cap_rad)
        if float(np.linalg.norm(alpha)) <= cap:
            return tau_raw, alpha, s

        A = float(np.dot(a, a))
        B = float(np.dot(a, b))
        C = float(np.dot(b, b)) - cap**2

        if A <= 1e-16:
            s = 0.0
        else:
            D = B**2 - A * C
            if D >= 0.0:
                r = np.sqrt(D)
                s1 = (-B - r) / A
                s2 = (-B + r) / A
                cand = [sv for sv in (s1, s2) if 0.0 <= sv <= 1.0]
                s = max(cand) if cand else float(np.clip(-B / A, 0.0, 1.0))
            else:
                s = float(np.clip(-B / A, 0.0, 1.0))

        tau_limited = s * tau_raw
        alpha_limited = s * a + b
        return tau_limited, alpha_limited, s

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
        m_max = float(
            self.no_of_coils * self.coil_area * self.max_current * self.safety_factor
        )
        norm = np.linalg.norm(m_cmd)
        if norm > m_max and norm > 0:
            m_cmd = m_cmd * (m_max / norm)

        # map dipole to coil currents and clamp per-axis as a final safety
        i_cmd = m_cmd / (self.no_of_coils * self.coil_area)
        i_lim = self.max_current * self.safety_factor
        i_cmd = np.clip(i_cmd, -i_lim, i_lim)
        return i_cmd, m_cmd

    def coils_to_dipole(self, current_per_axis: np.ndarray) -> np.ndarray:
        """
        Convert coil currents [A] to magnetic dipole [A·m²].
        """
        return (
            np.asarray(current_per_axis, dtype=float)
            * self.no_of_coils
            * self.coil_area
        )

    def filtered_derivative(
        self,
        magnetic_field: np.ndarray,
        sensing_time: float,
        alpha: float = 0.7,
    ) -> np.ndarray:
        """
        Calculate the filtered derivative of the magnetic field.

        Args:
            magnetic_field (np.ndarray): The current magnetic field vector.
            sensing_time (float): Sensor on time used to get the mean derivative
                from the magnetic field measurements.
            alpha (float): The filter coefficient.

        Returns:
            np.ndarray: The filtered derivative of the magnetic field.
        """
        difference = magnetic_field - self.magnetic_field_prev
        db_dt = difference / sensing_time
        if self.db_dt_prev is None:
            filtered_db_dt = db_dt
        elif np.array_equal(db_dt, np.array([0.0, 0.0, 0.0])):
            filtered_db_dt = self.db_dt_prev
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
        magnetic_field_sb = magnetic_field_sbf * 1e-9  # nT to T
        align_axis = np.asarray(align_axis, dtype=float)
        align_axis = ut.normalize(align_axis)
        target_dir_body = ut.normalize(target_dir_body)

        angular_velocity_rad_s = ut.degrees_to_rad(self._satellite.angular_velocity)
        omega_norm = np.linalg.norm(angular_velocity_rad_s)

        error_vec, theta = self._compute_error_vec_and_theta(
            align_axis, target_dir_body
        )

        b, b_norm, b_hat, inv_b2, omega_par, omega_perp = self._b_decompose(
            magnetic_field_sb, angular_velocity_rad_s
        )

        # Base PD terms
        k_c_eff, k_cp_eff = self._bcross_schedule(theta)
        m_align_base = k_c_eff * np.cross(error_vec, b) * inv_b2
        m_damp_base = k_cp_eff * np.cross(omega_perp, b) * inv_b2

        # Geometry metric: sin(gamma) between target and B
        sin_gamma = float(np.linalg.norm(np.cross(target_dir_body, b))) / max(
            b_norm, 1e-12
        )

        # High-rate gate → damping only (takes precedence)
        gated = self._apply_high_rate_damp_gate(omega_norm, on_deg=0.18, off_deg=0.10)
        if gated:
            m_align_cmd = np.zeros_like(m_align_base)
            m_damp_cmd = 1.6 * m_damp_base
        else:
            # Poor-geometry gate → tiny damping, no alignment
            poor = self._apply_poor_geometry_gate(sin_gamma, on=0.25, off=0.35)
            if poor:
                m_align_cmd = np.zeros_like(m_align_base)
                m_damp_cmd = 0.1 * m_damp_base  # very small damping
            else:
                m_align_cmd = m_align_base
                m_damp_cmd = m_damp_base

        commanded_dipole_raw = m_align_cmd + m_damp_cmd

        dt = float(getattr(self._setup, "iterations_info", {}).get("Step", 1.0))
        commanded_dipole = self._shape_dipole(commanded_dipole_raw, dt)

        current_per_axis = self.apply_torquer_with_saturation(commanded_dipole)
        angular_acceleration_rad_s2, current_per_axis = (
            self.current_to_angular_acceleration(commanded_dipole, b, current_per_axis)
        )
        m_actual = self.coils_to_dipole(current_per_axis)

        # --- diagnostics to state vector (consistent with commanded terms) ---
        sv = getattr(
            self._satellite,
            "state_vector",
            getattr(self._satellite, "_state_vector", None),
        )
        if sv is not None:
            m_cmd_norm = float(np.linalg.norm(commanded_dipole_raw))
            m_act_norm = float(np.linalg.norm(m_actual))
            theta_deg = float(np.degrees(theta))

            sv.register_value("bcross_theta_deg", theta_deg)
            sv.register_value("bcross_m_cmd", m_cmd_norm)
            sv.register_value("bcross_m_act", m_act_norm)

        # store angle in degrees (fix unit bug)
        self._prev_angle_deg = float(np.degrees(theta))
        self.omega_prev = np.rad2deg(angular_velocity_rad_s)
        return angular_acceleration_rad_s2

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
                delta *= max_step / n
            self._prev_m = self._prev_m + delta
            return self._prev_m
        else:
            self._prev_m = m_lp
            return m_lp

    def _compute_error_vec_and_theta(
        self,
        align_axis: np.ndarray,
        target_dir_body: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Rotation vector (rad) to rotate align_axis into target_dir_body.
        Returns (error_vec [rad], theta [rad]). Includes small deadzone.
        """
        a = ut.normalize(np.asarray(align_axis, dtype=float))
        tgt = ut.normalize(np.asarray(target_dir_body, dtype=float))

        dot = float(np.clip(np.dot(a, tgt), -1.0, 1.0))
        v = np.cross(a, tgt)

        if dot >= 1.0 - 1e-12:
            error_vec = np.zeros(3, dtype=float)
            theta = 0.0
        elif dot <= -1.0 + 1e-12:
            # 180 deg: pick any axis orthogonal to a
            tmp = np.array([1.0, 0.0, 0.0], dtype=float)
            if abs(a[0]) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0], dtype=float)
            axis = np.cross(a, tmp)
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            error_vec = axis * np.pi
            theta = np.pi
        else:
            q = np.array([v[0], v[1], v[2], 1.0 + dot], dtype=float)  # [x,y,z,w]
            q = q / (np.linalg.norm(q) + 1e-12)
            rot = R.from_quat(q)
            error_vec = rot.as_rotvec()
            theta = float(np.linalg.norm(error_vec))

        # small deadzone
        if theta < np.deg2rad(1.0):  # was 0.5 deg
            error_vec = np.zeros(3, dtype=float)

        return error_vec, theta

    def _b_decompose(
        self,
        magnetic_field_sb: np.ndarray,
        angular_velocity_rad_s: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray, float, np.ndarray, np.ndarray]:
        """
        B-field quantities and ω decomposition.
        Returns (b, b_norm, b_hat, inv_b2, omega_par, omega_perp).
        """
        b = np.asarray(magnetic_field_sb, dtype=float)
        b_norm = float(np.linalg.norm(b))
        b_safe = max(b_norm, 1e-9)
        inv_b2 = 1.0 / (b_safe * b_safe)

        if b_norm > 0.0:
            b_hat = b / b_norm
            omega_par = b_hat * float(np.dot(angular_velocity_rad_s, b_hat))
        else:
            b_hat = np.zeros(3, dtype=float)
            omega_par = np.zeros(3, dtype=float)

        omega_perp = angular_velocity_rad_s - omega_par
        return b, b_norm, b_hat, inv_b2, omega_par, omega_perp

    def _apply_high_rate_damp_gate(
        self,
        omega_norm: float,
        on_deg: float = 0.3,
        off_deg: float = 0.2,
    ) -> bool:
        """
        Force damp-only when |ω| > on_deg [deg/s] until it falls below off_deg.
        Returns current gate state.
        """
        omega_on = ut.degrees_to_rad(on_deg)
        omega_off = ut.degrees_to_rad(off_deg)

        if self._bcross_damp_only_gate:
            if omega_norm <= omega_off:
                self._bcross_damp_only_gate = False
        elif omega_norm > omega_on:
            self._bcross_damp_only_gate = True

        return self._bcross_damp_only_gate

    def _omega_trend_scale(
        self,
        omega_norm: float,
        gated: bool,
        deadband_deg: float = 1e-3,
        scale_inc: float = 0.4,
        scale_dec: float = 1.8,
    ) -> float:
        """
        Scale dipole based on |ω| trend:
        - if increasing by > deadband → scale_inc
        - if decreasing by > deadband → scale_dec
        Disabled while 'gated' is True (state still updated).
        """
        if gated:
            self._bcross_omega_norm_prev = omega_norm
            return 1.0

        prev = getattr(self, "_bcross_omega_norm_prev", None)
        deadband = ut.degrees_to_rad(deadband_deg)
        scale = 1.0
        if prev is not None:
            d_omega = float(omega_norm - prev)
            if d_omega > deadband:
                scale = scale_inc
            elif d_omega < -deadband:
                scale = scale_dec

        self._bcross_omega_norm_prev = omega_norm
        return scale

    def _apply_angle_bands(
        self,
        theta: float,
        m_align_base: np.ndarray,
        m_damp_base: np.ndarray,
        pointing_deg_threshold_deg: float = 50.0,
        mid_deg: float = 15.0,
        small_angle_damp_multiplier: float = 1.0,
        mid_damp_fraction: float = 0.30,  # NEW
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply angle-band logic and return (m_align, m_damp).
        Defaults:
        - theta > pointing_deg_threshold_deg → pure damping
        - mid_deg < theta ≤ pointing_deg_threshold_deg → alignment + mid damping
        - theta ≤ mid_deg → pure damping (scaled by small_angle_damp_multiplier)
        """
        m_align = m_align_base.copy()
        m_damp = m_damp_base.copy()

        if theta > np.deg2rad(pointing_deg_threshold_deg):
            m_align[:] = 0.0
        elif theta > np.deg2rad(mid_deg):
            # mid band: do not zero damping completely
            m_damp = m_damp_base * mid_damp_fraction
        else:
            m_align[:] = 0.0
            m_damp *= small_angle_damp_multiplier

        return m_align, m_damp

    def _bcross_schedule(self, theta_rad: float) -> tuple[float, float]:
        th = np.degrees(theta_rad)
        if th > 60:  # large: mostly damp
            return 0.4 * self.k_c, 1.25 * self.k_cp
        if th > 25:
            return self.k_c, self.k_cp
        # small: hold & fine align
        return 1.3 * self.k_c, 0.9 * self.k_cp

    def _apply_poor_geometry_gate(
        self, sin_gamma: float, on: float = 0.25, off: float = 0.35
    ) -> bool:
        """
        Gate when target ~ parallel to B (sin_gamma small).
        on: enter when sin_gamma < on; off: exit when sin_gamma > off.
        """
        if self._poor_geom_gate:
            if sin_gamma > off:
                self._poor_geom_gate = False
        else:
            if sin_gamma < on:
                self._poor_geom_gate = True
        return self._poor_geom_gate
