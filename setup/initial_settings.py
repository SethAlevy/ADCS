import json
import datetime
from pathlib import Path
import numpy as np
from typing import Any
from templates.initial_settings_template import SimulationSetup


class SimulationSetupReader(SimulationSetup):
    def __init__(self, setup_file: str) -> None:
        self._setup = self._read_initial_parameters(setup_file)

    def _read_initial_parameters(self, setup_file: str) -> Any:
        """
        Reads the initial parameters from setup/initial_parameters.json file.
        """
        if not Path(setup_file).exists():
            raise RuntimeError(f"Setup file not found at: {setup_file}")
        with open(Path(setup_file), "r") as f:
            return json.load(f)

    @property
    def euler_angles(self) -> np.ndarray:
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
        euler_angles = self._setup["InitialState"][0]["eulerAngles"]
        phi = euler_angles[0]
        theta = euler_angles[1]
        psi = euler_angles[2]

        return np.array([phi, theta, psi])

    @property
    def angular_velocity(self) -> np.ndarray:
        """
        Initial angular velocity (q, r, p) in rad/s.

        returns:
            float: q - velocity around y-axis.
            float: r - velocity around z-axis.
            float: p - velocity around x-axis.
        """
        omega = self._setup["InitialState"][0]["angularVelocity"]
        p = omega[0]
        q = omega[1]
        r = omega[2]

        return np.array([p, q, r])

    @property
    def iterations_info(self) -> tuple[int, int, int]:
        """
        Simulation time parameters (t0, t_end, t_step) in seconds.

        returns:
            int: t0 - start time
            int: t_end - end time
            int: t_step - time step
        """
        data_dict = dict()
        data_dict['start'] = self._setup["Iterations"][0]["start"]
        data_dict['stop'] = self._setup["Iterations"][0]["stop"]
        data_dict['stop'] = self._setup["Iterations"][0]["step"]

        return data_dict

    @property
    def magnetorquer_params(self) -> tuple[int, int, float]:
        """
        Magnetorquer parameters (n, A, maxI), works for every axis of rotation.

        returns:
            int: n_coils - number of coils.
            int: coil_area - area of each coil in m^2.
            float: max_current - maximum current in A.
        """
        n_coils = self._setup["Magnetorquer"][0]["n"]
        coil_area = self._setup["Magnetorquer"][0]["A"]
        max_current = self._setup["Magnetorquer"][0]["maxI"]

        return n_coils, coil_area, max_current

    @property
    def satellite_params(self) -> tuple[int, np.ndarray]:
        """
        Satellite parameters (I, m).

        returns:
            int: mass of the satellite in kg.
            np.ndarray: inertia matrix in kg*m^2.
        """
        inertia = self._setup["SatelliteParams"][0]["I"]
        mass = self._setup["SatelliteParams"][0]["mass"]

        return inertia, mass

    @property
    def planet_data(self) -> dict[float, float, float]:
        """
        Parameters and constants describing the planet (G, M, R).

        returns:
            float: G - gravitational constant in m^3/(kg*s^2).
            float: M - mass of the planet in kg.
            float: R - radius of the planet in m.
        """
        data_dict = dict()
        data_dict['G'] = self._setup["PlanetConst"][0]["G"]
        data_dict['M'] = self._setup["PlanetConst"][0]["M"]
        data_dict['R'] = self._setup["PlanetConst"][0]["R"]

        return data_dict

    @property
    def date_time(self) -> datetime.datetime:
        """
        Date and time of the simulation start.

        returns:
            datetime: date_time - date and time of the simulation start.
        """
        return (
            datetime.datetime.now()
            if bool(self._setup["Date"][0]["now"])
            else datetime.datetime(
                self._setup["Date"][0]["year"],
                self._setup["Date"][0]["month"],
                self._setup["Date"][0]["day"],
                self._setup["Date"][0]["hour"],
                self._setup["Date"][0]["minute"],
                self._setup["Date"][0]["second"],
            )
        )
