
import json
import numpy as np
from typing import Any
from pathlib import Path
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
        Initial Euler angles phi, theta, psi. Extrinsic rotation conventions
        is used: Z-X-Z (or 3-1-3) convention, where the first rotation is
        around the Z-axis, the second rotation is around the X-axis, and the
        third rotation is around the Z-axis again. This rotation is supported
        by scipy rotation library. The angles are in degrees.

        returns:
            float: phi -180 to 180 degrees.
            float: theta -90 to 90 degrees.
            float: psi -180 to 180 degrees.
        """
        pass

    @property
    @abstractmethod
    def angular_velocity(self) -> tuple[float, float, float]:
        """
        Initial angular velocity (p, q, r) in rad/s.

        returns:
            float: p - velocity around x-axis.
            float: q - velocity around y-axis.
            float: r - velocity around z-axis.
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
    def euler_angles(self) -> tuple[float, float, float]:
        """
        Initial Euler angles phi, theta, psi. Extrinsic rotation conventions
        is used: Z-X-Z (or 3-1-3) convention, where the first rotation is
        around the Z-axis, the second rotation is around the X-axis, and the
        third rotation is around the Z-axis again. This rotation is supported
        by scipy rotator library. The angles are in degrees.

        returns:
            float: phi -180 to 180 degrees.
            float: theta -90 to 90 degrees.
            float: psi -180 to 180 degrees.
        """
        euler_angles = self._setup["InitialState"][0]["eulerAngles"]
        phi = euler_angles[0]
        theta = euler_angles[1]
        psi = euler_angles[2]

        return phi, theta, psi

    @property
    def angular_velocity(self) -> tuple[float, float, float]:
        """
        Initial angular velocity (p, q, r) in rad/s.

        returns:
            float: p - velocity around x-axis.
            float: q - velocity around y-axis.
            float: r - velocity around z-axis.
        """
        omega = self._setup["InitialState"][0]["angularVelocity"]
        p = omega[0]
        q = omega[1]
        r = omega[2]

        return p, q, r

    @property
    def iterations_info(self) -> tuple[int, int, int]:
        """
        Simulation time parameters (t0, t_end, t_step) in seconds.

        returns:
            int: t0 - start time
            int: t_end - end time
            int: t_step - time step
        """
        t0 = self._setup["Iterations"][0]["start"]
        tend = self._setup["Iterations"][0]["stop"]
        tstep = self._setup["Iterations"][0]["step"]

        return t0, tend, tstep

    @property
    def magnetorquer_params(self) -> tuple[int, int]:
        """
        Magnetorquer parameters (n, A), works for every axis of rotation.

        returns:
            int: n_coils - number of coils.
            int: coil_area - area of each coil in m^2.
        """
        n_coils = self._setup["Magnetorquer"][0]["n"]
        coil_area = self._setup["Magnetorquer"][0]["A"]

        return n_coils, coil_area

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
    def planet_data(self) -> tuple[float, float, float]:
        """
        Parameters and constants describing the planet (G, M, R).

        returns:
            float: G - gravitational constant in m^3/(kg*s^2).
            float: M - mass of the planet in kg.
            float: R - radius of the planet in m.
        """
        G = self._setup["PlanetConst"][0]["G"]
        M = self._setup["PlanetConst"][0]["M"]
        R = self._setup["PlanetConst"][0]["R"]

        return G, M, R

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
