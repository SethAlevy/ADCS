from templates.satellite_template import Satellite
import numpy as np
from setup.two_line_element import TwoLineElement
from setup.initial_settings import SimulationSetup
from templates.magnetic_template import Magnetometer
import skyfield.api as skyfield
import setup.utilities as ut
import setup.transformations as tr


class SatelliteImplementation(Satellite):
    """
    Implementation of the Satellite class.
    """

    def __init__(
        self,
        setup: SimulationSetup,
        tle: TwoLineElement,
        magnetometer: Magnetometer,
    ):
        """
        Initialize the satellite using json file and TLE file.
        """
        self.setup = setup
        self._angular_velocity = self.setup.angular_velocity
        self._euler_angles = self.setup.euler_angles
        self._iteration = self.setup.iterations_info["start"]

        self._two_line_element = tle
        self.magnetometer = magnetometer

        self._satellite_model = skyfield.EarthSatellite(
            self.two_line_element.line_1, self.two_line_element.line_2
        )

    def update_iteration(self, iteration: int) -> None:
        """
        Update the current iteration of the simulation.

        Args:
            iteration (int): The current iteration of the simulation.
            Equals the time in seconds from its start.
        """
        self._iteration = iteration

    @property
    def iteration(self) -> int:
        """
        Current iteration of the simulation. Equals the time in seconds
        from its start.
        """
        return self._iteration

    @property
    def mass(self) -> float:
        return self.setup.satellite_params[0]

    @property
    def inertia_matrix(self) -> np.ndarray:
        return self.setup.satellite_params[1]

    @property
    def position(self) -> np.ndarray:
        """
        Position of the satellite obtained using the skyfield library.
        By default returns GCRS (Geocentric Celestial Reference System)
        which is an ECI (Earth-Centered Inertial) frame (fixed to the stars)
        almost similar to J2000 frame. Distance is given in km and calculated
        for the given simulation time.

        Returns:
            np.ndarray: X, Y and Z position of the satellite in km.
        """
        julian_date = ut.time_julian_date(self)
        return self._satellite_model.at(julian_date).position.km

    @property
    def linear_velocity(self) -> np.ndarray:
        """
        Linear velocity of the satellite obtained using the skyfield library.
        By default returns GCRS (Geocentric Celestial Reference System)
        which is an ECI (Earth-Centered Inertial) frame (fixed to the stars)
        almost similar to J2000 frame. Distance is given in km and calculated
        for the given simulation time.

        Returns:
            np.ndarray: X, Y and Z velocity of the satellite in km/s.
        """
        julian_date = ut.time_julian_date(self)
        return self._satellite_model.at(julian_date).velocity.km_per_s

    @property
    def latitude(self) -> np.ndarray:
        """
        Latitude of the satellite in degrees calculated for the
        given simulation time and using WGS84 (World Geodetic System 1984).

        Returns:
            np.ndarray: Latitude of the satellite in degrees.
        """
        julian_date = ut.time_julian_date(self)
        latlon = skyfield.wgs84.latlon_of(self._satellite_model.at(julian_date))
        return latlon[0].degrees

    @property
    def longitude(self) -> np.ndarray:
        """
        Longitude of the satellite in degrees calculated for the
        given simulation time and using WGS84 (World Geodetic System 1984).

        Returns:
            np.ndarray: Longitude of the satellite in degrees.
        """
        julian_date = ut.time_julian_date(self)
        return skyfield.wgs84.latlon_of(self._satellite_model.at(julian_date))[
            1
        ].degrees

    @property
    def altitude(self) -> np.ndarray:
        """
        Altitude of the satellite in km calculated for the
        given simulation time and using WGS84 (World Geodetic System 1984).

        Returns:
            np.ndarray: Altitude of the satellite in km.
        """
        julian_date = ut.time_julian_date(self)
        geocentric = self._satellite_model.at(julian_date)
        # .subpoint().elevation gives altitude in meters; convert to km
        return geocentric.subpoint().elevation.m / 1000.0

    @property
    def angular_velocity(self) -> np.ndarray:
        """
        Angular velocity of the satellite in degrees/s. According to the
        aerospace convention, the angular velocity is given in the order
        wx, wy, wz (roll, pitch, yaw)

        Returns:
            np.ndarray: Angular velocity of the satellite in degrees/s.
        """
        new_velocity = self._angular_velocity
        self._angular_velocity = new_velocity
        return new_velocity

    @property
    def euler_angles(self) -> np.ndarray:
        """
        Euler angles of the satellite in degrees. The angles are updated
        by adding the angular velocity to the current angles. Assumption is
        that the time step is equal to 1 second. Used convention is X-Y-Z
        (known as roll-pitch-yaw).

        Args:
            angular_velocity (np.ndarray): Angular velocity of the satellite
            in degrees/second. Choosen convention is to keep the angles
            in [-180, 180).

        Returns:
            np.ndarray: Updated Euler angles of the satellite in degrees.
        """
        new_euler = self._euler_angles + self.angular_velocity
        # Keep first and third angles in [-180, 180)
        new_euler[0] = ((new_euler[0] + 180) % 360) - 180
        new_euler[2] = ((new_euler[2] + 180) % 360) - 180
        # Keep the second angle in [-90, 90]
        new_euler[1] = np.clip(new_euler[1], -90, 90)
        self._euler_angles = new_euler
        return new_euler

    @property
    def quaternion(self) -> np.ndarray:
        """
        Convert the Euler angles to a quaternion.

        Returns:
            np.ndarray: Quaternion of the satellite.
        """
        return tr.euler_xyz_to_quaternion(
            self.euler_angles[0], self.euler_angles[1], self.euler_angles[2]
        )

    @property
    def two_line_element(self) -> TwoLineElement:
        return self._two_line_element

    @property
    def magnetic_field(self) -> np.ndarray:
        """
        Get the magnetic field vector at the satellite's position in the SBF and ECI frames.
        The second is rather for debugging purposes. Both are in nT (nanoTesla).

        Returns:
            np.ndarray: Magnetic field vector in the SBF and ECI frames.
        """
        julian_date = ut.time_julian_date(self)
        return self.magnetometer.simulate_magnetometer(self, julian_date)
