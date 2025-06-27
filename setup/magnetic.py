import ppigrf as igrf
import numpy as np
import datetime as dt
import skyfield.api as skyfield
import setup.transformations as tr
from templates.satellite_template import Satellite


class MagnetometerImplementation:
    def __init__(
        self,
        noise: bool = False,
        noise_min: float = 0.1,
        noise_max: float = 0.1,
    ):
        """
        Initialize the Magnetometer class.

        Args:
            noise (bool, optional): If True, adds noise to the magnetic field vector.
            Defaults to False.
            noise_min (float, optional): Minimum noise factor to apply. Defaults to 0.1.
            noise_max (float, optional): Maximum noise factor to apply. Defaults to 0.1.
        """
        self.noise = noise
        self.noise_min = noise_min
        self.noise_max = noise_max

    def get_magnetic_field(
        self, satellite: Satellite, date: dt.datetime
    ) -> np.ndarray:
        """
        Get the magnetic field vector at a given satellite and date.
        This method uses the ppigrf library to compute the magnetic field
        vector in the East-North-Up (ENU) reference frame in nT.

        Args:
            satellite (Satellite): The satellite object containing
        the TLE data and current status.
            date (dt.datetime): time object representing the date and time
            for which the magnetic field vector is to be computed.

        Returns:
            np.ndarray: vector containing the magnetic field components
            in the East-North-Up (ENU) reference frame.
        """

        be, bn, bu = igrf.igrf(
            satellite.latitude, satellite.longitude, satellite.altitude, date
        )

        return np.array([be[0], bn[0], bu[0]])

    def simulate_magnetometer(
        self,
        satellite: object,
        julian_date: skyfield.Time,
    ) -> np.ndarray:
        """
        Simulate the magnetometer readings. This method computes the
        magnetic field vector at a given satellite and date, optionally adds
        noise and transforms it to the Satellite Body Frame (SBF) and Earth
        Centered Inertial Frame (ECI).

        Args:
            satellite (object): The satellite object containing the TLE data and current status.
            julian_date (skyfield.TIme): Julian date for which the magnetic field vector is to be computed.

        Returns:
            np.ndarray: Simulated magnetic field vectors in the Satellite Body Frame (SBF) and Earth-Centered Inertial (ECI) frame.
        The first three elements are in the SBF frame, and the next three are in the ECI frame.
        """
        date_time = dt.datetime.fromtimestamp(julian_date.tt)
        mag_field_enu = self.get_magnetic_field(satellite, date_time)
        mag_field_ecef = tr.enu_to_ecef(
            mag_field_enu, satellite.latitude, satellite.longitude
        )
        mag_field_eci = tr.ecef_to_eci(mag_field_ecef, julian_date)
        mag_field_sbf = tr.eci_to_sbf(
            mag_field_eci,
            tr.euler_xyz_to_quaternion(
                satellite.euler_angles[0],
                satellite.euler_angles[1],
                satellite.euler_angles[2],
            ),
        )

        if self.noise:
            noise_vector = np.random.uniform(
                1 - self.noise_min, 1 + self.noise_max, 3
            )
            mag_field_eci *= noise_vector
            mag_field_sbf *= noise_vector

        return mag_field_sbf, mag_field_eci
