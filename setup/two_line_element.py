from abc import ABC
from abc import abstractmethod
from pathlib import Path


class TwoLineElement(ABC):
    """
    Abstract class for two-line elements (TLE).
    """

    @property
    @abstractmethod
    def line_1(self) -> str:
        """
        First line of the TLE.

        Returns:
            str: First line of the TLE.
        """
        pass

    @property
    @abstractmethod
    def line_2(self) -> str:
        """
        Second line of the TLE.

        Returns:
            str: Second line of the TLE.
        """
        pass

    @property
    @abstractmethod
    def satellite_name(self) -> int:
        """
        Name of the satellite (Satellite Catalog Number) is a 9-digit number
        assigned to each satellite by the United States Space Command (USSC).

        Returns:
            int: 9-digit number.
        """
        pass

    @property
    @abstractmethod
    def classification(self) -> str:
        """
        Classification of the satellite (e.g., "U" for unclassified).

        Returns:
            str: Classification of the satellite.
        """
        pass

    @property
    @abstractmethod
    def launch_year(self) -> int:
        """
        Launch year of the satellite.

        Returns:
            int: Two last digits of the launch year.
        """
        pass

    @property
    @abstractmethod
    def launch_number(self) -> int:
        """
        Launch number of the satellite in the given year.

        Returns:
            int: 3-digit launch number of the satellite.
        """
        pass

    @property
    @abstractmethod
    def piece_launch(self) -> str:
        """
        Piece of the launch.

        Returns:
            str: 3-digit piece of the launch.
        """
        pass

    @property
    @abstractmethod
    def epoch_year(self) -> int:
        """
        Epoch year of the TLE, representing the time at which the orbital
        elements are valid given as the year.

        Returns:
            int: Two last digits of the year. May skip the first 0.
        """
        pass

    @property
    @abstractmethod
    def epoch_day(self) -> float:
        """
        Epoch day of the TLE, representing the time at which the orbital
        elements are valid given as the day of the year.

        Returns:
            float: Day of the year.
        """
        pass

    @property
    @abstractmethod
    def mean_motion_derivative_1(self) -> float:
        """
        First derivative of the mean motion also known as the ballistic
        coefficient.

        Returns:
            float: First derivative of the mean motion.
        """
        pass

    @property
    @abstractmethod
    def mean_motion_derivative_2(self) -> float:
        """
        Second derivative of the mean motion.

        Returns:
            float: Second derivative of the mean motion.
        """
        pass

    @property
    @abstractmethod
    def bstar_drag(self) -> float:
        """
        BSTAR is a way of modeling satellite aerodynamic drag used in the
        SGP4 model. The drag term is also known as the radiation pressure
        coefficient.

        Returns:
            float: BSTAR drag term.
        """
        pass

    @property
    @abstractmethod
    def ephemeris_type(self) -> int:
        """
        Ephemeris type of the TLE.

        Returns:
            int: Usually 0.
        """
        pass

    @property
    @abstractmethod
    def element_number(self) -> int:
        """
        Element set number of the TLE.

        Returns:
            int: 3-digit number.
        """
        pass

    @property
    @abstractmethod
    def checksum_line1(self) -> int:
        """
        Checksum of the first line of the TLE.

        Returns:
            int: Checksum of the first line.
        """
        pass

    @property
    @abstractmethod
    def inclination(self) -> float:
        """
        Inclination of the satellite orbit in degrees. Angle between the
        orbital plane and the reference plane.

        Returns:
            float: Positive value in degrees.
        """
        pass

    @property
    @abstractmethod
    def raan(self) -> float:
        """
        Right Ascension of Ascending Node (RAAN) in degrees. The angle
        between the reference direction and the ascending node of the orbit.

        Returns:
            float: Angle from 0 to 360 degrees.
        """
        pass

    @property
    @abstractmethod
    def eccentricity(self) -> float:
        """
        Eccentricity of the satellite orbit. A measure of how much the orbit
        deviates from a perfect circle. A value of 0 indicates a circular
        orbit, while values close to 1 indicate an elongated orbit.

        Returns:
            float: Eccentricity value between 0 and 1.
        """
        pass

    @property
    @abstractmethod
    def argument_of_perigee(self) -> float:
        """
        Argument of perigee in degrees. The angle between the ascending node
        and the point of closest approach to the Earth.

        Returns:
            float: Angle from 0 to 360 degrees.
        """
        pass

    @property
    @abstractmethod
    def mean_anomaly(self) -> float:
        """
        Mean anomaly in degrees. The angle between the perigee and the
        current position of the satellite.

        Returns:
            float: Angle from 0 to 360 degrees.
        """
        pass

    @property
    @abstractmethod
    def mean_motion(self) -> float:
        """
        Mean motion in revolutions per day. The number of orbits the
        satellite completes in one day.

        Returns:
            float: Number of revolutions per day.
        """
        pass

    @property
    @abstractmethod
    def revolution_number(self) -> int:
        """
        Revolution number at epoch. The number of orbits the satellite has
        completed since launch.

        Returns:
            int: Revolution number.
        """
        pass

    @property
    @abstractmethod
    def checksum_line2(self) -> int:
        """
        Checksum of the second line of the TLE.

        Returns:
            int: Checksum of the second line.
        """
        pass


class TwoLineElementReader(TwoLineElement):
    """
    Class for reading two-line elements (TLE) from a file.
    """

    def __init__(self, tle_txt: Path):
        """
        Initialize the TLE reader with the given filename.

        Args:
            filename (Path): Path to the TLE txt file.
        """
        self.tle = tle_txt

    def read_tle(self) -> list[str]:
        """
        Read the TLE from the file.

        Returns:
            str: TLE string.
        """
        if not self.tle.exists():
            raise FileNotFoundError(f"TLE file {self.tle} not found.")
        with open(self.tle, "r") as f:
            lines = f.readlines()
        return lines

    # TODO add checksum validation
    @property
    def line_1(self) -> str:
        """
        First line of the TLE.

        Returns:
            str: First line of the TLE.
        """
        return self.read_tle()[0]

    @property
    def line_2(self) -> str:
        """
        Second line of the TLE.

        Returns:
            str: Second line of the TLE.
        """
        return self.read_tle()[1]

    @property
    def satellite_name(self) -> str:
        """
        Name of the satellite (Satellite Catalog Number).

        Returns:
            str: 9-digit number.
        """
        return self.line_1[2:7]

    @property
    def classification(self) -> str:
        """
        Classification of the satellite (e.g., "U" for unclassified).

        Returns:
            str: Classification of the satellite.
        """
        return self.line_1[7]

    @property
    def launch_year(self) -> int:
        """
        Launch year of the satellite.

        Returns:
            int: Two last digits of the launch year.
        """
        return int(self.line_1[9:11])

    @property
    def launch_number(self) -> int:
        """
        Launch number of the satellite in the given year.

        Returns:
            int: 3-digit launch number of the satellite.
        """
        return int(self.line_1[11:14])

    @property
    def piece_launch(self) -> str:
        """
        Piece of the launch.

        Returns:
            str: 3-digit piece of the launch.
        """
        return self.line_1[14:17]

    @property
    def epoch_year(self) -> int:
        """
        Epoch year of the TLE, representing the time at which the orbital
        elements are valid given as the year.

        Returns:
            int: Two last digits of the year. May skip the first 0.
        """
        return int(self.line_1[18:20])

    @property
    def epoch_day(self) -> float:
        """
        Epoch day of the TLE, representing the time at which the orbital
        elements are valid given as the day of the year.

        Returns:
            float: Day of the year.
        """
        return float(self.line_1[20:32])

    @property
    def mean_motion_derivative_1(self) -> float:
        """
        First derivative of the mean motion also known as the ballistic
        coefficient.

        Returns:
            float: First derivative of the mean motion.
        """
        return float(self.line_1[33:43])

    @property
    def mean_motion_derivative_2(self) -> float:
        """
        Second derivative of the mean motion.

        Returns:
            float: Second derivative of the mean motion.
        """
        return float(self.line_1[44:52])

    @property
    def bstar_drag(self) -> float:
        """
        BSTAR is a way of modeling satellite aerodynamic drag used in the
        SGP4 model. The drag term is also known as the radiation pressure
        coefficient.

        Returns:
            float: BSTAR drag term.
        """
        # TODO add conversion to float
        return str(self.line_1[53:61])

    @property
    def ephemeris_type(self) -> int:
        """
        Ephemeris type of the TLE.

        Returns:
            int: Usually 0.
        """
        return int(self.line_1[62])

    @property
    def element_number(self) -> int:
        """
        Element set number of the TLE.

        Returns:
            int: 3-digit number.
        """
        return int(self.line_1[64:68])

    @property
    def checksum_line1(self) -> int:
        """
        Checksum of the first line of the TLE.

        Returns:
            int: Checksum of the first line.
        """
        return int(self.line_1[68])

    @property
    def inclination(self) -> float:
        """
        Inclination of the satellite orbit in degrees. Angle between the
        orbital plane and the reference plane.

        Returns:
            float: Positive value in degrees.
        """
        return float(self.line_2[8:16])

    @property
    def raan(self) -> float:
        """
        Right Ascension of Ascending Node (RAAN) in degrees. The angle
        between the reference direction and the ascending node of the orbit.

        Returns:
            float: Angle from 0 to 360 degrees.
        """
        return float(self.line_2[17:25])

    @property
    def eccentricity(self) -> float:
        """
        Eccentricity of the satellite orbit. A measure of how much the orbit
        deviates from a perfect circle. A value of 0 indicates a circular
        orbit, while values close to 1 indicate an elongated orbit.

        Returns:
            float: Eccentricity value between 0 and 1.
        """
        return float(f"0.{self.line_2[26:33]}")

    @property
    def argument_of_perigee(self) -> float:
        """
        Argument of perigee in degrees. The angle between the ascending node
        and the point of closest approach to the Earth.

        Returns:
            float: Angle from 0 to 360 degrees.
        """
        return float(self.line_2[34:43])

    @property
    def mean_anomaly(self) -> float:
        """
        Mean anomaly in degrees. The angle between the perigee and the
        current position of the satellite.

        Returns:
            float: Angle from 0 to 360 degrees.
        """
        return float(self.line_2[43:51])

    @property
    def mean_motion(self) -> float:
        """
        Mean motion in revolutions per day. The number of orbits the
        satellite completes in one day.

        Returns:
            float: Number of revolutions per day.
        """
        return float(self.line_2[52:63])

    @property
    def revolution_number(self) -> int:
        """
        Revolution number at epoch. The number of orbits the satellite has
        completed since launch.

        Returns:
            int: Revolution number.
        """
        return int(self.line_2[63:68])

    @property
    def checksum_line2(self) -> int:
        """
        Checksum of the second line of the TLE.

        Returns:
            int: Checksum of the second line.
        """
        return int(self.line_2[68])
