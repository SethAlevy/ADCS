import numpy as np
import skyfield.api as skyfield

from spacecraft.sensors import MagnetometerImplementation, SunsensorImplementation


class FakeSetup:
    magnetometer = {
        "Noise": False,
        "AbsoluteNoise": 0.0
    }
    sunsensor = {
        "Noise": False,
        "AngularNoise": 0.0
    }


class FakeSatellite:
    def __init__(self):
        # Identity quaternion (no rotation between frames)
        self.quaternion = np.array([0.0, 0.0, 0.0, 1.0])

        self.latitude = 0.0
        self.longitude = 0.0
        self.altitude = 425.0  # km

    @property
    def magnetic_field(self):
        # Return tuple (sbf, eci) as expected by simulation code
        return self._mag_sbf, self._mag_eci

    @property
    def sun_vector(self):
        # Return tuple (sbf, eci)
        return self._sun_sbf, self._sun_eci


def test_simulate_magnetometer_identity_no_noise():
    setup = FakeSetup()
    sat = FakeSatellite()
    sensor = MagnetometerImplementation(setup)

    ts = skyfield.load.timescale()
    jd = ts.utc(2025, 8, 1)

    b_sbf, b_eci = sensor.simulate_magnetometer(sat, julian_date=jd)

    # 1. Identity quaternion → vectors equal
    assert np.allclose(b_sbf, b_eci, atol=1e-9)
    # 2. Components within plausible Earth field range
    assert b_sbf.shape == (3,)
    assert np.all(np.abs(b_sbf) < 70000.0)


def test_simulate_sunsensor_identity_no_noise():
    setup = FakeSetup()
    sat = FakeSatellite()
    sensor = SunsensorImplementation(setup)

    ts = skyfield.load.timescale()
    jd = ts.utc(2025, 8, 1)

    sun_sbf, sun_eci = sensor.simulate_sunsensor(sat, julian_date=jd)

    # 1. Identity quaternion → vectors equal
    assert np.allclose(sun_sbf, sun_eci, atol=1e-9)
    # 2. Unit length
    assert np.isclose(np.linalg.norm(sun_sbf), 1.0, atol=1e-9)
    # Components in [-1, 1]
    assert np.all(np.abs(sun_sbf) <= 1.0 + 1e-12)
