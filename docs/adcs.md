# Introduction

The following code performs an ADCS (Attitude Determination and Control System) simulation for a small satellite (CubeSat). It recreates the environment on a given orbit, the object’s parameters, and determines its attitude and dynamics (angular velocity and angular acceleration). The main tasks are detumbling (B-dot) and pointing (B-cross). A magnetometer, Sun sensor, and the TRIAD, QUEST, and EKF algorithms have been implemented. Control is executed using three electromagnetic coils.

The simulation consists of three main elements:
 - Reconstructing orbital and environmental conditions, including Earth’s gravitational field, atmospheric drag, magnetic field, and Sun position.
 - Determining the satellite’s attitude and dynamics based on measurements from sensors (magnetometer, Sun sensor) and data fusion algorithms (TRIAD, QUEST, EKF).
 - Controlling the satellite to carry out detumbling (reducing rotation) and pointing (orienting toward Earth or the Sun) using electromagnetic coils.

# Preparation

## Installation

All code is written in Python and uses dedicated libraries both for mathematical operations and for implementing basic astronomical models. The remainder assumes a working Python installation (preferred 3.13). The environment is managed with Poetry. A concise installation guide is available in its documentation (https://python-poetry.org/docs/). One of the key project files for Poetry is pyproject.toml in the repository root. It defines required libraries and their versions. To ensure compatibility and proper operation, versions are pinned.

To build the environment execute:

```powershell
poetry install
```

Ways to use the environment:
- Invoking poetry directly (recommended):

```powershell
poetry run python adcs.py
```

- Activating the environment in a terminal:

```powershell
poetry env activate  # returns a personalized activation command
# example activation (PowerShell):
& "C:\path\to\env\Scripts\Activate.ps1"
python adcs.py
deactivate
```

- Selecting the created environment in the code editor. This step is required for using Jupyter Notebook.

## Project Structure

```text
ADCS/
├─ adcs.py
├─ mkdocs.yml
├─ pyproject.toml
├─ docs/
│  ├─ index.md
│  ├─ adcs.md
│  └─ adcs_pl.md
├─ core/
│  ├─ logger.py
│  ├─ state.py
│  ├─ transformations.py
│  └─ utilities.py
├─ spacecraft/
│  ├─ actuator.py
│  ├─ sensors.py
│  └─ satellite.py
├─ setup/
│  ├─ initial_settings.py
│  ├─ initial_settings.json
│  ├─ two_line_element.py
│  └─ tle
├─ templates/
│  ├─ actuator_template.py
│  ├─ sensor_template.py
│  ├─ satellite_template.py
│  ├─ two_line_element_template.py
│  └─ initial_settings_template.py
├─ visualizations/
│  └─ visualizations.py
└─ tests/
   ├─ conftest.py
   ├─ test_actuator.py
   ├─ test_sensors.py
   └─ test_transformations.py
```

The backbone of the project is the spacecraft folder, which contains the core code responsible for satellite systems.
- The satellite.py script handles general management and coordination of individual elements, integrating everything into a single object through which key satellite parameters can be accessed.
- The sensors.py script contains all code related to measurements: environment simulation, adding noise, and processing via algorithms (TRIAD, QUEST, EKF).
- The actuator.py script handles active elements (in this case only electromagnetic coils). Its role is implementing control algorithms (detumbling and pointing) and managing operation to achieve the expected effect.

The core folder contains helper functions, mainly related to mathematical operations and broadly defined spatial attitude handling and its processing.
- logger.py contains the configured logging module.
- state.py manages the state vector.
- transformations.py focuses on functions related to attitude, such as frame conversions, quaternion operations, and rotations.
- utilities.py contains all remaining helpful functions like date format conversion.

The setup folder gathers information related to simulation settings, input data, and scripts responsible for processing them.
- The tle file stores the Two-Line Element Set describing orbital parameters.
- two_line_element.py loads it and extracts parameters conveniently.
- initial_settings.json stores parameters and settings describing initial state, constants, satellite parameters, and modes. Most variables affecting simulation flow can be managed from it.
- initial_settings.py loads and manages settings from initial_settings.json.

The visualizations folder is related to plots that allow checking simulation progress.
- visualizations.py contains code responsible for creating plots.

The templates folder defines files that form skeletons describing basic contents of key repository elements. Their parameters are required by implementation scripts.

## First Run

Execution can occur in two ways. After building the environment either run adcs.py, which contains an integrated simulation, or use a Jupyter Notebook. With default settings the code will run and should produce normal logs:

```
2025-11-14 23:42:20 | INFO | Simulation initialized with the following parameters:
2025-11-14 23:42:20 | INFO | Number of iterations: 10000
2025-11-14 23:42:20 | INFO | Satellite mass: 1.2 kg
2025-11-14 23:42:20 | INFO | Satellite inertia: [[0.002 0.    0.   ]
 [0.    0.002 0.   ]
 [0.    0.    0.002]] kg*m^2
2025-11-14 23:42:20 | INFO | Initial angular velocity: [ 2. -3.  4.] deg/s
2025-11-14 23:42:20 | INFO | Initial attitude (Euler angles): [0. 0. 0.] deg
2025-11-14 23:42:20 | INFO | Selected sensor fusion algorithm: EKF
2025-11-14 23:42:20 | INFO | Magnetometer noise: 10.0 nT
2025-11-14 23:42:20 | INFO | Sunsensor noise: 0.2 deg
2025-11-14 23:42:20 | INFO | Sensor on time: 2 seconds, actuator on time: 8 seconds

2025-11-14 23:42:20 | INFO | Iteration 0 of 10000
2025-11-14 23:42:25 | INFO | Iteration 100 of 10000
```

This form means the code runs correctly.

# Fundamentals

Below the key theoretical concepts are briefly presented with usage examples and references to their application in simulation code. Since some functions and methods were prepared specifically for this repository, correct functioning requires initialization of base objects and parameters. This can be done with the following code (in most cases placed at the beginning):

```python
from pathlib import Path
from spacecraft.satellite import SatelliteImplementation
from setup.initial_settings import SimulationSetupReader
from setup.two_line_element import TwoLineElementReader
from spacecraft.sensors import MagnetometerImplementation, SunsensorImplementation, SensorFusionImplementation

setup = SimulationSetupReader(Path('setup/initial_settings.json'))
tle = TwoLineElementReader(Path('setup/tle'))
magnetometer = MagnetometerImplementation(setup)
sunsensor = SunsensorImplementation(setup)
sensor_fusion = SensorFusionImplementation(setup, ['triad', 'quest', 'ekf'], tr.euler_xyz_to_quaternion(setup.euler_angles))

satellite = SatelliteImplementation(setup, tle, magnetometer, sunsensor, sensor_fusion)
```

## Orbit and Environment

### Julian Date (JD)

The Julian Date is a frequently used time representation in astronomical computations and algorithms. It is the fraction representing the number of days since noon January 1, 4713 BC. For example, for UTC 2025-10-18 00:00:00.000 JD equals 2460966.5. Sometimes the Modified Julian Date (MJD) format is encountered, intended to simplify the fraction: MJD = JD − 2400000.5. For the above example it is 60966. The code uses the standard JD applied in the Skyfield library. It is passed as an argument to the propagator model and allows obtaining orbital parameters. Conversion is handled by a function returning current satellite time (simulation start time plus iteration count). With default settings using current time as start:

```python
import core.utilities as ut

time_satellite = satellite.setup.date_time
satellite_julian_date = ut.time_julian_date(satellite)

print(f"Satellite time: {time_satellite}")
print(f"Satellite Julian date: {satellite_julian_date}")
```

Example output:

```text
Satellite time: 2025-10-22 13:25:41.386584
Satellite Julian date: <Time tt=2460818.709556771>
```

Note: Skyfield presents time in TT (Terrestrial Time). It differs from UTC by ΔT (leap seconds and drift). For IGRF this is negligible, but for precise conversions TT↔UTC should be considered.

### TLE

The most popular storage format for parameters of Earth-orbiting objects is the TLE (Two-Line Element Set), which consists of two lines where each character sequence has meaning. An example used in this repository (also under ADCS/setup/tle):

```
1 25544U 98067A 25143.20875603 .00008836 00000-0 16445-3 0 9994
2 25544 51.6382 70.8210 0002488 135.0606 10.4960 15.49676890511280
```

To load and extract data from TLE conveniently, code (two_line_element.py) was created. It allows retrieving individual elements by name or entire lines as strings. Meanings are briefly added in function descriptions. A good summary is available on Wikipedia (https://en.wikipedia.org/wiki/Two-line_element_set).

Initialize TLE to use it:

```python
print(f'Two Line Element:\n{tle.line_1}\n{tle.line_2}')

print(f"Epoch Year: {tle.epoch_year}")
print(f"Epoch Day: {tle.epoch_day}")

print(f"Inclination: {tle.inclination}")
print(f"Bstar Drag: {tle.bstar_drag}")
```

Output:

```text
Two Line Element:
1 25544U 98067A 25143.20875603 .00008836 00000-0 16445-3 0 9994
2 25544 51.6382 70.8210 0002488 135.0606 10.4960 15.49676890511280
Epoch Year: 25
Epoch Day: 143.20875603
Inclination: 51.6382
Bstar Drag: 16445-3
```

Note the format: Bstar Drag “16445-3” must be interpreted as 0.16445e-3 (1.6445e-4).

### Propagator

The orbital propagator determines satellite position, velocity, and acceleration at a given moment. This simulation uses SGP4 (Simplified General Perturbations 4). It uses TLE data and requires periodic updates to limit growing errors. Implementation comes from Skyfield.

```python
import skyfield.api as skyfield

satellite_model = skyfield.EarthSatellite(tle.line_1, tle.line_2)
julian_date = ut.time_julian_date(satellite)

position = satellite_model.at(julian_date).position.km
velocity = satellite_model.at(julian_date).velocity.km_per_s

print(f"Position vector: {position} [km]")
print(f"Velocity vector: {velocity} [km/s]")
```

Example output:

```text
Position vector: [-4107.48809952 -4489.58941621  3014.80595056] [km]
Velocity vector: [ 2.23875988 -5.3942253  -4.96501356] [km/s]
```

Position and velocity are primary results, but altitude above the WGS‑84 ellipsoid, geographic longitude and latitude can also be obtained. Skyfield results are in GCRS (inertial, aligned with ICRF at J2000); “ECI” is used as shorthand. If raw SGP4/TEME is used, transform to ECEF/ITRF (precession, nutation, sidereal time).

More: https://www.aero.iitb.ac.in/satelliteWiki/index.php/Orbit_Propagator

### Low Earth Orbit (LEO)

The region of space extending from Earth up to the Van Allen belts is referred to as Low Earth Orbit. In practice altitudes from about 200 km to 2000 km are considered. It is popular due to proximity and lower launch cost. Drawbacks include residual atmospheric drag (especially below ~300 km) and limited field of view. From an attitude determination and control perspective proximity to Earth offers another advantage: relatively strong magnetic field usable by sensors and actuators.

### Earth’s Magnetic Field

Earth naturally generates a magnetic field internally and around itself. Its shape approximates a dipole slightly tilted from the rotation axis. It has static and variable components (the latter typically 1–5%). Local anomalies exist. Thanks to observations accurate models such as IGRF (International Geomagnetic Reference Field) were developed. This simulation uses pyIGRF. It requires geographic coordinates and date as a decimal fraction (time from start of year expressed as a fraction, e.g. 2024.25 as first quarter).

```python
import pyIGRF
import core.utilities as ut
import datetime

lat = satellite.latitude
lon = satellite.longitude
alt_km = satellite.altitude
julian_date = ut.time_julian_date(satellite)

dt = julian_date.utc_datetime()
start = datetime.datetime(dt.year, 1, 1, tzinfo=dt.tzinfo)
end = datetime.datetime(dt.year + 1, 1, 1, tzinfo=dt.tzinfo)
decimal_year = dt.year + (dt - start).total_seconds() / (end - start).total_seconds()

# IGRF returns NED components in nT
d, i, h, b_n, b_e, b_d, f = pyIGRF.igrf_value(lat, lon, alt_km, decimal_year)

print(f"IGRF Magnetic Field NED: B_n={b_n:.2f}, B_e={b_e:.2f}, B_d={b_d:.2f} [nT]")
print(f"IGRF Magnetic Field Magnitude: F={f:.2f} [nT]")
print(f"IGRF Declination: D={d:.2f} [deg], Inclination: i={i:.2f} [deg], Horizontal Intensity: H={h:.2f} [nT]")
```

Output:

```text
IGRF Magnetic Field NED: B_n=20470.13, B_e=-760.77, B_d=29507.77 [nT]
IGRF Magnetic Field Magnitude: F=35920.94 [nT]
IGRF Declination: D=-2.13 [deg], Inclination: i=55.23 [deg], Horizontal Intensity: H=20484.26 [nT]
```

The library returns vector and resultant field value, declination, inclination, and horizontal intensity. For ADCS only the vector matters. Note pyIGRF returns NED (North-East-Down); transform before further work (see transformations chapter).

More: https://geomag.bgs.ac.uk/research/modelling/IGRF

### Sun Position

The Sun and its position are another element useful for attitude determination in space due to well-predictable position relative to Earth. Considering the enormous distance between both bodies, for simple calculations on low orbits Earth-centric description suffices, ignoring satellite orbital motion. A good step-by-step description can be found on Wikipedia: https://en.wikipedia.org/wiki/Position_of_the_Sun

Here Skyfield is used to compute Sun vector relative to Earth for a given Julian Date.

```python
import skyfield.api as skyfield
import core.utilities as ut

# load ephemeris data for sun and earth
eph = skyfield.load('de421.bsp')
sun = eph['sun']
earth = eph['earth']

julian_date = ut.time_julian_date(satellite)

sun_position_eci = earth.at(julian_date).observe(sun).position.km

print(f"Sun Position ECI: {sun_position_eci} [km]")
```

Output:

```text
Sun Position ECI: [7.11550861e+07 1.22678186e+08 5.31785009e+07] [km]
```

Returned Sun vector is in ICRF (treated as “ECI” for simplicity).

## Attitude and Transformations

### Reference Frames

To determine spatial attitude frames must be established. Due to diversity of objects and applications in space a range of different frames is used. Below several appearing in this repository are described.

- Satellite Body (SB) – frame rigidly attached to the spacecraft, moving and rotating with it. In this simplified simulation it is tied to sensors; measurements are expressed in it. In practice a navigation base may be defined and sensors may have local frames.
- Earth-Centered Inertial (ECI) – Earth-centered frame fixed relative to distant stars. Z axis toward North Pole, X along intersection of equator and ecliptic, Y completing orthonormal basis. Origin at Earth’s center of mass. Practically GCRS (ICRF-aligned) often used.
- Earth-Centered Earth-Fixed (ECEF) – rotating with Earth. Z along Earth’s rotation axis, X intersection of equator and 0° meridian, Y intersection of equator and 90°E meridian.
- East-North-Up / North-East-Down (ENU / NED) – two variants of a local frame near Earth. First two axes (X and Y) oriented along east or north direction; main difference lies in Z, pointing away from Earth (Up) or toward Earth (Down). ENU: x=East, y=North, z=Up. NED: x=North, y=East, z=Down. Appears for magnetic field (IGRF models return values in this frame).
- International Celestial Reference Frame (ICRF) – inertial frame with origin at Solar System barycenter; used for ephemerides. For near-Earth orbits treated as “ECI”.
- Longitude / Latitude / Altitude (LLA) – geodetic form relative to WGS‑84.

### Attitude Representation and Rotation

For an ADCS system on low Earth orbit the most important frames are ECI and SBF where the former is reference and the latter tied to onboard measurements. Transition between them consists of translation and rotation. Translation is provided by the propagator returning satellite position at a moment. Finding the rotation representing spatial attitude is ADCS task achieved using sensor measurements and comparing them with reference. Attitude can be represented several ways:

- Euler Angles – three successive rotations about chosen axes. Intuitive (popular e.g. in aviation) but limited for space applications (e.g. gimbal lock). In code used for initialization and visualization.
- Quaternion – q = [u_x sin(α/2), u_y sin(α/2), u_z sin(α/2), cos(α/2)]. Four-element rotation representation consisting of vector and scalar part. Vector part corresponds to rotation axis u, scalar part is cos(α/2). Quaternion is compact, convenient, numerically stable, and free of Euler angle singularities.
- Rotation Matrix – most basic form: a matrix that multiplies the original vector to obtain the target vector.

Rotation handling can be tricky and error-prone; to minimize risk the reliable scipy library is used for most operations. Below an example of a simple rotation using Euler angles and quaternion:

```python
import scipy.spatial.transform as R

euler_angles_init = [90.0, 0.0, 0.0]  # degrees
quaternion_init = [0.70710678, 0.0, 0.0, 0.70710678]  # [x, y, z, w]

reproduced_quaternion = R.Rotation.from_euler('xyz', euler_angles_init, degrees=True).as_quat()
reproduced_euler_angles = R.Rotation.from_quat(quaternion_init).as_euler('xyz', degrees=True)

print(f"Initial Euler Angles: {euler_angles_init} [deg]")
print(f"Initial Quaternion: {quaternion_init} [x, y, z, w]")
print(f"Reproduced Quaternion from Euler Angles: {reproduced_quaternion} [x, y, z, w]")
print(f"Reproduced Euler Angles from Quaternion: {reproduced_euler_angles} [deg]")

v1 = [0.0, 1.0, 0.0]

rotation = R.Rotation.from_quat(quaternion_init)
v2 = rotation.apply(v1)

print(f"Original Vector: {v1}")
print(f"Rotated Vector: {v2}")
```

Output:

```text
Initial Euler Angles: [90.0, 0.0, 0.0] [deg]
Initial Quaternion: [0.70710678 0.         0.         0.70710678] [x, y, z, w]
Reproduced Quaternion from Euler Angles: [0.70710678 0.         0.         0.70710678] [x, y, z, w]
Reproduced Euler Angles from Quaternion: [90.  0.  0.] [deg]
Original Vector: [0.0, 1.0, 0.0]
Rotated Vector: [0. 0. 1.]
```

Note: after each update the quaternion should be normalized to limit numerical drift.

Similarly functions for transitioning between frames were created (core/transformations.py). Below five basic implemented conversions:

```python
import core.transformations as tr
import core.utilities as ut

v1 = [0.0, 1.0, 0.0]
julian_date = ut.time_julian_date(satellite)
quaternion_init = [0.70710678, 0.0, 0.0, 0.70710678]

enu_ecef = tr.enu_to_ecef(v1, satellite.latitude, satellite.longitude)
ned_ecef = tr.ned_to_ecef(v1, satellite.latitude, satellite.longitude)
ecef_eci = tr.ecef_to_eci(v1, julian_date)
eci_sbf = tr.eci_to_sbf(v1, quaternion_init)
sbf_eci = tr.sbf_to_eci(v1, quaternion_init)

print(f"Original Vector: {v1}")
print(f"ENU to ECEF: {enu_ecef}")
print(f"NED to ECEF: {ned_ecef}")
print(f"ECEF to ECI: {ecef_eci}")
print(f"ECI to SBF: {eci_sbf}")
print(f"SBF to ECI: {sbf_eci}")
```

Output:

```text
Original Vector: [0.0, 1.0, 0.0]
ENU to ECEF: [ 0.02791803  0.44449952 -0.89534393]
NED to ECEF: [ 0.99961022 -0.01241439  0.02500598]
ECEF to ECI: [0.69189531 0.72199784 0.        ]
ECI to SBF: [0. 0. 1.]
SBF to ECI: [ 0.  0. -1.]
```

## Sensors and Active Elements

### Magnetometers

A magnetometer measures the magnetic field. There are many types of magnetometers with different capabilities and sizes depending on application and mission character. For small satellites on low Earth orbit three-axis magnetometers are used. Often they are fluxgate (core saturation) or magnetoresistive devices. Miniaturization yields small, light, low-power units.

In this simulation the magnetometer was chosen as the main sensor. Magnetometer simulation uses magnetic field from the IGRF model. To obtain a value equivalent to an actual onboard measurement, the vector must be transformed to the SBF frame. Noise can be added using a maximum absolute deviation.

```
mag_field_sbf += noise_vector
```

where noise_vector = <-x: +x>.

Noise settings are in initial_settings.json:

```
"Magnetometer":
{
  "Noise": true,
  "AbsoluteNoise": 10,
  "UnitInfo": "nT"
},
```

Noise model is simplified (uniform ±A per axis). Real sensors more often exhibit Gaussian distribution + bias + drift.

### Sun Sensor

Sun sensors enable determination of the Sun vector (direction of the Sun relative to the satellite). Depending on role they may be highly accurate array sensors or simpler multi-photodiode units on exterior panels. The latter is common on small satellites. Due to simplicity they complement other sensors well. In this case typically treated as second sensor.

Relative Sun position is obtained using Skyfield (as described earlier), then transformed to SB frame for an equivalent onboard measurement vector. It can then be noisified. Sun sensor noise is implemented as rotation of the vector around a random axis by a random angle in the specified range.

```
sun_vector = raw_sun_vector * R
```

where R is a rotation matrix representing rotation by an angle from <-x: +x>.

Noise settings in initial_settings.json:

```
"SunSensor":
{
  "Noise": true,
  "AngularNoise": 0.2,
  "UnitInfo": "deg"
},
```

### Gyroscopes

A gyroscope measures angular velocity directly (not angular acceleration). Classically it is a large rotating mechanical device using conservation of angular momentum. In space missions MEMS gyros are typically used—devices with small vibrating elements measuring rotation via Coriolis force. They are small and inexpensive, but have limited accuracy and accumulating drift. In this simulation angular velocity can be determined from attitude difference between iterations: divide by Δt and preferably use quaternion derivative (ω = 2 q̇ ⊗ q⁻¹). Drift is not modeled in detail.

```
"Gyroscope":
{
  "Bias": [0.0, 0.0, 0.0],
  "ProcessNoise": [1e-8, 1e-8, 1e-8],
  "UnitInfo": "deg/s"
},
```

### Sensor Fusion

A basic problem in attitude determination is combining measurements from two or more sources to obtain a single more reliable and accurate estimate. Using multiple types balances potential shortcomings when one would perform worse. Specialized algorithms process measurements. In space missions three are often distinguished:

- TRIAD (Three-Axis Attitude Determination) – simplest. Processes measurements from only two sensors knowing vectors in satellite frame and their inertial counterparts. Finds rotation matrix by constructing triads and relative arrangement.
- QUEST (QUaternion ESTimator) – solves Wahba’s problem for two or more vectors. Returns a quaternion (eigenvector of matrix built from measurement data). Optionally assigns weights to measurements.
- EKF (Extended Kalman Filter) – more advanced nonlinear state estimation. Iteratively uses dynamic model to predict future state and sensor measurements to correct it. For ADCS it uses measurement pairs plus angular velocity and previous attitude. EKF estimates should be noticeably more accurate than TRIAD and QUEST. Below an example obtaining quaternion for two vectors rotated by 90 degrees at zero angular velocity.

```python
import numpy as np

# two reference vectors
v1_i = np.array([1.0, 0.0, 0.0])
v2_i = np.array([0.70710678, 0.70710678, 0.0])

# two measured vectors with some noise rotated by 90 deg around z axis
v1_b = np.array([0.0, 1.0, 0.0]) + np.random.uniform(-0.02, 0.02, size=3)
v2_b = np.array([-0.70710678, 0.70710678, 0.0]) + np.random.uniform(-0.05, 0.05, size=3)

q_reference = np.array([0.0, 0.0, 0.70710678, 0.70710678])

# typically the first vector should be the most accurate one
v_i_list = [v1_i, v2_i]
v_b_list = [v1_b, v2_b]

# normalize before estimation (TRIAD/QUEST/EKF)
v_i_list = [v/np.linalg.norm(v) for v in v_i_list]
v_b_list = [v/np.linalg.norm(v) for v in v_b_list]

q_triad = sensor_fusion.triad(v_b_list, v_i_list)
q_quest = sensor_fusion.quest(v_b_list, v_i_list)
q_ekf = sensor_fusion.ekf(v_b_list, v_i_list, np.array([0.0, 0.0, 0.0, 0.0]), 1, q_reference)

# signs may differ between algorithms, but represent the same rotation q = -q
print(f"Reference quaternion: {q_reference} [x, y, z, w]")
print(f"TRIAD Quaternion: {q_triad} [x, y, z, w]")
print(f"QUEST Quaternion: {q_quest} [x, y, z, w]")
print(f"EKF Quaternion: {q_ekf} [x, y, z, w]")
```

Output:

```text
Reference quaternion: [0.         0.         0.70710678 0.70710678] [x, y, z, w]
TRIAD Quaternion: [-0.01693025 -0.01254686  0.7105929   0.70328776] [x, y, z, w]
QUEST Quaternion: [-0.01691007 -0.01257405  0.70946175  0.70442883] [x, y, z, w]
EKF Quaternion: [-0.00855144 -0.00616122  0.70900606  0.70512362] [x, y, z, w]
```

QUEST estimate is slightly better than TRIAD, but EKF is closest (result differs due to random noise). Selection of fusion algorithm and parameters is set in initial_settings.json (TRIAD has no extra parameters).

### Electromagnetic Coils

Sensors provide reliable information about satellite dynamics and attitude; active elements use it for control consistent with mission profile. Electromagnetic coils (magnetorquers) are basic control elements for small satellites on low Earth orbit. Most often three-axis coil assemblies on magnetic or air cores. Air-core assumed here (easier integration).

Operation basis: a coil with current generates magnetic dipole. Interacting with Earth’s magnetic field produces a torque:

```
τ = m × B
```

Units: m [A·m²], B [T], τ [N·m].

Control is limited by locally weak field and geometry: when required moment is parallel to B, m × B ≈ 0 and control becomes ineffective.

Parameters in initial_settings.json:

```json
"Magnetorquer": {
  "Coils": 200,
  "MaxCurrent": 0.2,
  "CoilArea": 90,
  "SafetyFactor": 0.9,
  "AlphaCap": 0.5,
  "UnitInfo": "coils, cm^2, A, unitless, deg/s2"
}
```

Note: CoilArea in cm² converted to m² (×1e‑4).

# Simulation

These components form a model enabling simulation of a small satellite on Earth orbits.

## Input Settings

To simplify management of all parameters affecting simulation flow they were placed in a JSON file and initial_settings.py created to manage it. This allows control over each stage without editing underlying code. Multiple variants can be created since the JSON path is passed as an input parameter rather than hard-coded. Entire configuration is grouped like a Python dict with five top-level keys and more detailed nested ones. An informational unit entry appears for parameters (not processed).

- Simulation – basic simulation parameters such as physical constants, iteration control, start date.
- Satellite – parameters describing the satellite: mass, inertia, initial attitude.
- Sensors – parameters related to sensors and measurements; includes sensor settings and fusion algorithm parameters.
- Actuators – parameters of active elements (here only coil): number of turns, cross-sectional area, safety factor, acceleration cap.
- Controls – parameters related to satellite control: fusion algorithm choice, detumbling and pointing algorithm settings, mode switching.

initial_settings.py is prepared for current structure. Missing required keys raise errors, but additional entries can be added. If added at leaf level they are simply appended. Adding a unit is optional. Example:

```
"Magnetorquer":
{
  "Coils": 200,
  "MaxCurrent": 0.2,
  "CoilArea": 90,
  "SafetyFactor": 0.9,
  "AlphaCap": 0.5,
  "WireDia": 0.8,
  "UnitInfo": "coils, cm^2, A, unitless, deg/s2, mm"
}
```

Then in the defined dict it appears as a key:

```
setup.magnetorquer_params["WireDia"]
```

If an entry is added elsewhere it goes to other parameters stored in self.other_parameters:

```
{
  "Simulation": { "PlanetConst": ... },
  "SomeOtherData": {
    "A": 2,
    "B": 4
  }
}
```

Loaded identically:

```
data_dict = setup.other_parameters["SomeOtherData"]
A = data_dict["A"]
B = data_dict["B"]
```

## Satellite Parameters

To simplify satellite operations it is treated as an object composed of components and key describing parameters are defined as properties in satellite.py. Note initialization:

```
satellite = SatelliteImplementation(setup, tle, magnetometer, sunsensor, sensor_fusion)
```

So elements affecting satellite state are input data, TLE, magnetometer, Sun sensor, and fusion algorithm. Both constant parameters and updating ones describing current state can be accessed. Predefined properties appear as hints in editor and their description can be shown quickly. Example parameters:

```
print(satellite.angular_velocity)
print(satellite.inertia_matrix)
print(satellite.magnetic_field)
print(satellite.torque)
```

Output:

```
[  8. -14.  11.]
[[0.002 0.    0.   ]
 [0.    0.002 0.   ]
 [0.    0.    0.002]] (array([-12117.938582  , -12371.50694404, -31464.45657624]), array([-12121.11401262, -12375.32291477, -31468.13563725]))
[0. 0. 0.]
```

## Algorithms

Satellite behavior is based on implemented control algorithms. Their definition depends on mission character, available sensors, and actuators. Three basic states for satellites: detumbling (reducing rotation), pointing (target alignment), and off state. Transitions defined by achieving certain criteria and found in initial_settings.json.

### Detumbling (B-dot)

Reducing angular velocity is usually the first and most important control task. A deployed satellite often reaches orbit rotating relatively fast uncontrollably. To proceed to mission operations spatial attitude must be stabilized by slowing to a threshold. Bringing and maintaining sufficiently low rotation is detumbling’s task. For the assumed use of only electromagnetic coils in this project it is implemented via B-dot. Its name and operation relate to magnetic field derivative. Change rate of measured field is proportional to angular velocity components. Practically generated dipole must oppose rotation direction and be scaled by a gain.

```
m = -k * dB/dt
```

There are several modifications helpful in certain cases. For an example 1U satellite differences are small. Below variants:

- Modified B-dot – does not rely on field derivative but directly on measured angular velocity. Less sensitive to field perturbations but requires angular velocity source.

```
m = -k (ω × B)
```

- Proportional B-dot – adds damping term based also on angular velocity. Can stabilize under changing field but if too strong may cause oscillations or destabilize at low speeds.

```
m = -k * dB/dt - k_p * ω
```

- Speed adaptive – same principle but gain depends on current angular velocity relative to a reference threshold. Above increased, below decreased. Speeds initial reduction but smooths controller later; may hinder damping at very low speeds.

```
k = k * (|ω|/ω_ref)^a
```

- Field adaptive – also modifies gain only. Depends on absolute magnetic field value relative to a reference (often near Earth average ~45000 nT). Helps mitigate orbital variation influence: increases gain when field weakens, decreases when it strengthens.

```
k = k * (B_ref/|B|)^b
```

Variants can combine. In most cases basic B-dot suffices. Algorithm is simple and reliable; for a small satellite it should reduce even large speeds quickly; time spans from under one hour to a few (simulation time) depending on initial conditions. Resulting angular velocity plot should smoothly converge to 0 resembling 1/x. If oscillations, direction changes, or instability are observed verify parameters.

- Gain – primary scaling converting derivative to sufficiently large dipole within coil capabilities. For example 1U values between 1000 and 4000 are sufficient; larger satellites can increase accordingly. Monitor coil load: if at max for long periods reduce gain; if very low for long periods increase. Auxiliary parameter: speed drop—too slow suggests gain too small.
- Proportional gain – auxiliary; value should remain small. Considering direct combination with angular velocity in radians use values between 0.05 and 0.6.

Satellite mass, inertia matrix, and coil parameters must be considered; though not in algorithm they are key for converting dipole to torque.

### Pointing (B-cross)
WARNING: This element remains under development and is not fully stable.

Pointing orients a given satellite face toward a target. For near-Earth objects typically Earth (e.g. camera) or Sun (e.g. solar panels). One algorithm usable with coils is B-cross. It relies on cross products minimizing angle between two vectors (target and current). Basic version has alignment and damping components:

```
m = m_align + m_damp = k_a (error_vector × B) + k_d (ω × B)
```

A normalized variant relative to magnetic field is often used for stability; it changes gain magnitudes:

```
m = m_align + m_damp = k_a (error_vector × B)/||B||^2 + k_d (ω × B)/||B||^2
```

Using only coils brings difficulties: sensitivity to local field variation, inability to generate moment when component is parallel, gain tuning trouble, limited precise control. Thus algorithm is used for initial alignment or coarse pointing.

### Computing Angular Acceleration

Theoretical magnetic dipole from coils is first step to desired angular acceleration. First convert components to required coil current considering turns and cross-sectional area:

```
i = m / (n_coils * A_coil)
```

A in m² (from RodArea_cm2 × 1e‑4). Torque τ equals sum of magnetic torque and disturbance torques:

```
τ_total = τ_mag + τ_bias
τ_mag = m × B
α = I^{-1} ( τ_total − ω × (I ω) )
```

where:

```
coriolis = ω × (I * w)
```

After obtaining acceleration apply caps to reduce destabilization from abrupt spikes. Scale current accordingly.

### Mode Switching

Each mode has defined start and end parameters. Transitions are automatic and defined in initial_settings.json:

```
"ModeManagement":
{
  "DetumblingOff": 0.5,
  "DetumblingOn": 1.0,
  "PointingOff": 10.0,
  "PointingOn": 12.0,
  "PointingDwellTime": 90,
  "UnitInfo": "deg/s, deg, deg, deg, s"
}
```

For detumbling key is angular speed; for pointing maintaining angular error within bounds for dwell time. Thresholds have hysteresis: PointingOff < PointingOn and DetumblingOff < DetumblingOn to avoid frequent toggling.

# Simulation Progress and Analysis

In a standard situation simulation starts with higher angular velocity reduced by detumbling, then pointing (B-cross) and off state alternate. If pointing causes excessive spin-up, detumbling restarts. Full cycle typically in several to tens of thousands of iterations. Pointing not fully finished and stable; unpredictable behavior may occur. Besides confirming some initial parameters the terminal usually shows only mode change info and iteration timer. While adding logs is convenient for temporary checks and development, to avoid clutter and simplify work without digging into functions, analysis tools were implemented: state vector and plotting.

### State Vector

The state vector is a table containing all possible data describing satellite state at given times: angular velocity, position, magnetic field measurements, Sun vector, torque, etc. Class design allows adding any value at any time. If not measured or computed during an iteration empty entries are filled with NaN. Code below shows registering values:

```python
satellite.state_vector.reset()  # without reset it would accumulate data from previous tests
satellite.state_vector.next_row()  # initialization and iterating row index

satellite._state_vector.register_vector("angular_velocity", satellite.angular_velocity, labels=["x", "y", "z"])
satellite._state_vector.register_value("latitude", satellite.latitude)

print(satellite._state_vector.to_dataframe())
```

Output:

```text
angular_velocity_x angular_velocity_y angular_velocity_z latitude
0 2.0               -3.0               4.0                26.40241
```

If the first line satellite.state_vector.reset() is removed and code executed multiple times data begins accumulating. If a new parameter is registered mid-way earlier values are auto-filled:

```python
# satellite.state_vector.reset()
satellite.state_vector.next_row()

satellite._state_vector.register_vector("angular_velocity", satellite.angular_velocity, labels=["x", "y", "z"])
satellite._state_vector.register_value("latitude", satellite.latitude)
satellite._state_vector.register_value("pointing_error", satellite.pointing_error_angle)

print(satellite._state_vector.to_dataframe())
```

Example:

```text
  angular_velocity_x angular_velocity_y angular_velocity_z latitude pointing_error
0 2.0                -3.0               4.0                26.40241 NaN
1 2.0                -3.0               4.0                26.40241 NaN
2 2.0                -3.0               4.0                26.40241 NaN
3 2.0                -3.0               4.0                26.40241 0.0
4 2.0                -3.0               4.0                26.40241 0.0
```

To simplify basic table management utilities.py contains basic_state_vector() which records selected basic parameters each iteration. For full analysis save table to CSV:

```python
satellite.state_vector.to_csv('simulation_state_vector.csv')
```

### Plots

Another helpful element: plots. Implemented in two variants. One uses matplotlib—popular tool enabling creation of many data representations, styling, and saving as images. The second uses plotly—a more advanced tool enabling interactive zoom and precise value readout in HTML or notebook window plus defining live plots (helpful during development). Both variants implemented similarly in visualizations.py as separate classes with templates for line, scatter, and 3D plots (orbit visualization). Using these templates ready functions for selected basic parameters were created. All collected in basic_plots() that generates and saves plots. Below example using templates:

```
from visualizations.visualizations import MatplotlibPlots
from visualizations.visualizations import PlotlyPlots
import numpy as np

x = np.arange(0, 10, 0.1)
y = np.sin(x)

mpl = MatplotlibPlots(save=False, show=True)
pp = PlotlyPlots(save=False, show=True)

mpl.line_plot({"Sine Wave": (x, y)}, "Sine Function", "X-axis", "sin(x)", "sine_wave_matplotlib")
pp.line_plot({"Sine Wave": (x, y)}, "Sine Function", "X-axis", "sin(x)", "sine_wave_plotly")
```

This code should display two plots. Below how to add live plotting with plotly. Full example in notebook version. Add at start:

```python
live_w = LivePlotlyLine(
  labels=["wx", "wy", "wz", "|w|"],
  title="Angular velocity (live)",
  xlabel="Time (s)",
  ylabel="deg/s",
  window=1000,
)
```

Update each iteration:

```python
wx, wy, wz = map(float, satellite.angular_velocity)
wmag = float(np.sqrt(wx*wx + wy*wy + wz*wz))
live_w.update(float(x), [wx, wy, wz, wmag])
```

Interactive plot shows below including defined number of points. Multiple live plots possible. For matplotlib plots a configuration class defines styling parameters.

### Tests

Since this repository is more scientific-engineering in nature, tests are not implemented in classical production sense. Their goal is facilitating editing, confirming operation of certain code elements, and ensuring fragments return values in expected shape—helpful for identifying potential errors introduced during editing. Run tests with:

```powershell
pytest
```

Optionally clear terminal before running to improve readability.

Additionally Jupyter Notebook examples.ipynb contains usage examples of fragments described in text; correct execution also helps during development.

# Summary

The created code is a base for further modification and expansion of a simple ADCS system for a small satellite; modular architecture makes adding additional sensors (e.g. horizon sensor, star trackers, full IMU) and actuators (e.g. reaction wheels and hybrid electromagnetic coil) easier. Pointing mode (B‑cross) still requires refinement—stability, gain tuning, and switching logic. You are encouraged to use, modify, and draw inspiration from this repository for your own projects. If you notice an error or have an improvement suggestion, feel free to get in touch.

## Materials

Below are materials that were helpful at various stages of creating this simulation. Note that near some specific functions links to helpful resources directly related to them were added. This is not all used sources. For certain topics—especially mathematics, transformations, or code—forum threads proved invaluable. For strictly technical subjects English Wikipedia often offers good summaries. Language models (external chat and GitHub Copilot) were also used.

https://www.aero.iitb.acs/satelliteWiki/index.php/Main_Page
https://automaticaddison.com/extended-kalman-filter-ekf-with-python-code-example/
https://digitalcommons.usu.edu/smallsat/2019/all2019/49/
https://www.diva-portal.org/smash/get/diva2:1018210/FULLTEXT02.pdf
https://docs.advancednavigation.com/gnss-compass/Foundation%20Knowledge.htm
https://files.core.ac.uk/download/pdf/286701577.pdf
https://www.gov.br/inpe/pt-br/area-conhecimento/unidade-nordeste/conasat/documentacja/nano-satelites-pelo-mundo/aausat-3-aalborg-university-denmark/aausat-3-adcs-attitudedeterminationandcontrolsystem.pdf
https://magneticearth.org/pages/models.html
https://medium.com/@sasha_przybylski/the-math-behind-extended-kalman-filtering-0df981a87453
https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/Tutorials/pdf/individual_docs/17_frames_and_coordinate_systems.pdf
https://quaternions.online/
https://probablydance.com/2017/08/05/intuitive-quaternions/
https://www.3dgep.com/understanding-quaternions/