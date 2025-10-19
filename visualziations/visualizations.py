import matplotlib.pyplot as plt
from templates.initial_settings_template import SimulationSetup
from pathlib import Path
import pandas as pd
import numpy as np


def plot_orbit(
        state_vector: pd.DataFrame,
        setup: SimulationSetup,
        output_dir: Path = Path(__file__).resolve().parent
) -> None:
    """
    Plot the orbit of the satellite in 3D. All plots are saved in the
    plots directory.

    Args:
        state_vector (pd.DataFrame): The state vector of the satellite.
        setup (SimulationSetup): The simulation setup object containing
            parameters about the satellite, planet etc.
        output_dir (Path): The directory to save the plots.
    """
    output_dir.joinpath('plots').mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # draw sphere
    u, v = np.mgrid[0: 2 * np.pi: 40j, 0: np.pi: 20j]
    x = np.cos(u) * np.sin(v) * setup.planet_data['R'] / 1000
    y = np.sin(u) * np.sin(v) * setup.planet_data['R'] / 1000
    z = np.cos(v) * setup.planet_data['R'] / 1000

    ax.set_title('Orbit GCRS (ECEF)')
    ax.plot_wireframe(x, y, z, color="g")
    ax.scatter(
        state_vector["position_x"],
        state_vector["position_y"],
        state_vector["position_z"],
        color="r",
        s=20,
    )
    ax.set_aspect('equal')
    ax.legend(['Earth Surface', 'Orbit'])
    plt.savefig(
        output_dir.joinpath('plots', 'orbit.png'),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()


def plot_lla(
        state_vector: pd.DataFrame,
        output_dir: Path = Path(__file__).resolve().parent
) -> None:
    """
    Plot the latitude, longitude and altitude of the satellite over time.

    Args:
        state_vector (pd.DataFrame): The state vector of the satellite.
        output_dir (Path): The directory to save the plots.
    """
    output_dir.joinpath('plots').mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Latitude and longitude (degrees)', color='tab:blue')
    ax1.plot(
        state_vector.index,
        state_vector['latitude'],
        color='tab:blue',
        label='Latitude'
    )
    ax1.plot(
        state_vector.index,
        state_vector['longitude'],
        color='tab:orange',
        label='Longitude'
    )

    ax2 = ax1.twinx()
    ax2.set_ylabel('Altitude (km)', color='tab:green')

    ax2.plot(
        state_vector.index,
        state_vector['altitude'],
        color='tab:green',
        label='Altitude'
    )

    ax2.tick_params(axis='y', labelcolor='tab:green')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.grid()

    ax1.set_title('Satellite Position in LLA')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    plt.savefig(
        output_dir.joinpath('plots', 'lla.png'),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()


def plot_position(
        state_vector: pd.DataFrame,
        output_dir: Path = Path(__file__).resolve().parent
) -> None:
    """
    Plot the position of the satellite in GCRS (ECI) frame over time.

    Args:
        state_vector (pd.DataFrame): The state vector of the satellite.
        output_dir (Path): The directory to save the plots.

    """
    output_dir.joinpath('plots').mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (km)', color='tab:blue')
    ax.plot(state_vector.index, state_vector['position_x'], color='tab:blue')
    ax.plot(state_vector.index, state_vector['position_y'], color='tab:orange')
    ax.plot(state_vector.index, state_vector['position_z'], color='tab:green')

    fig.tight_layout()
    ax.grid()

    ax.set_title('Satellite Position in GCRS')
    ax.legend(['X', 'Y', 'Z'], loc=0)

    plt.savefig(
        output_dir.joinpath('plots', 'position_GCRS.png'),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()


def plot_magnetic_field_sbf(
        state_vector: pd.DataFrame,
        output_dir: Path = Path(__file__).resolve().parent
) -> None:
    """
    Plot the magnetic field measured by the satellite in SBF over time.

    Args:
        state_vector (pd.DataFrame): The state vector of the satellite.
        output_dir (Path): The directory to save the plots.
    """
    output_dir.joinpath('plots').mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Magnetic Field (nT)', color='tab:blue')
    ax.plot(
        state_vector.index,
        state_vector['magnetic_field_sbf_x'],
        color='tab:blue',
        label='SBF X'
    )
    ax.plot(
        state_vector.index,
        state_vector['magnetic_field_sbf_y'],
        color='tab:orange',
        label='SBF Y'
    )
    ax.plot(
        state_vector.index,
        state_vector['magnetic_field_sbf_z'],
        color='tab:green',
        label='SBF Z'
    )

    fig.tight_layout()
    ax.grid()

    ax.set_title('Satellite Magnetic Field in SBF')
    ax.legend(loc=0)

    plt.savefig(
        output_dir.joinpath('plots', 'magnetic_field_sbf.png'),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()


def plot_magnetic_field_eci(
        state_vector: pd.DataFrame,
        output_dir: Path = Path(__file__).resolve().parent
) -> None:
    """
    Plot the magnetic field in ECI over time.

    Args:
        state_vector (pd.DataFrame): The state vector of the satellite.
        output_dir (Path): The directory to save the plots.
    """
    output_dir.joinpath('plots').mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Magnetic Field (nT)', color='tab:blue')
    ax.plot(
        state_vector.index,
        state_vector['magnetic_field_eci_x'],
        color='tab:blue',
        label='ECI X'
    )
    ax.plot(
        state_vector.index,
        state_vector['magnetic_field_eci_y'],
        color='tab:orange',
        label='ECI Y'
    )
    ax.plot(
        state_vector.index,
        state_vector['magnetic_field_eci_z'],
        color='tab:green',
        label='ECI Z'
    )

    fig.tight_layout()
    ax.grid()

    ax.set_title('Satellite Magnetic Field in ECI')
    ax.legend(loc=0)

    plt.savefig(
        output_dir.joinpath('plots', 'magnetic_field_eci.png'),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()


def plot_angular_velocity(
    state_vector: pd.DataFrame,
    output_dir: Path = Path(__file__).resolve().parent
) -> None:
    """
    Plot the angular velocity of the satellite over time.

    Args:
        state_vector (pd.DataFrame): The state vector of the satellite.
        output_dir (Path): The directory to save the plots.
    """
    output_dir.joinpath('plots').mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular Velocity (deg/s)', color='tab:blue')
    ax.plot(state_vector.index, state_vector['angular_velocity_x'], color='tab:blue', label='wx')
    ax.plot(state_vector.index, state_vector['angular_velocity_y'], color='tab:orange', label='wy')
    ax.plot(state_vector.index, state_vector['angular_velocity_z'], color='tab:green', label='wz')

    angular_velocity_magn = np.sqrt(
        state_vector['angular_velocity_x'] ** 2 + state_vector['angular_velocity_y'] ** 2 + state_vector['angular_velocity_z'] ** 2
    )

    ax.plot(state_vector.index, angular_velocity_magn, color='tab:red', label='|w|')

    fig.tight_layout()
    ax.grid()

    ax.set_title('Satellite Angular Velocity')
    ax.legend(loc=0)

    plt.savefig(
        output_dir.joinpath('plots', 'angular_velocity.png'),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()


def plot_euler_angles(
    state_vector: pd.DataFrame,
    output_dir: Path = Path(__file__).resolve().parent
) -> None:
    """
    Plot the Euler angles of the satellite over time.

    Args:
        state_vector (pd.DataFrame): The state vector of the satellite.
        output_dir (Path): The directory to save the plots.
    """
    output_dir.joinpath('plots').mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Euler Angles (degrees)', color='tab:blue')
    ax.plot(
        state_vector.index,
        state_vector['euler_angles_x1'],
        color='tab:blue',
        label='roll (Phi)'
    )
    ax.plot(
        state_vector.index,
        state_vector['euler_angles_y1'],
        color='tab:orange',
        label='pitch (Theta)'
    )
    ax.plot(
        state_vector.index,
        state_vector['euler_angles_z1'],
        color='tab:green',
        label='yaw (Psi)'
    )

    fig.tight_layout()
    ax.grid()

    ax.set_title('Satellite Euler Angles')
    ax.legend(loc=0)

    plt.savefig(
        output_dir.joinpath('plots', 'euler_angles.png'),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()


def plot_torque(
    state_vector: pd.DataFrame,
    output_dir: Path = Path(__file__).resolve().parent
) -> None:
    """
    Plot the torque applied by the magnetorquers over time.

    Args:
        state_vector (pd.DataFrame): The state vector of the satellite.
        output_dir (Path): The directory to save the plots.
    """
    output_dir.joinpath('plots').mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque (N·m)', color='tab:blue')
    ax.plot(state_vector.index, state_vector['torque_x'], color='tab:blue', label='Torque X')
    ax.plot(state_vector.index, state_vector['torque_y'], color='tab:orange', label='Torque Y')
    ax.plot(state_vector.index, state_vector['torque_z'], color='tab:green', label='Torque Z')

    torque_magn = np.sqrt(
        state_vector['torque_x'] ** 2 + state_vector['torque_y'] ** 2 + state_vector['torque_z'] ** 2
    )

    ax.plot(state_vector.index, torque_magn, color='tab:red', label='|Torque|')

    fig.tight_layout()
    ax.grid()

    ax.set_title('Magnetorquer Applied Torque')
    ax.legend(loc=0)

    plt.savefig(
        output_dir.joinpath('plots', 'torque.png'),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()


def plot_angular_acceleration(
    state_vector: pd.DataFrame,
    output_dir: Path = Path(__file__).resolve().parent
) -> None:
    """
    Plot the angular acceleration of the satellite over time.

    Args:
        state_vector (pd.DataFrame): The state vector of the satellite.
        output_dir (Path): The directory to save the plots.
    """
    output_dir.joinpath('plots').mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular Acceleration (deg/s²)', color='tab:blue')
    ax.plot(state_vector.index, state_vector['angular_acceleration_x'], color='tab:blue', label='Alpha X')
    ax.plot(state_vector.index, state_vector['angular_acceleration_y'], color='tab:orange', label='Alpha Y')
    ax.plot(state_vector.index, state_vector['angular_acceleration_z'], color='tab:green', label='Alpha Z')

    angular_acceleration_magn = np.sqrt(
        state_vector['angular_acceleration_x'] ** 2 +
        state_vector['angular_acceleration_y'] ** 2 +
        state_vector['angular_acceleration_z'] ** 2
    )

    ax.plot(state_vector.index, angular_acceleration_magn, color='tab:red', label='|Alpha|')

    fig.tight_layout()
    ax.grid()

    ax.set_title('Satellite Angular Acceleration')
    ax.legend(loc=0)

    plt.savefig(
        output_dir.joinpath('plots', 'angular_acceleration.png'),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()


def plot_pointing_error(
    state_vector: pd.DataFrame,
    output_dir: Path = Path(__file__).resolve().parent
) -> None:
    """
    Plot the pointing error of the satellite over time.

    Args:
        state_vector (pd.DataFrame): The state vector of the satellite.
        output_dir (Path): The directory to save the plots.
    """
    output_dir.joinpath('plots').mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pointing Error (degrees)', color='tab:blue')
    ax.scatter(
        state_vector.index,
        state_vector['pointing_error'],
        color='tab:blue',
        marker='x',
        label='Pointing Error'
    )

    fig.tight_layout()
    ax.grid()

    ax.set_title('Satellite Pointing Error')
    ax.legend(loc=0)

    plt.savefig(
        output_dir.joinpath('plots', 'pointing_error.png'),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()
    

def plot_sun_vector_eci(
    state_vector: pd.DataFrame,
    output_dir: Path = Path(__file__).resolve().parent
) -> None:
    """
    Plot the sun vector in ECI frame over time.
    """
    output_dir.joinpath('plots').mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Sun Vector (ECI)', color='tab:blue')
    ax.plot(
        state_vector.index,
        state_vector['sun_vector_eci_x'],
        color='tab:blue',
        label='ECI X'
    )
    ax.plot(
        state_vector.index,
        state_vector['sun_vector_eci_y'],
        color='tab:orange',
        label='ECI Y'
    )
    ax.plot(
        state_vector.index,
        state_vector['sun_vector_eci_z'],
        color='tab:green',
        label='ECI Z'
    )

    fig.tight_layout()
    ax.grid()
    ax.set_title('Satellite Sun Vector in ECI')
    ax.legend(loc=0)

    plt.savefig(
        output_dir.joinpath('plots', 'sun_vector_eci.png'),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()


def plot_sun_vector_sbf(
    state_vector: pd.DataFrame,
    output_dir: Path = Path(__file__).resolve().parent
) -> None:
    """
    Plot the sun vector in SBF frame over time.
    """
    output_dir.joinpath('plots').mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Sun Vector (SBF)', color='tab:blue')
    ax.plot(
        state_vector.index,
        state_vector['sun_vector_sbf_x'],
        color='tab:blue',
        label='SBF X'
    )
    ax.plot(
        state_vector.index,
        state_vector['sun_vector_sbf_y'],
        color='tab:orange',
        label='SBF Y'
    )
    ax.plot(
        state_vector.index,
        state_vector['sun_vector_sbf_z'],
        color='tab:green',
        label='SBF Z'
    )

    fig.tight_layout()
    ax.grid()
    ax.set_title('Satellite Sun Vector in SBF')
    ax.legend(loc=0)

    plt.savefig(
        output_dir.joinpath('plots', 'sun_vector_sbf.png'),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()
