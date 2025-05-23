import matplotlib.pyplot as plt
from setup.initial_settings import SimulationSetup
from pathlib import Path
import pandas as pd
import numpy as np


def plot_orbit(state_vector: pd.DataFrame, setup: SimulationSetup, output_dir: Path = Path(__file__).resolve().parent) -> None:
    """
    Plot the orbit of the satellite in 3D. All plots are saved in the
    plots directory.

    Args:
        state_vector (pd.DataFrame): The state vector of the satellite.
        setup (SimulationSetup): The simulation setup object containing
        parameters about the satellite, planet etc.
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
    plt.savefig(
        output_dir.joinpath('plots', 'orbit.png'),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()
