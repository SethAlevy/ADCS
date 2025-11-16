from matplotlib.testing import setup
import core.utilities as ut
import core.transformations as tr
from core.logger import log
from setup.two_line_element import TwoLineElementReader
from spacecraft.satellite import SatelliteImplementation
from setup.initial_settings import SimulationSetupReader
from spacecraft.sensors import (
    MagnetometerImplementation,
    SunsensorImplementation,
    SensorFusionImplementation,
)
from visualizations.visualizations import MatplotlibPlots, PlotlyPlots

import argparse
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run ADCS simulation for small sat with parameters defined in"
        "the initial settings JSON file."
    )

    parser.add_argument(
        "--settings-path",
        type=str,
        required=False,
        default=str(Path(__file__).parent / "setup" / "initial_settings.json"),
        help="Path to the initial settings JSON file.",
    )

    parser.add_argument(
        "--tle-path",
        type=str,
        required=False,
        default=str(Path(__file__).parent / "setup" / "tle"),
        help="Path to the TLE file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default=str(Path(__file__).parent / "output"),
        help="Directory to save the output plots.",
    )

    parser.add_argument(
        "--mpl-plots",
        action="store_true",
        help="Flag to save plots generated with matplotlib to the output directory.",
    )

    parser.add_argument(
        "--pl-plots",
        action="store_true",
        help="Flag to save plots generated with plotly to the output directory.",
    )

    parser.add_argument(
        "--save-state-vector",
        action="store_true",
        help="Flag to save the state vector to a CSV file.",
    )

    parser.add_argument(
        "--no-detumbling",
        action="store_true",
        help="Disable detumbling mode in the simulation.",
    )

    parser.add_argument(
        "--no-pointing",
        action="store_true",
        help="Disable pointing mode in the simulation.",
    )

    return parser.parse_args()


def main():
    print('start')
    args = parse_arguments()

    settings_path = Path(args.settings_path)
    tle_path = Path(args.tle_path)
    output_dir = Path(args.output_dir)
    mpl_plots = args.mpl_plots
    pl_plots = args.pl_plots
    save_state_vector = args.save_state_vector
    no_detumbling = args.no_detumbling
    no_pointing = args.no_pointing

    output_dir.mkdir(parents=True, exist_ok=True)

    setup = SimulationSetupReader(settings_path)
    tle = TwoLineElementReader(tle_path)
    magnetometer = MagnetometerImplementation(setup)
    sunsensor = SunsensorImplementation(setup)
    sensor_fusion = SensorFusionImplementation(
        setup, 
        setup.sensor_fusion_algorithm, 
        tr.euler_xyz_to_quaternion(setup.euler_angles)
    )
    satellite = SatelliteImplementation(
        setup, 
        tle, 
        magnetometer, 
        sunsensor, 
        sensor_fusion
    )

    quaternion_prev = satellite.quaternion.copy()

    satellite.state_vector.reset()
    satellite.state_vector.next_row()
    ut.basic_state_vector(satellite)
    ut.log_init_state(setup)

    t_start = setup.iterations_info["Start"]
    dt = setup.iterations_info["Step"]
    t_end = setup.iterations_info["Stop"]

    for x in range(t_start, t_end, dt):
        if x % setup.iterations_info["LogInterval"] == 0:
            log(f"Iteration {x} of {t_end}")
        
        satellite.manage_actuators_sensors_timing()
        satellite.update_iteration(x)
        satellite.apply_rotation()

        mag_field_sbf, mag_field_eci = satellite.magnetic_field
        sun_vector_sbf, sun_vector_eci = satellite.sun_vector

        satellite.fuse_sensors(
            [mag_field_sbf, sun_vector_sbf],
            [mag_field_eci, sun_vector_eci],
            quaternion_prev
        )

        satellite.manage_modes()
        if not no_detumbling:
            satellite.apply_detumbling()
        if not no_pointing:
            satellite.apply_pointing()

        satellite.state_vector.next_row()
        ut.basic_state_vector(satellite)

        quaternion_prev = satellite.quaternion.copy() 
    state_df = satellite.state_vector.to_dataframe()
    if save_state_vector:
        satellite.state_vector.to_csv(output_dir / "state_vector.csv")
    if mpl_plots:
        mpl = MatplotlibPlots(output_dir, save=True)
        mpl.basic_plots(state_df, setup)
    if pl_plots:
        pp = PlotlyPlots(output_dir, save=True)
        pp.basic_plots(state_df, setup)


if __name__ == "__main__":
    main()
    