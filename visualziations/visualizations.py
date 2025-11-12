import matplotlib.pyplot as plt
from templates.initial_settings_template import SimulationSetup
from typing import Iterable, Tuple, Optional
from dataclasses import dataclass, field
from matplotlib import cycler
from pathlib import Path
import pandas as pd
import numpy as np


@dataclass
class PlotConfig:
    output_dir: Path
    subdir: str = "plots"
    save: bool = True
    show: bool = False
    dpi: int = 300
    format: str = "png"

    # Style
    figsize: Tuple[float, float] = (8.0, 6.0)
    style: Optional[str] = None              # e.g. "seaborn-v0_8-paper"
    grid: bool = True
    legend_loc: int | str = 0
    title_prefix: str = ""                   # e.g., "Satellite "
    color_cycle: Optional[Iterable[str]] = field(default=None)
    linewidth: float = 1.5
    markersize: float = 4.0

    # 3D Earth wireframe (plot_orbit)
    sphere_res_u: int = 40
    sphere_res_v: int = 20

    def apply(self) -> None:
        if self.style:
            plt.style.use(self.style)
        if self.color_cycle:
            plt.rcParams["axes.prop_cycle"] = cycler(color=list(self.color_cycle))

    @property
    def plots_dir(self) -> Path:
        p = self.output_dir / self.subdir
        p.mkdir(parents=True, exist_ok=True)
        return p

    def figure(self, projection: Optional[str] = None):
        # Always apply styling before creating a figure
        self.apply()
        kwargs = {}
        if projection:
            kwargs["subplot_kw"] = {"projection": projection}
        fig, ax = plt.subplots(figsize=self.figsize, **kwargs)
        return fig, ax

    def finalize(self, fig, filename: str) -> None:
        if self.save:
            fig.savefig(self.plots_dir /
                        f"{filename}.{self.format}", dpi=self.dpi, bbox_inches="tight")
        if self.show:
            plt.show()
        plt.close(fig)


class MatplotlibPlots:
    def __init__(self, output_dir=Path(__file__).resolve().parent) -> None:
        self.cfg = PlotConfig(output_dir=output_dir)

    def _setup_ax(self, ax, title: str, xlabel: str, ylabel: str, grid: bool) -> None:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if grid:
            ax.grid()

    def line_plot(
        self,
        series: (
            list[tuple[np.ndarray, np.ndarray, str]]
            | dict[str, tuple[np.ndarray, np.ndarray]]
        ),
        title: str,
        xlabel: str,
        ylabel: str,
        filename: str,
        legend: bool = True,
    ) -> None:
        fig, ax = self.cfg.figure()
        self._setup_ax(ax, f"{self.cfg.title_prefix}{title}", xlabel, ylabel, self.cfg.grid)

        if isinstance(series, dict):
            items = [(xy[0], xy[1], label) for label, xy in series.items()]
        else:
            items = series

        for x, y, label in items:
            ax.plot(x, y, linewidth=self.cfg.linewidth, label=label)

        if legend:
            ax.legend(loc=self.cfg.legend_loc)

        fig.tight_layout()
        self.cfg.finalize(fig, filename)

    def scatter_plot(
        self,
        series: (
            list[tuple[np.ndarray, np.ndarray, str]]
            | dict[str, tuple[np.ndarray, np.ndarray]]
        ),
        title: str,
        xlabel: str,
        ylabel: str,
        filename: str,
        marker: str = "x",
        legend: bool = True,
    ) -> None:
        fig, ax = self.cfg.figure()
        self._setup_ax(ax, f"{self.cfg.title_prefix}{title}", xlabel, ylabel, self.cfg.grid)

        if isinstance(series, dict):
            items = [(xy[0], xy[1], label) for label, xy in series.items()]
        else:
            items = series

        for x, y, label in items:
            ax.scatter(x, y, s=self.cfg.markersize, marker=marker, label=label)

        if legend:
            ax.legend(loc=self.cfg.legend_loc)

        fig.tight_layout()
        self.cfg.finalize(fig, filename)

    def twin_axis_line_plot(
        self,
        primary: dict[str, tuple[np.ndarray, np.ndarray]],
        secondary: dict[str, tuple[np.ndarray, np.ndarray]],
        title: str,
        xlabel: str,
        y1_label: str,
        y2_label: str,
        filename: str,
    ) -> None:
        fig, ax1 = self.cfg.figure()
        self._setup_ax(ax1, f"{self.cfg.title_prefix}{title}", xlabel, y1_label, self.cfg.grid)

        # primary series
        for label, (x, y) in primary.items():
            ax1.plot(x, y, linewidth=self.cfg.linewidth, label=label)

        # secondary axis
        ax2 = ax1.twinx()
        ax2.set_ylabel(y2_label)
        for label, (x, y) in secondary.items():
            ax2.plot(x, y, linewidth=self.cfg.linewidth, label=label)

        # combined legend on ax2
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc=self.cfg.legend_loc)

        fig.tight_layout()
        self.cfg.finalize(fig, filename)

    def orbit_3d_plot(
        self,
        positions_km: np.ndarray,
        planet_radius_km: float,
        filename: str = "orbit",
        title: str = "Orbit GCRS (ECEF)",
        point_size: float = 20.0,
        orbit_color: str = "r",
        surface_color: str = "g",
    ) -> None:
        fig, ax = self.cfg.figure(projection="3d")

        # Sphere wireframe
        u = np.linspace(0, 2 * np.pi, self.cfg.sphere_res_u)
        v = np.linspace(0, np.pi, self.cfg.sphere_res_v)
        uu, vv = np.meshgrid(u, v)
        x = np.cos(uu) * np.sin(vv) * planet_radius_km
        y = np.sin(uu) * np.sin(vv) * planet_radius_km
        z = np.cos(vv) * planet_radius_km
        ax.plot_wireframe(x, y, z, color=surface_color)

        # Orbit points
        ax.scatter(positions_km[:, 0], positions_km[:, 1],
                   positions_km[:, 2], color=orbit_color, s=point_size)

        ax.set_aspect("auto")
        ax.set_title(f"{self.cfg.title_prefix}{title}")
        ax.legend(["Surface", "Orbit"], loc=self.cfg.legend_loc)
        self.cfg.finalize(fig, filename)

    # High-level plots
    def plot_orbit(
        self,
        state_vector: pd.DataFrame,
        setup: SimulationSetup,
    ) -> None:
        positions = np.column_stack(
            [
                state_vector["position_x"],
                state_vector["position_y"],
                state_vector["position_z"],
            ]
        )
        planet_radius_km = setup.planet_data["R"] / 1000.0
        self.orbit_3d_plot(
            positions_km=positions,
            planet_radius_km=planet_radius_km,
            filename="orbit",
            title="Orbit GCRS (ECEF)",
        )

    def plot_lla(self, state_vector: pd.DataFrame) -> None:
        primary = {
            "Latitude": (state_vector.index.values, state_vector["latitude"].values),
            "Longitude": (state_vector.index.values, state_vector["longitude"].values),
        }
        secondary = {
            "Altitude": (state_vector.index.values, state_vector["altitude"].values),
        }
        self.twin_axis_line_plot(
            primary=primary,
            secondary=secondary,
            title="Satellite Position in LLA",
            xlabel="Time (s)",
            y1_label="Latitude and longitude (degrees)",
            y2_label="Altitude (km)",
            filename="lla",
        )

    def plot_position(self, state_vector: pd.DataFrame) -> None:
        series = {
            "X": (state_vector.index.values, state_vector["position_x"].values),
            "Y": (state_vector.index.values, state_vector["position_y"].values),
            "Z": (state_vector.index.values, state_vector["position_z"].values),
        }
        self.line_plot(
            series=series,
            title="Satellite Position in GCRS",
            xlabel="Time (s)",
            ylabel="Position (km)",
            filename="position_GCRS",
        )

    def plot_magnetic_field_sbf(self, state_vector: pd.DataFrame) -> None:
        series = {
            "SBF X": (
                state_vector.index.values,
                state_vector["magnetic_field_sbf_x"].values,
            ),
            "SBF Y": (
                state_vector.index.values,
                state_vector["magnetic_field_sbf_y"].values,
            ),
            "SBF Z": (
                state_vector.index.values,
                state_vector["magnetic_field_sbf_z"].values,
            ),
        }
        self.line_plot(
            series=series,
            title="Satellite Magnetic Field in SBF",
            xlabel="Time (s)",
            ylabel="Magnetic Field (nT)",
            filename="magnetic_field_sbf",
        )

    def plot_magnetic_field_eci(self, state_vector: pd.DataFrame) -> None:
        series = {
            "ECI X": (
                state_vector.index.values,
                state_vector["magnetic_field_eci_x"].values,
            ),
            "ECI Y": (
                state_vector.index.values,
                state_vector["magnetic_field_eci_y"].values,
            ),
            "ECI Z": (
                state_vector.index.values,
                state_vector["magnetic_field_eci_z"].values,
            ),
        }
        self.line_plot(
            series=series,
            title="Satellite Magnetic Field in ECI",
            xlabel="Time (s)",
            ylabel="Magnetic Field (nT)",
            filename="magnetic_field_eci",
        )

    def plot_angular_velocity(self, state_vector: pd.DataFrame) -> None:
        wmag = np.sqrt(
            state_vector["angular_velocity_x"].values**2
            + state_vector["angular_velocity_y"].values**2
            + state_vector["angular_velocity_z"].values**2
        )
        series = {
            "wx": (state_vector.index.values, state_vector["angular_velocity_x"].values),
            "wy": (state_vector.index.values, state_vector["angular_velocity_y"].values),
            "wz": (state_vector.index.values, state_vector["angular_velocity_z"].values),
            "|w|": (state_vector.index.values, wmag),
        }
        self.line_plot(
            series=series,
            title="Satellite Angular Velocity",
            xlabel="Time (s)",
            ylabel="Angular Velocity (deg/s)",
            filename="angular_velocity",
        )

    def plot_euler_angles(self, state_vector: pd.DataFrame) -> None:
        series = {
            "roll (Phi)": (
                state_vector.index.values,
                state_vector["euler_angles_x1"].values,
            ),
            "pitch (Theta)": (
                state_vector.index.values,
                state_vector["euler_angles_y1"].values,
            ),
            "yaw (Psi)": (
                state_vector.index.values,
                state_vector["euler_angles_z1"].values,
            ),
        }
        self.line_plot(
            series=series,
            title="Satellite Euler Angles",
            xlabel="Time (s)",
            ylabel="Euler Angles (degrees)",
            filename="euler_angles",
        )

    def plot_torque(self, state_vector: pd.DataFrame) -> None:
        tmag = np.sqrt(
            state_vector["torque_x"].values**2
            + state_vector["torque_y"].values**2
            + state_vector["torque_z"].values**2
        )
        series = {
            "Torque X": (state_vector.index.values, state_vector["torque_x"].values),
            "Torque Y": (state_vector.index.values, state_vector["torque_y"].values),
            "Torque Z": (state_vector.index.values, state_vector["torque_z"].values),
            "|Torque|": (state_vector.index.values, tmag),
        }
        self.line_plot(
            series=series,
            title="Magnetorquer Applied Torque",
            xlabel="Time (s)",
            ylabel="Torque (N·m)",
            filename="torque",
        )

    def plot_angular_acceleration(self, state_vector: pd.DataFrame) -> None:
        amag = np.sqrt(
            state_vector["angular_acceleration_x"].values**2
            + state_vector["angular_acceleration_y"].values**2
            + state_vector["angular_acceleration_z"].values**2
        )
        series = {
            "Alpha X": (
                state_vector.index.values,
                state_vector["angular_acceleration_x"].values,
            ),
            "Alpha Y": (
                state_vector.index.values,
                state_vector["angular_acceleration_y"].values,
            ),
            "Alpha Z": (
                state_vector.index.values,
                state_vector["angular_acceleration_z"].values,
            ),
            "|Alpha|": (state_vector.index.values, amag),
        }
        self.line_plot(
            series=series,
            title="Satellite Angular Acceleration",
            xlabel="Time (s)",
            ylabel="Angular Acceleration (deg/s²)",
            filename="angular_acceleration",
        )

    def plot_pointing_error(self, state_vector: pd.DataFrame) -> None:
        series = {
            "Pointing Error": (
                state_vector.index.values,
                state_vector["pointing_error"].values,
            ),
        }
        self.scatter_plot(
            series=series,
            title="Satellite Pointing Error",
            xlabel="Time (s)",
            ylabel="Pointing Error (degrees)",
            filename="pointing_error",
            marker="x",
        )

    def plot_sun_vector_eci(self, state_vector: pd.DataFrame) -> None:
        series = {
            "ECI X": (state_vector.index.values, state_vector["sun_vector_eci_x"].values),
            "ECI Y": (state_vector.index.values, state_vector["sun_vector_eci_y"].values),
            "ECI Z": (state_vector.index.values, state_vector["sun_vector_eci_z"].values),
        }
        self.line_plot(
            series=series,
            title="Satellite Sun Vector in ECI",
            xlabel="Time (s)",
            ylabel="Sun Vector (ECI)",
            filename="sun_vector_eci",
        )

    def plot_sun_vector_sbf(self, state_vector: pd.DataFrame) -> None:
        series = {
            "SBF X": (state_vector.index.values, state_vector["sun_vector_sbf_x"].values),
            "SBF Y": (state_vector.index.values, state_vector["sun_vector_sbf_y"].values),
            "SBF Z": (state_vector.index.values, state_vector["sun_vector_sbf_z"].values),
        }
        self.line_plot(
            series=series,
            title="Satellite Sun Vector in SBF",
            xlabel="Time (s)",
            ylabel="Sun Vector (SBF)",
            filename="sun_vector_sbf",
        )

    def basic_plots(
        self,
        state_vector: pd.DataFrame,
        setup: SimulationSetup,
    ) -> None:
        self.plot_orbit(state_vector, setup)
        self.plot_position(state_vector)
        self.plot_lla(state_vector)
        self.plot_magnetic_field_sbf(state_vector)
        self.plot_magnetic_field_eci(state_vector)
        self.plot_angular_velocity(state_vector)
        self.plot_euler_angles(state_vector)
        self.plot_torque(state_vector)
        self.plot_angular_acceleration(state_vector)
        self.plot_pointing_error(state_vector)
        self.plot_sun_vector_eci(state_vector)
        self.plot_sun_vector_sbf(state_vector)
