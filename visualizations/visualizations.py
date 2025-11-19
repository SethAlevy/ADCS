import matplotlib.pyplot as plt
from templates.initial_settings_template import SimulationSetup
from typing import Iterable, Tuple, Optional
from dataclasses import dataclass, field
from matplotlib import cycler
from pathlib import Path
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio


@dataclass
class PlotConfig:
    """
    Central configuration for plot styling and output parameters.

    This object is used by MatplotlibPlots to:
    - apply a consistent Matplotlib style (figure size, grid, color cycle)
    - define output directory and file format for saved figures
    - control some 3D helper parameters (e.g., sphere mesh resolution)

    Attributes:
        output_dir: Base directory for outputs (plots directory is created under it).
        subdir: Subdirectory name where plots will be saved.
        dpi: Resolution for saved raster images.
        format: File extension for saved images (e.g., "png").
        figsize: Default figure size in inches (width, height).
        style: Optional Matplotlib style string (e.g., "seaborn-v0_8-paper").
        grid: Whether to enable grid by default on axes.
        legend_loc: Default legend location (int or string understood by Matplotlib).
        title_prefix: Optional prefix added to all plot titles.
        color_cycle: Optional iterable of color strings to override default cycle.
        linewidth: Default line width for line plots.
        markersize: Default marker size for scatter plots.
        sphere_res_u: Horizontal resolution for orbit 3D sphere mesh.
        sphere_res_v: Vertical resolution for orbit 3D sphere mesh.
    """
    output_dir: Path
    subdir: str = "plots"
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
        """
        Apply configured Matplotlib style parameters globally.

        Notes:
            This is called inside figure() to ensure the latest style is applied
            for every new figure creation.
        """
        if self.style:
            plt.style.use(self.style)
        if self.color_cycle:
            plt.rcParams["axes.prop_cycle"] = cycler(color=list(self.color_cycle))

    @property
    def plots_dir(self) -> Path:
        """
        Ensure and return the effective directory where plots are saved.

        Returns:
            Path: Absolute path to the plots subdirectory.
        """
        p = self.output_dir / self.subdir
        p.mkdir(parents=True, exist_ok=True)
        return p

    def figure(self, projection: Optional[str] = None):
        """
        Create a new Matplotlib figure/axis pair with configured defaults.

        Args:
            projection: Optional 3D projection ("3d") or None for 2D.

        Returns:
            tuple: (fig, ax) newly created Matplotlib figure and axis.
        """
        # Always apply styling before creating a figure
        self.apply()
        kwargs = {}
        if projection:
            kwargs["subplot_kw"] = {"projection": projection}
        fig, ax = plt.subplots(figsize=self.figsize, **kwargs)
        return fig, ax

    def finalize(self, fig, filename: str, *, save: bool, show: bool) -> None:
        """
        Finalize figure lifecycle: optionally save and/or show, then close.

        Args:
            fig: Matplotlib figure to finalize.
            filename: Base filename without extension for saving.
            save: If True, save the figure to disk using config.format and config.dpi.
            show: If True, show the figure window (or inline in notebooks).
        """
        if save:
            fig.savefig(self.plots_dir / f"{filename}.{self.format}",
                        dpi=self.dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


class MatplotlibPlots:
    def __init__(self, output_dir=Path(__file__).resolve().parent,
                 save: bool = True, show: bool = False) -> None:
        """
        This class handles the creation of plots using matplotlib library. Main
        parameters may be adjusted in PlotConfig dataclass.

        Typical usage:
            mpl = MatplotlibPlots(save=True, show=False)
            mpl.line_plot({"Sine": (x, y)}, "Sine", "t", "sin(t)", "sine_plot")

        Args:
            output_dir (Path, optional): Path where the output plots will be saved.
                Defaults to the directory of the current file.
            save (bool, optional): Whether to save the plots. Defaults to True.
            show (bool, optional): Whether to display the plots. Defaults to False.
        """
        self.cfg = PlotConfig(output_dir=output_dir)
        self._save = save
        self._show = show

    def _finalize(self, fig, filename: str) -> None:
        """
        Internal helper to delegate finalize to PlotConfig with current flags.

        Args:
            fig: Matplotlib figure.
            filename: Base output filename without extension.
        """
        self.cfg.finalize(fig, filename, save=self._save, show=self._show)

    def _setup_ax(self, ax, title: str, xlabel: str, ylabel: str, grid: bool) -> None:
        """
        Configure basic axis properties (title, labels, grid).

        Args:
            ax: Matplotlib axis to configure.
            title: Plot title (prefix may be added from config).
            xlabel: X axis label.
            ylabel: Y axis label.
            grid: If True, enable axis grid.
        """
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
        """
        Line plot template for one or multiple series.

        Args:
            series: Either a dict of {label: (x, y)} or a list of (x, y, label).
            title: Plot title.
            xlabel: X axis label.
            ylabel: Y axis label.
            filename: Base output filename without extension.
            legend: If True, show a legend.
        """
        fig, ax = self.cfg.figure()
        self._setup_ax(ax, f"{self.cfg.title_prefix}{title}",
                       xlabel, ylabel, self.cfg.grid)

        if isinstance(series, dict):
            items = [(xy[0], xy[1], label) for label, xy in series.items()]
        else:
            items = series

        for x, y, label in items:
            ax.plot(x, y, linewidth=self.cfg.linewidth, label=label)

        if legend:
            ax.legend(loc=self.cfg.legend_loc)

        fig.tight_layout()
        self._finalize(fig, filename)

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
        """
        Scatter plot template for one or multiple series.

        Args:
            series: Either a dict of {label: (x, y)} or a list of (x, y, label).
            title: Plot title.
            xlabel: X axis label.
            ylabel: Y axis label.
            filename: Base output filename without extension.
            marker: Matplotlib marker style (e.g., "x", "o").
            legend: If True, show a legend.
        """
        fig, ax = self.cfg.figure()
        self._setup_ax(ax, f"{self.cfg.title_prefix}{title}",
                       xlabel, ylabel, self.cfg.grid)

        if isinstance(series, dict):
            items = [(xy[0], xy[1], label) for label, xy in series.items()]
        else:
            items = series

        for x, y, label in items:
            ax.scatter(x, y, s=self.cfg.markersize, marker=marker, label=label)

        if legend:
            ax.legend(loc=self.cfg.legend_loc)

        fig.tight_layout()
        self._finalize(fig, filename)

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
        """
        Two-axis (left/right y) line plot.

        Args:
            primary: Mapping label -> (x, y) for the left Y axis.
            secondary: Mapping label -> (x, y) for the right Y axis.
            title: Plot title.
            xlabel: X axis label.
            y1_label: Left Y axis label.
            y2_label: Right Y axis label.
            filename: Base output filename without extension.
        """
        fig, ax1 = self.cfg.figure()
        self._setup_ax(ax1, f"{self.cfg.title_prefix}{title}",
                       xlabel, y1_label, self.cfg.grid)

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
        self._finalize(fig, filename)

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
        """
        3D orbit visualization with a spherical Earth wireframe.

        Args:
            positions_km: Nx3 array of position samples in km (GCRS/ECEF).
            planet_radius_km: Planet radius in km (used to draw the sphere).
            filename: Base output filename without extension.
            title: Plot title.
            point_size: Size of orbit points.
            orbit_color: Color for orbit points.
            surface_color: Color for the planet wireframe.
        """
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
        self._finalize(fig, filename)

    # High-level plots
    def plot_orbit(
        self,
        state_vector: pd.DataFrame,
        setup: SimulationSetup,
    ) -> None:
        """
        Plot 3D orbit using position columns from the state vector.

        Args:
            state_vector: DataFrame with position_x/y/z (km).
            setup: Simulation setup (used to get planet radius).
        """
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
        """
        Plot latitude/longitude (left Y) and altitude (right Y) vs time.

        Requires: latitude, longitude, altitude columns in state_vector.
        """
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
        """
        Plot GCRS position components X/Y/Z vs time.

        Requires: position_x, position_y, position_z columns.
        """
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

    def plot_magnetic_field_sb(self, state_vector: pd.DataFrame) -> None:
        """
        Plot magnetic field components in SB frame vs time.

        Requires: magnetic_field_sb_x/y/z columns (nT).
        """
        series = {
            "SB X": (
                state_vector.index.values,
                state_vector["magnetic_field_sb_x"].values,
            ),
            "SB Y": (
                state_vector.index.values,
                state_vector["magnetic_field_sb_y"].values,
            ),
            "SB Z": (
                state_vector.index.values,
                state_vector["magnetic_field_sb_z"].values,
            ),
        }
        self.line_plot(
            series=series,
            title="Satellite Magnetic Field in SB",
            xlabel="Time (s)",
            ylabel="Magnetic Field (nT)",
            filename="magnetic_field_sb",
        )

    def plot_magnetic_field_eci(self, state_vector: pd.DataFrame) -> None:
        """
        Plot magnetic field components in ECI frame vs time.

        Requires: magnetic_field_eci_x/y/z columns (nT).
        """
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
        """
        Plot angular velocity components and magnitude vs time.

        Requires: angular_velocity_x/y/z columns (deg/s).
        """
        wmag = np.sqrt(
            state_vector["angular_velocity_x"].values**2
            + state_vector["angular_velocity_y"].values**2
            + state_vector["angular_velocity_z"].values**2
        )
        series = {
            "wx": (
                state_vector.index.values, state_vector["angular_velocity_x"].values),
            "wy": (
                state_vector.index.values, state_vector["angular_velocity_y"].values),
            "wz": (
                state_vector.index.values, state_vector["angular_velocity_z"].values),
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
        """
        Plot Euler angles roll/pitch/yaw vs time (degrees).

        Requires: euler_angles_x1, euler_angles_y1, euler_angles_z1.
        """
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
        """
        Plot applied magnetorquer torque components and magnitude vs time.

        Requires: torque_x/y/z columns (N·m).
        """
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
        """
        Plot angular acceleration components and magnitude vs time.

        Requires: angular_acceleration_x/y/z columns (deg/s²).
        """
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
        """
        Plot pointing error as a scatter vs time (degrees).

        Requires: pointing_error column.
        """
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
        """
        Plot sun vector components in ECI vs time.

        Requires: sun_vector_eci_x/y/z columns.
        """
        series = {
            "ECI X": (
                state_vector.index.values, state_vector["sun_vector_eci_x"].values),
            "ECI Y": (
                state_vector.index.values, state_vector["sun_vector_eci_y"].values),
            "ECI Z": (
                state_vector.index.values, state_vector["sun_vector_eci_z"].values),
        }
        self.line_plot(
            series=series,
            title="Satellite Sun Vector in ECI",
            xlabel="Time (s)",
            ylabel="Sun Vector (ECI)",
            filename="sun_vector_eci",
        )

    def plot_sun_vector_sb(self, state_vector: pd.DataFrame) -> None:
        """
        Plot sun vector components in SB vs time.

        Requires: sun_vector_sb_x/y/z columns.
        """
        series = {
            "SB X": (
                state_vector.index.values, state_vector["sun_vector_sb_x"].values),
            "SB Y": (
                state_vector.index.values, state_vector["sun_vector_sb_y"].values),
            "SB Z": (
                state_vector.index.values, state_vector["sun_vector_sb_z"].values),
        }
        self.line_plot(
            series=series,
            title="Satellite Sun Vector in SB",
            xlabel="Time (s)",
            ylabel="Sun Vector (SB)",
            filename="sun_vector_sb",
        )

    def basic_plots(
        self,
        state_vector: pd.DataFrame,
        setup: SimulationSetup,
    ) -> None:
        """
        Generate a standard set of Matplotlib plots for the simulation.

        Args:
            state_vector: DataFrame containing the simulation state over time.
            setup: Simulation setup with planet data for orbit plotting.
        """
        self.plot_orbit(state_vector, setup)
        self.plot_position(state_vector)
        self.plot_lla(state_vector)
        self.plot_magnetic_field_sb(state_vector)
        self.plot_magnetic_field_eci(state_vector)
        self.plot_angular_velocity(state_vector)
        self.plot_euler_angles(state_vector)
        self.plot_torque(state_vector)
        self.plot_angular_acceleration(state_vector)
        self.plot_pointing_error(state_vector)
        self.plot_sun_vector_eci(state_vector)
        self.plot_sun_vector_sb(state_vector)


class PlotlyPlots:
    """
    Plotly plotting utilities with reusable templates and high-level helpers.

    Notes:
        - Saving produces standalone HTML files in output_dir/plots.
        - Showing uses Plotly's renderer (set via the renderer argument).
        - Methods return None when show=True to avoid duplicate notebook rendering.
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        save: bool = False,
        show: bool = False,
        renderer: str | None = None
    ) -> None:
        """
        Initialize Plotly plot helper.

        Args:
            output_dir: Base directory for outputs (plots subfolder is created).
            save: If True, write HTML files on every plot call.
            show: If True, call fig.show() on every plot call.
            renderer: Optional Plotly renderer (e.g., "vscode", "browser").
        """
        self.output_dir = (output_dir or Path(__file__).resolve().parent)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.save = save
        self.show = show
        if renderer:
            pio.renderers.default = renderer

    def _save(self, fig: go.Figure, filename: str) -> None:
        """
        Save a Plotly figure as a standalone HTML file.

        Args:
            fig: Plotly figure.
            filename: Base output filename (without extension).
        """
        out_html = self.plots_dir / f"{filename}.html"
        fig.write_html(str(out_html), include_plotlyjs="cdn")

    def _finalize(self, fig: go.Figure, filename: str) -> go.Figure | None:
        """
        Save/show a Plotly figure according to flags and return appropriately.

        Args:
            fig: Plotly figure.
            filename: Base output filename (without extension).

        Returns:
            Figure or None: Returns None when show=True to prevent notebook
            auto-rendering duplicates; otherwise returns the figure.
        """
        if self.save:
            self._save(fig, filename)
        if self.show:
            fig.show()
            return None  # avoid returning Figure to prevent auto-render duplication
        return fig

    # Example template usage
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
    ) -> go.Figure | None:
        """
        Plotly line plot template for one or multiple series.

        Args:
            series: Either a dict of {label: (x, y)} or a list of (x, y, label).
            title: Plot title.
            xlabel: X axis label.
            ylabel: Y axis label.
            filename: Base output filename without extension.
            legend: If True, display legend.
        """
        fig = go.Figure()
        if isinstance(series, dict):
            items = [(xy[0], xy[1], label) for label, xy in series.items()]
        else:
            items = series
        for x, y, label in items:
            fig.add_scatter(x=x, y=y, mode="lines", name=label)
        fig.update_layout(title=title, xaxis_title=xlabel,
                          yaxis_title=ylabel, showlegend=legend)
        return self._finalize(fig, filename)

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
        mode: str = "markers",
        legend: bool = True,
    ) -> go.Figure | None:
        """
        Plotly scatter template for one or multiple series.

        Args:
            series: Either a dict of {label: (x, y)} or a list of (x, y, label).
            title: Plot title.
            xlabel: X axis label.
            ylabel: Y axis label.
            filename: Base output filename without extension.
            mode: Plotly scatter mode (e.g., "markers", "lines+markers").
            legend: If True, display legend.
        """
        fig = go.Figure()
        if isinstance(series, dict):
            items = [(xy[0], xy[1], label) for label, xy in series.items()]
        else:
            items = series
        for x, y, label in items:
            fig.add_scatter(x=x, y=y, mode=mode, name=label)
        fig.update_layout(title=title, xaxis_title=xlabel,
                          yaxis_title=ylabel, showlegend=legend)
        fig = self._finalize(fig, filename)
        return fig

    def twin_axis_line_plot(
        self,
        primary: dict[str, tuple[np.ndarray, np.ndarray]],
        secondary: dict[str, tuple[np.ndarray, np.ndarray]],
        title: str,
        xlabel: str,
        y1_label: str,
        y2_label: str,
        filename: str,
    ) -> go.Figure | None:
        """
        Plotly two-axis (left/right y) line plot template.

        Args:
            primary: Mapping label -> (x, y) for the left Y axis.
            secondary: Mapping label -> (x, y) for the right Y axis.
            title: Plot title.
            xlabel: X axis label.
            y1_label: Left Y axis label.
            y2_label: Right Y axis label.
            filename: Base output filename without extension.
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for label, (x, y) in primary.items():
            fig.add_scatter(x=x, y=y, mode="lines", name=label, secondary_y=False)
        for label, (x, y) in secondary.items():
            fig.add_scatter(x=x, y=y, mode="lines", name=label, secondary_y=True)
        fig.update_layout(title=title)
        fig.update_xaxes(title_text=xlabel)
        fig.update_yaxes(title_text=y1_label, secondary_y=False)
        fig.update_yaxes(title_text=y2_label, secondary_y=True)
        fig = self._finalize(fig, filename)
        return fig

    def orbit_3d_plot(
        self,
        positions_km: np.ndarray,
        planet_radius_km: float | None = None,
        filename: str = "orbit_plotly",
        title: str = "Orbit GCRS (ECEF)",
        orbit_color: str = "red",
        sphere_color: str = "rgba(0,150,0,0.2)",
        sphere_res_u: int = 40,
        sphere_res_v: int = 20,
    ) -> go.Figure | None:
        """
        Plotly 3D orbit with optional spherical body surface.

        Args:
            positions_km: Nx3 array of positions in km (GCRS/ECEF). If None/empty,
                only sphere is drawn.
            planet_radius_km: Planet radius in km. If provided, a semi-transparent
                surface is added.
            filename: Base output filename without extension.
            title: Plot title.
            orbit_color: Color for the orbit trace.
            sphere_color: RGBA color for the planet surface.
            sphere_res_u: Horizontal resolution of the planet surface grid.
            sphere_res_v: Vertical resolution of the planet surface grid.
        """
        fig = go.Figure()
        if positions_km is not None and positions_km.size:
            fig.add_trace(
                go.Scatter3d(
                    x=positions_km[:, 0],
                    y=positions_km[:, 1],
                    z=positions_km[:, 2],
                    mode="lines",
                    line=dict(width=4, color=orbit_color),
                    name="Orbit"
                )
            )
        # Optional sphere
        if planet_radius_km and planet_radius_km > 0:
            u = np.linspace(0, 2*np.pi, sphere_res_u)
            v = np.linspace(0, np.pi, sphere_res_v)
            uu, vv = np.meshgrid(u, v)
            xs = planet_radius_km * np.cos(uu) * np.sin(vv)
            ys = planet_radius_km * np.sin(uu) * np.sin(vv)
            zs = planet_radius_km * np.cos(vv)
            fig.add_surface(
                x=xs,
                y=ys,
                z=zs,
                showscale=False,
                opacity=0.25,
                colorscale=[[0, sphere_color], [1, sphere_color]],
                name="Surface"
            )
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Z (km)"
            ),
            scene_aspectmode="data",
        )
        fig = self._finalize(fig, filename)
        return fig

    # High-level plots using the templates

    def plot_orbit(
        self,
        df: pd.DataFrame,
        planet_radius_km: float | None = None
    ) -> go.Figure | None:
        """
        High-level orbit plot from a state DataFrame with position columns.

        Args:
            df: State DataFrame with position_x/y/z (km).
            planet_radius_km: Optional planet radius in km to display a sphere.
        """
        pos_cols = ["position_x", "position_y", "position_z"]
        positions = None
        if all(c in df for c in pos_cols):
            positions = np.column_stack([df[c].values for c in pos_cols])
        return self.orbit_3d_plot(
            positions,
            planet_radius_km,
            filename="orbit_plotly",
            title="Orbit GCRS (ECEF)"
        )

    def plot_position(self, df: pd.DataFrame) -> go.Figure | None:
        """
        Plot GCRS position components over time using Plotly.

        Requires: position_x, position_y, position_z columns.
        """
        x = df.index.values
        series = {
            "position_x": (x, df["position_x"].values) if "position_x" in df else None,
            "position_y": (x, df["position_y"].values) if "position_y" in df else None,
            "position_z": (x, df["position_z"].values) if "position_z" in df else None,
        }
        series = {k: v for k, v in series.items() if v is not None}
        return self.line_plot(
            series,
            "Satellite Position in GCRS",
            "Time (s)",
            "Position (km)",
            "position_GCRS_plotly"
        )

    def plot_lla(self, df: pd.DataFrame) -> go.Figure | None:
        """
        Plot latitude/longitude (left) and altitude (right) vs time.

        Requires: latitude, longitude, altitude columns.
        """
        x = df.index.values
        primary = {}
        if "latitude" in df:
            primary["Latitude"] = (x, df["latitude"].values)
        if "longitude" in df:
            primary["Longitude"] = (x, df["longitude"].values)
        secondary = {}
        if "altitude" in df:
            secondary["Altitude"] = (x, df["altitude"].values)
        return self.twin_axis_line_plot(
            primary, secondary,
            "Satellite Position in LLA",
            "Time (s)",
            "Latitude/Longitude (deg)",
            "Altitude (km)",
            "lla_plotly"
        )

    def plot_magnetic_field_sb(self, df: pd.DataFrame) -> go.Figure | None:
        """
        Plot magnetic field components in SB vs time (nT).

        Requires: magnetic_field_sb_x/y/z columns.
        """
        x = df.index.values
        series = {}
        if "magnetic_field_sb_x" in df:
            series["SB X"] = (x, df["magnetic_field_sb_x"].values)
        if "magnetic_field_sb_y" in df:
            series["SB Y"] = (x, df["magnetic_field_sb_y"].values)
        if "magnetic_field_sb_z" in df:
            series["SB Z"] = (x, df["magnetic_field_sb_z"].values)
        return self.line_plot(
            series,
            "Satellite Magnetic Field in SB",
            "Time (s)",
            "Magnetic Field (nT)",
            "magnetic_field_sb_plotly"
        )

    def plot_magnetic_field_eci(self, df: pd.DataFrame) -> go.Figure | None:
        """
        Plot magnetic field components in ECI vs time (nT).

        Requires: magnetic_field_eci_x/y/z columns.
        """
        x = df.index.values
        series = {}
        if "magnetic_field_eci_x" in df:
            series["ECI X"] = (x, df["magnetic_field_eci_x"].values)
        if "magnetic_field_eci_y" in df:
            series["ECI Y"] = (x, df["magnetic_field_eci_y"].values)
        if "magnetic_field_eci_z" in df:
            series["ECI Z"] = (x, df["magnetic_field_eci_z"].values)
        return self.line_plot(
            series,
            "Satellite Magnetic Field in ECI",
            "Time (s)",
            "Magnetic Field (nT)",
            "magnetic_field_eci_plotly"
        )

    def plot_angular_velocity(self, df: pd.DataFrame) -> go.Figure | None:
        """
        Plot angular velocity components and magnitude vs time (deg/s).

        Requires: angular_velocity_x/y/z columns.
        """
        x = df.index.values
        cols = ["angular_velocity_x", "angular_velocity_y", "angular_velocity_z"]
        labels = ["wx", "wy", "wz"]
        series = {l: (x, df[c].values) for c, l in zip(cols, labels) if c in df}
        if all(c in df for c in cols):
            wmag = np.sqrt(
                df[cols[0]]**2 + df[cols[1]]**2 + df[cols[2]]**2
            ).values if hasattr(df[cols[0]], "values") else np.sqrt(
                df[cols[0]]**2 + df[cols[1]]**2 + df[cols[2]]**2
            )
            series["|w|"] = (x, wmag)
        return self.line_plot(
            series,
            "Satellite Angular Velocity",
            "Time (s)",
            "deg/s",
            "angular_velocity_plotly"
        )

    def plot_euler_angles(self, df: pd.DataFrame) -> go.Figure | None:
        """
        Plot Euler angles roll/pitch/yaw vs time (degrees).

        Requires: euler_angles_x1/y1/z1 columns.
        """
        x = df.index.values
        series = {}
        if "euler_angles_x1" in df:
            series["roll (Phi)"] = (x, df["euler_angles_x1"].values)
        if "euler_angles_y1" in df:
            series["pitch (Theta)"] = (x, df["euler_angles_y1"].values)
        if "euler_angles_z1" in df:
            series["yaw (Psi)"] = (x, df["euler_angles_z1"].values)
        return self.line_plot(
            series,
            "Satellite Euler Angles",
            "Time (s)",
            "deg",
            "euler_angles_plotly"
        )

    def plot_torque(self, df: pd.DataFrame) -> go.Figure | None:
        """
        Plot torque components and magnitude vs time (N·m).

        Requires: torque_x/y/z columns.
        """
        x = df.index.values
        cols = ["torque_x", "torque_y", "torque_z"]
        labels = ["Torque X", "Torque Y", "Torque Z"]
        series = {l: (x, df[c].values) for c, l in zip(cols, labels) if c in df}
        if all(c in df for c in cols):
            tmag = np.sqrt(
                df[cols[0]]**2 + df[cols[1]]**2 + df[cols[2]]**2
            ).values if hasattr(df[cols[0]], "values") else np.sqrt(
                df[cols[0]]**2 + df[cols[1]]**2 + df[cols[2]]**2
            )
            series["|Torque|"] = (x, tmag)
        return self.line_plot(
            series,
            "Magnetorquer Applied Torque",
            "Time (s)",
            "N·m",
            "torque_plotly"
        )

    def plot_angular_acceleration(self, df: pd.DataFrame) -> go.Figure | None:
        """
        Plot angular acceleration components and magnitude vs time (deg/s²).

        Requires: angular_acceleration_x/y/z columns (magnitude only if all exist).
        """
        x = df.index.values
        series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if "angular_acceleration_x" in df:
            series["Alpha X"] = (x, df["angular_acceleration_x"].values)
        if "angular_acceleration_y" in df:
            series["Alpha Y"] = (x, df["angular_acceleration_y"].values)
        if "angular_acceleration_z" in df:
            series["Alpha Z"] = (x, df["angular_acceleration_z"].values)
        # Magnitude only if all components exist
        cols = [
            "angular_acceleration_x",
            "angular_acceleration_y",
            "angular_acceleration_z"
        ]
        if all(c in df for c in cols):
            amag = np.sqrt(
                df[cols[0]]**2 + df[cols[1]]**2 + df[cols[2]]**2
            ).values
            series["|Alpha|"] = (x, amag)
        return self.line_plot(
            series,
            "Satellite Angular Acceleration",
            "Time (s)",
            "deg/s²",
            "angular_acceleration_plotly"
        )

    def plot_pointing_error(self, df: pd.DataFrame) -> go.Figure | None:
        """
        Plot pointing error vs time as a scatter (degrees).

        Requires: pointing_error column.
        """
        x = df.index.values
        series = {}
        if "pointing_error" in df:
            series["Pointing Error"] = (x, df["pointing_error"].values)
        return self.scatter_plot(
            series,
            "Satellite Pointing Error",
            "Time (s)",
            "deg",
            "pointing_error_plotly",
            mode="markers"
        )

    def plot_sun_vector_eci(self, df: pd.DataFrame) -> go.Figure | None:
        """
        Plot sun vector components in ECI vs time.

        Requires: sun_vector_eci_x/y/z columns.
        """
        x = df.index.values
        series = {}
        if "sun_vector_eci_x" in df:
            series["ECI X"] = (x, df["sun_vector_eci_x"].values)
        if "sun_vector_eci_y" in df:
            series["ECI Y"] = (x, df["sun_vector_eci_y"].values)
        if "sun_vector_eci_z" in df:
            series["ECI Z"] = (x, df["sun_vector_eci_z"].values)
        return self.line_plot(
            series,
            "Satellite Sun Vector in ECI",
            "Time (s)",
            "Sun Vector (ECI)",
            "sun_vector_eci_plotly"
        )

    def plot_sun_vector_sb(self, df: pd.DataFrame) -> go.Figure | None:
        """
        Plot sun vector components in SB vs time.

        Requires: sun_vector_sb_x/y/z columns.
        """
        x = df.index.values
        series = {}
        if "sun_vector_sb_x" in df:
            series["SB X"] = (x, df["sun_vector_sb_x"].values)
        if "sun_vector_sb_y" in df:
            series["SB Y"] = (x, df["sun_vector_sb_y"].values)
        if "sun_vector_sb_z" in df:
            series["SB Z"] = (x, df["sun_vector_sb_z"].values)
        return self.line_plot(
            series,
            "Satellite Sun Vector in SB",
            "Time (s)",
            "Sun Vector (SB)",
            "sun_vector_sb_plotly"
        )

    def basic_plots(self, df: pd.DataFrame, setup=None) -> None:
        """
        Generate a standard set of Plotly plots for the simulation.

        Args:
            df: State DataFrame containing time-indexed telemetry.
            setup: Optional setup object used to fetch planet radius for orbit plot.
        """
        planet_radius = None
        if setup:
            planet_data = getattr(setup, "planet_data", {})
            planet_radius = planet_data.get("R", None)
        self.plot_orbit(df, planet_radius)
        self.plot_position(df)
        self.plot_lla(df)
        self.plot_magnetic_field_sb(df)
        self.plot_magnetic_field_eci(df)
        self.plot_angular_velocity(df)
        self.plot_euler_angles(df)
        self.plot_torque(df)
        self.plot_angular_acceleration(df)
        self.plot_pointing_error(df)
        self.plot_sun_vector_eci(df)
        self.plot_sun_vector_sb(df)


class LivePlotlyLine:
    """
    Real-time line plotting with Plotly FigureWidget.
    Call update(t, ys) inside your loop; display appears automatically in notebooks.
    """

    def __init__(
        self,
        labels: list[str],
        title: str,
        xlabel: str,
        ylabel: str,
        window: float | None = None,
        max_points: int = 10000,
        output_dir: Path | None = None
    ):
        """
        Initialize a live-updating FigureWidget with multiple series.

        Args:
            labels: Names for each Y-series line.
            title: Plot title.
            xlabel: X axis label.
            ylabel: Y axis label.
            window: Optional rolling window width in X units; if set, only the
                last window of data is displayed.
            max_points: Hard cap on stored points per series (for memory bound).
            output_dir: Directory where finish() writes the HTML file.
        """
        self.fig = go.FigureWidget()
        self.window = window
        self.max_points = max_points
        # where to save HTML by default
        self.output_dir = (
            output_dir or (Path(__file__).resolve().parent / "plots")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._xs: list[float] = []
        self._ys: list[list[float]] = [[] for _ in labels]
        for lbl in labels:
            self.fig.add_scatter(mode="lines", name=lbl)
        self.fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel
        )
        try:
            from IPython.display import display
            display(self.fig)
        except Exception:
            pass

    def update(self, t: float, ys: list[float] | tuple[float, ...]) -> None:
        """
        Append a new sample and redraw the visible data window.

        Args:
            t: New X value (e.g., time in seconds).
            ys: Iterable of Y values, one per series defined at construction.
        """
        self._xs.append(t)
        for i, v in enumerate(ys):
            self._ys[i].append(float(v))
        # enforce max_points
        if len(self._xs) > self.max_points:
            cut = len(self._xs) - self.max_points
            self._xs = self._xs[cut:]
            self._ys = [series[cut:] for series in self._ys]
        # rolling window
        xs = self._xs
        if self.window is not None and len(xs) >= 2:
            left = xs[-1] - self.window
            start = next((i for i, xv in enumerate(xs) if xv >= left), 0)
            xs = xs[start:]
            ys_trim = [series[start:] for series in self._ys]
        else:
            ys_trim = self._ys
        with self.fig.batch_update():
            for i, tr in enumerate(self.fig.data):
                tr.x = xs
                tr.y = ys_trim[i]

    def finish(self, filename: str | None = None) -> None:
        """
        Save the current live plot to an HTML file (if a filename is provided).

        Args:
            filename: Base filename (without extension). When None, nothing is saved.
        """
        if filename:
            out = self.output_dir / f"{filename}.html"
            self.fig.write_html(str(out), include_plotlyjs="cdn")
