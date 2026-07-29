import itertools
from typing import Any, Literal

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from uncertainties import ufloat, unumpy

from pwassist.plotting.base import BasePWAPlotter


class ScanPlotter(BasePWAPlotter):
    """Plotter for all results that scan across a range of bins, e.g. mass, t, etc."""

    def coherent_sum(
        self,
        sum_label: str,
        data_legend: str = "GlueX-I Data",
        indices: list[int] | None = None,
        ax: matplotlib.axes.Axes | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> matplotlib.axes.Axes:
        """Plot coherent sum group across the mass bins with data points.

        A sum group is defined by the quantum numbers that the sum groups and the
        amplitude naming scheme. For example, in the `JLme` scheme, one can plot all
        the available `JLe` coherent sums, or those that sum over the spin-projection
        `m`.

        Args:
            sum_label (str): The label of the coherent sum group to plot. See the
                `coherent_sums` member of the `Results` class for a list of available
                sum groups and the amplitudes that belong to each group.
            data_legend (str): The legend label for the data points. Defaults to
                "GlueX-I Data".
            indices (list[int] | None): Optional list of indices to select specific mass
                bins. If None, all bins will be plotted.
            ax (matplotlib.axes.Axes | None): Optional axes to plot on. If None, a new
                figure and axes will be created.
            kwargs (dict[str, Any] | None): Optional dictionary of keyword arguments
                to customize the plot appearance.

        Raises:
            KeyError: If the specified sum_label is not found in the coherent sums.
                Prints available sum labels.
        """
        if sum_label not in self.results.coherent_sums:
            raise KeyError(
                f"Sum label '{sum_label}' not found in coherent sums."
                f" Available sum labels: {list(self.results.coherent_sums.keys())}"
            )

        coherent_sums = self.results.coherent_sums[sum_label]
        fit_df, data_df = self._coherent_sum_dataframes(coherent_sums, indices)

        # default to Dark2 colormap, and cycle if more columns than colors
        colors = plt.get_cmap("Dark2").colors  # type: ignore
        colors = list(itertools.islice(itertools.cycle(colors), len(coherent_sums)))

        kwargs = {
            "marker": ["." for _ in range(len(coherent_sums))],
            "linestyle": ["" for _ in range(len(coherent_sums))],
            "alpha": [0.7 for _ in range(len(coherent_sums))],
            "colors": colors,
        }
        kwargs.update(kwargs or {})

        with self._style():
            fig, ax = (
                plt.subplots(layout="constrained")
                if ax is None
                else (ax.get_figure(), ax)
            )

            # plot the data points with error bars, using the appropriate events column
            # based on whether the results are acceptance-corrected or not
            if self.results.is_acc_corrected:
                data_points = unumpy.uarray(
                    data_df["ac_events"], data_df["ac_events_err"]
                )
            else:
                data_points = unumpy.uarray(data_df["events"], data_df["events_err"])

            ax.errorbar(
                x=data_df["m_center"],
                xerr=data_df["bin_width"] / 2.0,
                y=unumpy.nominal_values(data_points),
                yerr=unumpy.std_devs(data_points),
                label=data_legend,
                marker=".",
                linestyle="",
                color="black",
            )

            # plot each coherent sum with error bars
            for sum_idx, coh_sum in enumerate(coherent_sums):
                label = self.results.parser.sum_to_latex(sum_label, coh_sum)

                ax.errorbar(
                    x=data_df["m_center"],
                    xerr=data_df["bin_width"] / 2.0,
                    y=fit_df[coh_sum],
                    yerr=fit_df[f"{coh_sum}_err"],
                    label=label,
                    marker=kwargs["marker"][sum_idx],
                    linestyle=kwargs["linestyle"][sum_idx],
                    alpha=kwargs["alpha"][sum_idx],
                    color=kwargs["colors"][sum_idx],
                )

            ax.set_xlabel(r"Mass $(GeV)$")
            ax.set_ylabel(rf"Events / {data_df['bin_width'].mean():.3f} GeV")
            ax.set_ylim(bottom=0)
            ax.legend()

        return ax

    def _coherent_sum_dataframes(
        self, columns: tuple[str, ...], indices: list[int] | None = None
    ) -> tuple[pd.DataFrame | pd.Series, pd.DataFrame]:
        """Prepare the dataframes for the coherent_sum plot

        Args:
            columns (list[str]): The list of column names to include in the dataframes.
            indices (list[int] | None): Optional list of indices to select specific mass
                bins.
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the fit dataframe and
                the data dataframe for the specified coherent sum group and indices.
        Todo:
            - this can potentially be generalized to other bin plotters
        """

        # bootstrap fits replace fit errors, if available
        fit_columns = list(columns)
        if self.results.bootstrap is not None:
            raise NotImplementedError(
                "Replacing column errors by bootstrap std() not yet supported"
            )
        else:
            fit_columns.extend([f"{col}_err" for col in fit_columns])

        fit_df = (
            self.results.fit.loc[indices, fit_columns]
            if indices is not None
            else self.results.fit[fit_columns]
        )

        data_columns = [
            "m_center",
            "m_low",
            "m_high",
            "events",
            "events_err",
            "ac_events",
            "ac_events_err",
        ]

        data_df = (
            self.results.data.loc[indices, data_columns]
            if indices is not None
            else self.results.data[data_columns]
        ).copy()
        data_df["bin_width"] = data_df["m_high"] - data_df["m_low"]

        return fit_df, data_df

    def amplitudes(
        self,
        fractional: bool = False,
        sharey: bool = False,
        reflectivity: Literal["positive", "negative", "all"] = "all",
        indices: list[int] | None = None,
        axs: np.ndarray | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Plot a grid of amplitudes, organized by spin (rows) and projection (columns)

        This plot will create a grid of all amplitudes plotted as a function of mass,
        with the rows corresponding to the spin+parity+angular momenta combo (J^P L) and
        the columns corresponding to the spin projection m. Any quantum numbers for the
        label not available due to the naming scheme are simply dropped. Reflectivities
        are plotted together in the same plot, with the option to select only positive
        or negative reflectivities.

        Args:
            fractional (bool, optional): Plot as fit fractions of the total intensity.
                Defaults to False.
            sharey (bool, optional): Share the y-axis across all subplots. Defaults to
                False. Note that specifying custom axes will override this option.
            reflectivity (Literal['positive', 'negative', 'all'], optional): The type of
                reflectivity to plot. Defaults to "all".
            indices (list[int] | None, optional): Optional list of indices to select
                specific mass bins. If None, all bins will be plotted. Defaults to None.
            axs (np.ndarray | None, optional): The array of axes to plot on. If None, a
                new figure and axes will be created. Note that one must be careful that
                the shape of axes matches the expected shape based on the number of
                amplitudes. Defaults to None.
            kwargs (dict[str, Any] | None, optional): Optional dictionary of keyword
                arguments to customize the plot appearance. Defaults to None.

        Returns:
            np.ndarray: The array of axes objects containing the amplitude plots.
        Raises:
            ValueError: If the reflectivity argument is not one of "positive",
                "negative", or "all".
            IndexError: If the provided axes shape does not match the expected shape
                based on the number of amplitudes.
        """

        if reflectivity not in ["positive", "negative", "all"]:
            raise ValueError(
                f"Invalid reflectivity value: {reflectivity}. Must be one of "
                "'positive', 'negative', or 'all'."
            )

        if axs is None:

            # TODO: parse through individual amplitudes, and determine max value of 'm'
            # Do the same for max "JPL", "JL", or "L" combos. This depends on naming
            # scheme though. Then build grid.
            nrows = 2
            ncols = 2

            fig, axs = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                sharey=sharey,
                figsize=(4 * ncols, 3 * nrows),
                layout="constrained",
            )

        return axs  # type: ignore

    def interference(
        self,
        amp1: str,
        amp2: str,
        indices: list[int] | None = None,
        amp1_kwargs: dict[str, Any] | None = None,
        amp2_kwargs: dict[str, Any] | None = None,
        amp_ax: matplotlib.axes.Axes | None = None,
        phase_ax: matplotlib.axes.Axes | None = None,
    ) -> np.ndarray:
        """Plot two amplitudes and their interference phase as a function of mass.

        Args:
            amp1 (str): The label of the first amplitude to plot.
            amp2 (str): The label of the second amplitude to plot.
            indices (list[int] | None): Optional list of indices to select specific mass
                bins. If None, all bins will be plotted.
            amp1_kwargs (dict[str, Any] | None): Optional dictionary of keyword
                arguments to customize the appearance of the first amplitude plot.
            amp2_kwargs (dict[str, Any] | None): Optional dictionary of keyword
                arguments to customize the appearance of the second amplitude plot.
            amp_ax (matplotlib.axes.Axes | None): Optional axes to plot the amplitudes
                on. If None, a new figure and axes will be created.
            phase_ax (matplotlib.axes.Axes | None): Optional axes to plot the phase
                difference on. If None, a new figure and axes will be created.
        Returns:
            np.ndarray: The array of axes objects containing the amplitude and phase
                difference plots.
        Raises:
            KeyError: If either phase is not in the fit results.
            ValueError: If only one of amp_ax or phase_ax is provided, but not both
        """

        if amp1 not in self.results.fit.columns:
            raise KeyError(
                f"Amplitude '{amp1}' not found in fit results. "
                f"Available amplitudes: {list(self.results.fit.columns)}"
            )
        if amp2 not in self.results.fit.columns:
            raise KeyError(
                f"Amplitude '{amp2}' not found in fit results. "
                f"Available amplitudes: {list(self.results.fit.columns)}"
            )

        if amp_ax is None and phase_ax is None:
            fig, axs = plt.subplots(
                nrows=2,
                ncols=1,
                sharex=True,
                gridspec_kw={"wspace": 0.0, "hspace": 0.07},
                height_ratios=[3, 1],
                layout="constrained",
            )
        elif amp_ax is None or phase_ax is None:
            raise ValueError(
                "Both amp_ax and phase_ax must be provided if one is specified."
            )
        else:
            axs = np.array([amp_ax, phase_ax])

        # TODO: obtain the relevant dataframes, and plot their amplitude and phase
        # differences on the appropriate axes. Use the provided kwargs for
        # customization.

        return axs

    def model_matrix(
        self,
        indices: list[int] | None = None,
        axs: np.ndarray | None = None,
    ) -> np.ndarray:
        """Plot the entire model behavior as a function of mass in a matrix of plots

        The matrix is organized where the amplitudes are plotted on the diagonal,
        grouped by reflectivity (similar to the amplitudes() method), and the
        off-diagonal plots show the interference between the amplitudes. The upper
        triangle is for the positive reflectivity amplitude interferences, and the
        lower triangle for the negative reflectivity amplitude interferences.

        Args:
            indices (list[int] | None): Optional list of indices to select specific mass
                bins. If None, all bins will be plotted.
            axs (np.ndarray | None): Optional array of axes to plot on. If None, a new
                figure and axes will be created. Note that one must be careful that the
                axes shape matches the expected shape.
        Returns:
            np.ndarray: The array of axes objects containing the amplitudes and
                interference plots.
        Raises:
            IndexError: If the provided axes shape does not match the expected shape
                based on the number of amplitudes and reflectivities.
        """

        if axs is None:
            # TODO: determine shape, reference amplitudes() method for guidance.
            nrows = 0
            ncols = 0
            fig, axs = plt.subplots(
                nrows,
                ncols,
                sharey=True,
                figsize=(4 * ncols, 3 * nrows),
                layout="constrained",
            )

        # TODO: make sure to plot errorbars as smooth fill_between due to small plots

        return axs  # type: ignore

    def convergence_rate(
        self,
        indices: list[int] | None = None,
        ax: matplotlib.axes.Axes | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> matplotlib.axes.Axes:
        """Plot the convergence rate of the fit across the mass bins.

        Requires randomized fits to be in the results. Plots the percentage of
        successful, failed, and converged-with-bad-error-matrix fits across the mass
        bins. This can be useful for diagnosing issues with the fit and understanding
        the stability of the fit across the mass range.

        Args:
            indices (list[int] | None): Optional list of indices to select specific mass
                bins. If None, all bins will be plotted.
            ax (matplotlib.axes.Axes | None): Optional axes to plot on. If None, a new
                figure and axes will be created.
            kwargs (dict[str, Any] | None): Optional dictionary of keyword arguments
                to customize the plot appearance.
        Returns:
            matplotlib.axes.Axes: The axes object containing the convergence rate plot.
        Raises:
            KeyError: If randomized fits are not available in the results.
        """

        if self.results.randomized is None:
            raise KeyError("Randomized fits are required to plot convergence rate.")

        if ax is None:
            fig, ax = plt.subplots(
                layout="constrained",
            )

        # TODO: determine each rate as percentage of total fits, and plot as stacked
        # bar chart with appropriate labels and legend.

        return ax

    def ridgeline(
        self,
        columns: list[str],
        indices: list[int] | None = None,
    ) -> np.ndarray:
        """Create a ridgeline plot of the columns from the bootstrap distributions

        Args:
            columns (list[str]): _description_
            indices (list[int] | None, optional): _description_. Defaults to None.

        Returns:
            np.ndarray: _description_

        Raises:
            KeyError: If the specified columns are not found in the bootstrap dataframe,
                or if the bootstrap dataframe is not available in the results.
        """

        if self.results.bootstrap is None:
            raise KeyError(
                "Bootstrap distributions are required to create a ridgeline plot."
            )

        for col in columns:
            if col not in self.results.bootstrap.columns:
                raise KeyError(
                    f"Column '{col}' not found in bootstrap distributions. "
                    f"Available columns: {list(self.results.bootstrap.columns)}"
                )

        # TODO: replace with joypy.joyplot, ridgeplot, seaborn, or similar library
        fig, axs = plt.subplots(2, 2)

        return axs
